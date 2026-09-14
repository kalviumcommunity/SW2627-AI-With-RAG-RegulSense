"""Unit tests for RegulSense Vector Database Retrieval & Similarity Search Module.

Validates:
1. Task 1: Embedding user query via OpenAI-compatible endpoint with dimension 384.
2. Task 2: Running top-k similarity search against ChromaDB collection.
3. Task 3: Inclusion of similarity scores, raw distance, source text, and provenance metadata.
4. Task 4: Demonstrating changing k (k=2 vs. k=5) with result preservation and score boundaries.
5. Task 5: Validating exported Markdown and JSON reports for sample query results.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.retriever import (
    DEFAULT_COMPLIANCE_QUERY,
    KVariationDemonstration,
    RetrievalRunResult,
    VectorRetriever,
    run_sample_query_retrieval,
)
from src.vector_db import (
    RetrievedRecord,
    VectorDatabaseManager,
    VectorRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVectorRetrieval(unittest.TestCase):
    """Test suite for vector database similarity retrieval and k variation analysis."""

    def setUp(self):
        """Initializes an ephemeral in-memory ChromaDB manager and mock records for isolated testing."""
        self.vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name="test_retrieval_collection",
            vector_dimension=384,
            distance_metric="cosine",
        )

        # Seed in-memory collection with 6 diverse synthetic regulatory chunks
        self.mock_records = []
        for i in range(6):
            # Create vectors with controlled directional similarity
            # Record 0 and 1 are very close to [1.0, 0.0, ...]
            vec = [0.0] * 384
            vec[0] = 1.0 - (i * 0.15)
            vec[1] = 0.1 * (i + 1)
            # Normalize vector
            norm = np.linalg.norm(vec)
            norm_vec = [float(x / norm) for x in vec]

            rec = VectorRecord(
                id=f"test_regulatory_circular_chunk_{i+1:03d}",
                embedding=norm_vec,
                document=f"Regulatory statutory rule clause {i+1} covering banking compliance.",
                metadata={
                    "source_document": f"circular_rbi_2024_{i//2}.txt",
                    "filename": f"circular_rbi_2024_{i//2}.txt",
                    "chunk_index": i,
                    "section": f"Section {i+1}: CDD and Risk Categorization",
                    "page_number": 1 + (i // 3),
                    "token_count": 50 + (i * 10),
                    "document_id": f"circular_rbi_2024_{i//2}",
                    "file_type": ".txt",
                },
            )
            self.mock_records.append(rec)

        self.vdb.insert_records(self.mock_records)

        # Mock OpenAI client that returns a deterministic 384-dim query vector
        self.mock_openai_client = MagicMock()
        mock_embedding_response = MagicMock()
        mock_data = MagicMock()
        query_vec = [0.0] * 384
        query_vec[0] = 1.0  # Closest to chunk 0 and chunk 1
        mock_data.embedding = query_vec
        mock_embedding_response.data = [mock_data]
        self.mock_openai_client.embeddings.create.return_value = mock_embedding_response

        self.retriever = VectorRetriever(
            vector_db=self.vdb,
            collection_name="test_retrieval_collection",
            openai_client=self.mock_openai_client,
        )

    # -------------------------------------------------------------------------
    # Task 1: Embed User Query
    # -------------------------------------------------------------------------

    def test_task1_embed_query_success(self):
        """Confirm user query is embedded with exact expected dimensionality (384)."""
        vec = self.retriever.embed_query("What are the KYC due diligence requirements?")
        self.assertEqual(len(vec), 384)
        self.mock_openai_client.embeddings.create.assert_called_once()

    def test_task1_embed_query_empty_raises_value_error(self):
        """Confirm empty query text is rejected with ValueError."""
        with self.assertRaises(ValueError):
            self.retriever.embed_query("   ")

    def test_task1_embed_query_dimension_mismatch_raises_value_error(self):
        """Confirm mismatched vector dimensions from API client raises ValueError."""
        bad_client = MagicMock()
        bad_resp = MagicMock()
        bad_data = MagicMock()
        bad_data.embedding = [0.1] * 128  # 128 dims instead of 384
        bad_resp.data = [bad_data]
        bad_client.embeddings.create.return_value = bad_resp

        r = VectorRetriever(
            vector_db=self.vdb,
            openai_client=bad_client,
        )
        with self.assertRaises(ValueError):
            r.embed_query("Sample query")

    # -------------------------------------------------------------------------
    # Task 2 & 3: Run Top-k Similarity Search with Scores and Metadata
    # -------------------------------------------------------------------------

    def test_task2_and_task3_retrieve_top_k_with_scores_and_metadata(self):
        """Confirm retrieve returns top-k chunks with similarity scores, distances, and metadata."""
        result = self.retriever.retrieve(
            query_text="What are the KYC due diligence requirements?",
            top_k=3,
        )

        self.assertIsInstance(result, RetrievalRunResult)
        self.assertEqual(result.k, 3)
        self.assertEqual(result.retrieved_count, 3)
        self.assertEqual(len(result.chunks), 3)

        # Confirm scores and rank ordering (Task 3)
        prev_score = 1.1
        for idx, chunk in enumerate(result.chunks, start=1):
            self.assertIsInstance(chunk, RetrievedRecord)
            self.assertEqual(chunk.rank, idx)
            # Scores must be bounded within [-1.0, 1.0]
            self.assertGreaterEqual(chunk.similarity_score, -1.0)
            self.assertLessEqual(chunk.similarity_score, 1.0)
            # Scores must be monotonically descending
            self.assertLessEqual(chunk.similarity_score, prev_score)
            prev_score = chunk.similarity_score

            # Check distance and score consistency (score = 1 - distance)
            self.assertAlmostEqual(chunk.similarity_score, 1.0 - chunk.distance, places=4)

            # Check verbatim source text
            self.assertIn("Regulatory statutory rule clause", chunk.document)

            # Check provenance metadata (Task 3)
            self.assertIn("source_document", chunk.metadata)
            self.assertIn("chunk_index", chunk.metadata)
            self.assertIn("section", chunk.metadata)
            self.assertIn("page_number", chunk.metadata)

    def test_task2_top_k_exceeding_collection_size_handled_gracefully(self):
        """Confirm top_k larger than collection count returns all available items without error."""
        result = self.retriever.retrieve(
            query_text="Sample compliance search",
            top_k=50,  # Only 6 items exist
        )
        self.assertEqual(result.retrieved_count, 6)

    def test_task2_invalid_top_k_raises_value_error(self):
        """Confirm top_k <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            self.retriever.retrieve("Valid query", top_k=0)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("Valid query", top_k=-2)

    # -------------------------------------------------------------------------
    # Task 4: Demonstrate Changing k
    # -------------------------------------------------------------------------

    def test_task4_demonstrate_k_variation_comparison(self):
        """Confirm changing k (k=2 vs k=5) expands result set while preserving top results."""
        demo = self.retriever.demonstrate_k_variation(
            query_text="KYC customer due diligence obligations",
            k_values=(2, 5),
        )

        self.assertIsInstance(demo, KVariationDemonstration)
        self.assertIn(2, demo.k_runs)
        self.assertIn(5, demo.k_runs)

        run_k2 = demo.k_runs[2]
        run_k5 = demo.k_runs[5]

        self.assertEqual(run_k2.retrieved_count, 2)
        self.assertEqual(run_k5.retrieved_count, 5)

        # Top 2 results in k=5 must match k=2 results exactly in identity and rank
        k2_ids = [c.id for c in run_k2.chunks]
        k5_top2_ids = [c.id for c in run_k5.chunks[:2]]
        self.assertEqual(k2_ids, k5_top2_ids)

        # Common chunks must be 2, unique to larger k must be 3
        self.assertEqual(len(demo.common_chunk_ids), 2)
        self.assertEqual(len(demo.unique_to_larger_k), 3)

        # Score degradation: lowest score in k=5 should be <= lowest score in k=2
        self.assertLessEqual(run_k5.lowest_score, run_k2.lowest_score)

        # Token count expands with higher k
        self.assertGreater(run_k5.total_tokens_retrieved, run_k2.total_tokens_retrieved)

    # -------------------------------------------------------------------------
    # Task 5: Export Retrieval Artifacts
    # -------------------------------------------------------------------------

    def test_task5_export_retrieval_artifacts(self):
        """Verify Markdown and JSON retrieval reports are exported with valid schema."""
        demo = self.retriever.demonstrate_k_variation(
            query_text=DEFAULT_COMPLIANCE_QUERY,
            k_values=(2, 4),
        )

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            md_path = Path(temp_dir) / "retrieval_results.md"
            json_path = Path(temp_dir) / "retrieval_results.json"

            self.retriever.export_retrieval_artifacts(
                demo=demo,
                output_markdown_path=md_path,
                output_json_path=json_path,
            )

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            # Validate JSON content
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("query_text", data)
            self.assertIn("runs", data)
            self.assertIn("2", data["runs"])
            self.assertIn("4", data["runs"])
            self.assertEqual(len(data["runs"]["2"]["chunks"]), 2)
            self.assertEqual(len(data["runs"]["4"]["chunks"]), 4)

            # Check chunk schema in JSON
            c0 = data["runs"]["2"]["chunks"][0]
            self.assertIn("similarity_score", c0)
            self.assertIn("distance", c0)
            self.assertIn("document", c0)
            self.assertIn("metadata", c0)
            self.assertIn("source_document", c0["metadata"])
            self.assertIn("chunk_index", c0["metadata"])

            # Validate Markdown content
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Vector Database Retrieval & Top-k Similarity Search Report", md_text)
            self.assertIn("Top-2 Retrieved Chunks", md_text)
            self.assertIn("Top-4 Retrieved Chunks", md_text)
            self.assertIn("Comparative Analysis: Demonstrating Changing $k$", md_text)

    # -------------------------------------------------------------------------
    # Direct VectorDatabaseManager.query_similarity tests
    # -------------------------------------------------------------------------

    def test_vdb_query_similarity_empty_collection(self):
        """Verify query_similarity returns empty list on an empty collection."""
        empty_vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name="empty_test_collection",
            vector_dimension=384,
        )
        res = empty_vdb.query_similarity(query_vector=[0.1] * 384, top_k=3)
        self.assertEqual(res, [])

    def test_vdb_query_similarity_dimension_guard(self):
        """Verify mismatched query vector dimension raises ValueError."""
        with self.assertRaises(ValueError):
            self.vdb.query_similarity(query_vector=[0.1] * 128, top_k=3)


if __name__ == "__main__":
    unittest.main()
