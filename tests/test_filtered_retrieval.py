"""Unit tests for RegulSense Metadata-Filtered & Hybrid Retrieval Engine.

Validates:
1. Task 1: Scoping vector retrieval using metadata filters (source_document, file_type, section).
2. Task 2: Comparing filtered vs. unfiltered results and verifying precision improvements.
3. Task 3: Keyword scoring, exact phrase/statutory term detection, and hybrid weighted blending.
4. Task 4: Precision gains and rank promotion demonstrations across benchmark compliance queries.
5. Task 5: Exporting comprehensive Markdown and JSON reports capturing filtered retrieval results.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.filtered_retriever import (
    FilterComparisonResult,
    FilteredRetriever,
    HybridComparisonResult,
    HybridRecord,
    KeywordScorer,
    run_filtered_retrieval_demo,
)
from src.retriever import RetrievalRunResult, VectorRetriever
from src.vector_db import (
    DEFAULT_VECTOR_DIMENSION,
    RetrievedRecord,
    VectorDatabaseManager,
    VectorRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFilteredRetrieval(unittest.TestCase):
    """Test suite for metadata filtering, keyword scoring, hybrid fusion, and precision verification."""

    def setUp(self):
        """Sets up an ephemeral in-memory ChromaDB vector store with synthetic regulatory chunks."""
        self.vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name="test_filtered_retrieval_collection",
            vector_dimension=DEFAULT_VECTOR_DIMENSION,
            distance_metric="cosine",
        )

        # Synthetic corpus with 6 distinct chunks across 3 source documents
        # Doc A: AML/KYC (.txt)
        # Doc B: Digital Lending recovery conduct (.html)
        # Doc C: Cyber Resilience incident reporting (.pdf)
        self.corpus_chunks = [
            VectorRecord(
                id="doc_aml_chunk_001",
                embedding=self._make_unit_vector(0.9, 0.1, 0.0),
                document=(
                    "Reserve Bank of India Master Direction on Know Your Customer (KYC). "
                    "Customer Due Diligence (CDD) procedure for individual bank accounts. "
                    "Requires official valid document (OVD) and PAN verification."
                ),
                metadata={
                    "source_document": "rbi_kyc_master_direction.txt",
                    "filename": "rbi_kyc_master_direction.txt",
                    "file_type": ".txt",
                    "section": "Chapter III: Customer Due Diligence",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 55,
                },
            ),
            VectorRecord(
                id="doc_aml_chunk_002",
                embedding=self._make_unit_vector(0.85, 0.15, 0.05),
                document=(
                    "Video-based Customer Identification Process (V-CIP) consent and facial match. "
                    "The automated facial match system shall achieve a confidence score of not less than 95% "
                    "with liveness check before account activation."
                ),
                metadata={
                    "source_document": "rbi_kyc_master_direction.txt",
                    "filename": "rbi_kyc_master_direction.txt",
                    "file_type": ".txt",
                    "section": "Section 18: Video-based Customer Identification Process (V-CIP)",
                    "page_number": 2,
                    "chunk_index": 1,
                    "token_count": 62,
                },
            ),
            VectorRecord(
                id="doc_lending_chunk_001",
                embedding=self._make_unit_vector(0.2, 0.8, 0.1),
                document=(
                    "Digital Lending Guidelines: Fair Practices Code and conduct of recovery agents. "
                    "Recovery agents shall contact delinquent borrowers only between 08:00 hours and 19:00 hours. "
                    "Harassment, abusive language, or persistent unsolicited calls are strictly prohibited."
                ),
                metadata={
                    "source_document": "digital_lending_guidelines.html",
                    "filename": "digital_lending_guidelines.html",
                    "file_type": ".html",
                    "section": "Section 6: Code of Conduct for Recovery Agents",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 68,
                },
            ),
            VectorRecord(
                id="doc_lending_chunk_002",
                embedding=self._make_unit_vector(0.15, 0.75, 0.2),
                document=(
                    "Digital Lending: Key Fact Statement (KFS) and Annual Percentage Rate (APR) disclosures. "
                    "Regulated Entities must provide a standardized KFS to the borrower prior to loan execution."
                ),
                metadata={
                    "source_document": "digital_lending_guidelines.html",
                    "filename": "digital_lending_guidelines.html",
                    "file_type": ".html",
                    "section": "Section 4: Key Fact Statement and Transparency",
                    "page_number": 1,
                    "chunk_index": 1,
                    "token_count": 58,
                },
            ),
            VectorRecord(
                id="doc_cyber_chunk_001",
                embedding=self._make_unit_vector(0.05, 0.1, 0.9),
                document=(
                    "Cyber Security Framework in Banks: Incident reporting baseline requirements. "
                    "All cyber security incidents of Severity 1 and 2 shall be reported to CERT-In and RBI "
                    "within 6 hours of detection without fail."
                ),
                metadata={
                    "source_document": "cyber_resilience_framework.pdf",
                    "filename": "cyber_resilience_framework.pdf",
                    "file_type": ".pdf",
                    "section": "Annex 1: Incident Reporting Architecture",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 64,
                },
            ),
            VectorRecord(
                id="doc_cyber_chunk_002",
                embedding=self._make_unit_vector(0.1, 0.05, 0.85),
                document=(
                    "Cyber Resilience: Security Operations Centre (SOC) continuous monitoring and threat intelligence. "
                    "Banks must maintain round-the-clock telemetry and quarterly penetration testing."
                ),
                metadata={
                    "source_document": "cyber_resilience_framework.pdf",
                    "filename": "cyber_resilience_framework.pdf",
                    "file_type": ".pdf",
                    "section": "Annex 3: SOC Governance and Telemetry",
                    "page_number": 2,
                    "chunk_index": 1,
                    "token_count": 60,
                },
            ),
        ]
        self.vdb.insert_records(self.corpus_chunks)

        # Mock OpenAI Client for deterministic query embeddings
        self.mock_client = MagicMock()
        self.retriever = FilteredRetriever(
            vector_db=self.vdb,
            collection_name="test_filtered_retrieval_collection",
            openai_client=self.mock_client,
        )

    def _make_unit_vector(self, x: float, y: float, z: float) -> list:
        """Constructs a normalized 384-dimensional unit vector with 3 primary coordinates."""
        vec = [0.0] * DEFAULT_VECTOR_DIMENSION
        vec[0] = x
        vec[1] = y
        vec[2] = z
        norm = np.linalg.norm(vec)
        return [float(v / norm) for v in vec]

    def _mock_query_vector(self, x: float, y: float, z: float):
        """Configures mock OpenAI embeddings response to return target coordinate direction."""
        vec = self._make_unit_vector(x, y, z)
        mock_resp = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = vec
        mock_resp.data = [mock_data]
        self.mock_client.embeddings.create.return_value = mock_resp

    # -------------------------------------------------------------------------
    # Task 1: Add a Metadata Filter
    # -------------------------------------------------------------------------

    def test_task1_metadata_filter_single_attribute(self):
        """Verify metadata filter correctly restricts search results to matching attribute values."""
        self._mock_query_vector(0.3, 0.7, 0.1)  # General lending-leaning query

        # Filter by file_type = ".html"
        results = self.retriever.retrieve_filtered(
            query_text="What are recovery agent conduct rules?",
            filter_criteria={"file_type": ".html"},
            top_k=5,
        )

        self.assertGreater(len(results.chunks), 0)
        self.assertLessEqual(len(results.chunks), 2)  # Only 2 .html chunks exist
        for chunk in results.chunks:
            self.assertEqual(chunk.metadata["file_type"], ".html")
            self.assertEqual(chunk.metadata["source_document"], "digital_lending_guidelines.html")

    def test_task1_metadata_filter_source_document(self):
        """Verify scoping search strictly to a specific source document."""
        self._mock_query_vector(0.1, 0.1, 0.9)

        results = self.retriever.retrieve_filtered(
            query_text="Cyber security incident notification",
            filter_criteria={"source_document": "cyber_resilience_framework.pdf"},
            top_k=3,
        )

        self.assertEqual(len(results.chunks), 2)
        for chunk in results.chunks:
            self.assertEqual(chunk.metadata["source_document"], "cyber_resilience_framework.pdf")

    def test_task1_empty_filter_returns_unrestricted(self):
        """Verify passing None or empty dict for filter returns unrestricted top-k."""
        self._mock_query_vector(0.9, 0.1, 0.0)

        results = self.retriever.retrieve_filtered(
            query_text="KYC guidelines",
            filter_criteria=None,
            top_k=4,
        )
        self.assertEqual(len(results.chunks), 4)

    # -------------------------------------------------------------------------
    # Task 2: Compare Filtered and Unfiltered Results
    # -------------------------------------------------------------------------

    def test_task2_compare_filtered_vs_unfiltered_precision_gain(self):
        """Verify comparing filtered vs. unfiltered results demonstrates noise elimination and precision gain."""
        # Query intentionally placed between lending and aml
        self._mock_query_vector(0.6, 0.6, 0.0)

        comparison = self.retriever.compare_filtered_vs_unfiltered(
            query_text="What rules apply to agent conduct during recovery?",
            filter_criteria={"file_type": ".html"},
            top_k=3,
        )

        self.assertIsInstance(comparison, FilterComparisonResult)
        self.assertEqual(comparison.k, 3)
        self.assertEqual(len(comparison.unfiltered_chunks), 3)
        self.assertEqual(len(comparison.filtered_chunks), 2)

        # In filtered results, all chunks must satisfy the filter
        for c in comparison.filtered_chunks:
            self.assertEqual(c.metadata["file_type"], ".html")

        # In unfiltered results, chunks from other document types are present
        unfiltered_types = {c.metadata["file_type"] for c in comparison.unfiltered_chunks}
        self.assertIn(".txt", unfiltered_types)

        # Check precision calculation
        self.assertEqual(comparison.precision_filtered, 1.0)
        self.assertLess(comparison.precision_unfiltered, 1.0)
        self.assertGreater(comparison.precision_gain_pct, 0.0)
        self.assertIn(".html", comparison.relevance_justification)
        self.assertEqual(comparison.filtered_chunks[0].metadata["source_document"], "digital_lending_guidelines.html")

    # -------------------------------------------------------------------------
    # Task 3: Add Keyword or Hybrid Matching
    # -------------------------------------------------------------------------

    def test_task3_keyword_scorer_tokenization_and_stopwords(self):
        """Verify KeywordScorer extracts clean tokens and discards stop words."""
        raw_text = "What is the mandatory 95% facial-match requirement for V-CIP onboarding?"
        tokens = KeywordScorer.tokenize(raw_text)
        self.assertIn("95%", tokens)
        self.assertIn("v-cip", tokens)
        self.assertIn("facial-match", tokens)

        salient = KeywordScorer.extract_salient_terms(raw_text)
        self.assertNotIn("what", salient)
        self.assertNotIn("is", salient)
        self.assertNotIn("the", salient)
        self.assertNotIn("for", salient)
        self.assertIn("95%", salient)
        self.assertIn("v-cip", salient)

    def test_task3_keyword_scorer_exact_phrase_bonus(self):
        """Verify verbatim exact phrases yield higher keyword scores."""
        query = "Cyber security incidents of Severity 1 and 2"
        matching_doc = (
            "All cyber security incidents of Severity 1 and 2 shall be reported "
            "to CERT-In within 6 hours."
        )
        non_matching_doc = "Banks must implement general cyber security incident monitoring."

        score_match, matches = KeywordScorer.score_document(query, matching_doc)
        score_nomatch, _ = KeywordScorer.score_document(query, non_matching_doc)

        self.assertGreater(score_match, score_nomatch)
        self.assertTrue(any("Exact phrase" in m for m in matches))

    def test_task3_retrieve_hybrid_blending(self):
        """Verify hybrid retrieval blends dense and sparse scores based on alpha parameter."""
        self._mock_query_vector(0.9, 0.1, 0.0)

        # Run with alpha=0.7 (70% dense, 30% sparse)
        results = self.retriever.retrieve_hybrid(
            query_text="What is the 95% facial match requirement for V-CIP?",
            target_terms=["95%", "V-CIP", "facial match"],
            top_k=3,
            alpha=0.7,
        )

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, HybridRecord)
            self.assertGreaterEqual(r.hybrid_score, 0.0)
            self.assertLessEqual(r.hybrid_score, 1.0)
            self.assertGreaterEqual(r.rank, 1)

        # Verify that doc_aml_chunk_002 matched the target exact terms
        top_rec = results[0]
        self.assertIn("doc_aml_chunk_002", [r.id for r in results])
        target_chunk = next(r for r in results if r.id == "doc_aml_chunk_002")
        self.assertIn("95%", target_chunk.exact_matches_found)
        self.assertIn("v-cip", [m.lower() for m in target_chunk.exact_matches_found])

    def test_task3_retrieve_hybrid_invalid_alpha(self):
        """Verify invalid alpha values raise ValueError."""
        with self.assertRaises(ValueError):
            self.retriever.retrieve_hybrid("query", alpha=1.5)
        with self.assertRaises(ValueError):
            self.retriever.retrieve_hybrid("query", alpha=-0.1)

    # -------------------------------------------------------------------------
    # Task 4: Demonstrate Improved Precision
    # -------------------------------------------------------------------------

    def test_task4_compare_dense_vs_hybrid_rank_promotion(self):
        """Verify that hybrid search promotes a chunk matching exact terms even if vector similarity is second."""
        # Query aligned slightly more towards chunk 001 than chunk 002
        self._mock_query_vector(0.95, 0.05, 0.0)

        comparison = self.retriever.compare_dense_vs_hybrid(
            query_text="facial match 95% liveness check",
            target_terms=["95%", "facial match", "liveness"],
            top_k=2,
            alpha=0.5,  # Equal weight to exact keywords
        )

        self.assertIsInstance(comparison, HybridComparisonResult)
        self.assertEqual(comparison.alpha, 0.5)
        self.assertEqual(len(comparison.dense_ranking), 2)
        self.assertEqual(len(comparison.hybrid_ranking), 2)

        # Chunk 002 contains 95%, facial match, and liveness - should be Rank #1 in hybrid
        self.assertEqual(comparison.hybrid_ranking[0].id, "doc_aml_chunk_002")
        self.assertIn("doc_aml_chunk_002", comparison.analysis)

    def test_task4_precision_demonstration_suite(self):
        """Verify the full 4-case precision demonstration benchmark runs completely."""
        self._mock_query_vector(0.5, 0.5, 0.5)

        suite_results = self.retriever.demonstrate_precision_improvement()
        self.assertIn("case1_document_filtering", suite_results)
        self.assertIn("case2_cyber_scoping", suite_results)
        self.assertIn("case3_vcip_hybrid_exact", suite_results)
        self.assertIn("case4_circular_code_hybrid", suite_results)

        # Verify case 1 and 2 are FilterComparisonResult
        self.assertIsInstance(suite_results["case1_document_filtering"], FilterComparisonResult)
        self.assertIsInstance(suite_results["case2_cyber_scoping"], FilterComparisonResult)

        # Verify case 3 and 4 are HybridComparisonResult
        self.assertIsInstance(suite_results["case3_vcip_hybrid_exact"], HybridComparisonResult)
        self.assertIsInstance(suite_results["case4_circular_code_hybrid"], HybridComparisonResult)

    # -------------------------------------------------------------------------
    # Task 5: Commit Sample Filtered-Search Results (Artifact Verification)
    # -------------------------------------------------------------------------

    def test_task5_export_filtered_search_artifacts(self):
        """Verify exported Markdown and JSON reports have expected structure and content."""
        self._mock_query_vector(0.5, 0.5, 0.5)
        benchmark_results = self.retriever.demonstrate_precision_improvement()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_md = Path(tmp_dir) / "test_filtered_results.md"
            tmp_json = Path(tmp_dir) / "test_filtered_results.json"

            out_md, out_json = self.retriever.export_filtered_search_artifacts(
                benchmark_results=benchmark_results,
                output_markdown_path=tmp_md,
                output_json_path=tmp_json,
            )

            self.assertTrue(out_md.exists())
            self.assertTrue(out_json.exists())

            # Verify JSON structure
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("timestamp", data)
            self.assertIn("collection_name", data)
            self.assertEqual(data["embedding_model"], "all-minilm")
            self.assertEqual(data["vector_dimension"], 384)
            self.assertIn("cases", data)
            self.assertIn("case1_document_filtering", data["cases"])
            self.assertIn("case3_vcip_hybrid_exact", data["cases"])

            # Verify Markdown structure
            content = out_md.read_text(encoding="utf-8")
            self.assertIn("# RegulSense: Metadata-Filtered & Hybrid Vector Retrieval Demonstration", content)
            self.assertIn("Benchmark Case 1: Scoped Document Type Filtering", content)
            self.assertIn("Benchmark Case 3: Hybrid Search with Exact Threshold Boosting", content)
            self.assertIn("Side-by-Side Comparison: Unfiltered vs. Filtered", content)
            self.assertIn("Dense vs. Hybrid Ranking Ledger", content)


if __name__ == "__main__":
    unittest.main()
