"""Unit tests for RegulSense Corpus Embedding Generation Pipeline.

Validates:
1. Task 1: Generating embeddings through OpenAI-compatible API on prepared text chunks.
2. Task 2: Storing vectors with source chunk text and complete retrieval metadata (document, index, section, page).
3. Task 3: Reading configuration dynamically from environment variables without hardcoding.
4. Task 4: Reporting verification metrics (chunk counts, vector length, trimmed values).
5. Task 5: Validating exported corpus JSON and Markdown verification artifacts.
"""

import json
import os
from pathlib import Path
import unittest

from src.corpus_embedder import (
    CorpusChunkEmbedder,
    EmbeddedChunkRecord,
    VerificationSummary,
    run_corpus_embedding,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCorpusEmbeddings(unittest.TestCase):
    """Test suite for corpus chunk embedding generation, metadata binding, and verification."""

    @classmethod
    def setUpClass(cls):
        """Run embedding on prepared corpus sample once for test suite."""
        cls.embedder = CorpusChunkEmbedder()
        cls.raw_chunks = cls.embedder.load_prepared_chunks()
        cls.records = cls.embedder.embed_chunks(cls.raw_chunks)
        cls.summary = cls.embedder.verify_embeddings(cls.records)

    # -------------------------------------------------------------------------
    # Task 1: Generate Embeddings Through API
    # -------------------------------------------------------------------------

    def test_task1_generate_embeddings_api_response(self):
        """Confirm API returns embedding vectors with expected dimension for prepared chunks."""
        self.assertGreater(len(self.records), 0)
        self.assertEqual(len(self.records), len(self.raw_chunks))

        expected_dim = 384
        for record in self.records:
            self.assertIsInstance(record.embedding, list)
            self.assertEqual(
                len(record.embedding),
                expected_dim,
                f"Chunk {record.chunk_id} vector dimension mismatch (expected {expected_dim}, got {len(record.embedding)})",
            )
            self.assertEqual(record.vector_length, expected_dim)
            self.assertTrue(all(isinstance(x, float) for x in record.embedding))

    def test_task1_empty_chunk_handling(self):
        """Confirm embedder gracefully returns empty list on empty input."""
        empty_records = self.embedder.embed_chunks([])
        self.assertEqual(empty_records, [])

    # -------------------------------------------------------------------------
    # Task 2: Store Vectors With Source Chunks and Retrieval Metadata
    # -------------------------------------------------------------------------

    def test_task2_vector_and_source_chunk_binding(self):
        """Verify each record stores embedding coupled with source text and essential metadata."""
        for record in self.records:
            # 1. Source text present
            self.assertIsInstance(record.source_text, str)
            self.assertGreater(len(record.source_text), 10, "Source text must not be empty")

            # 2. Retrieval metadata present
            meta = record.metadata
            self.assertIn("filename", meta, "Metadata must specify source document filename")
            self.assertIn("document_id", meta, "Metadata must specify document ID")
            self.assertIn("chunk_index", meta, "Metadata must specify chunk index")
            self.assertIn("section", meta, "Metadata must specify governing section")
            self.assertIn("page_number", meta, "Metadata must specify page number")

            # Verify chunk index is non-negative int
            self.assertIsInstance(meta["chunk_index"], int)
            self.assertGreaterEqual(meta["chunk_index"], 0)

            # Verify section is non-empty string
            self.assertIsInstance(meta["section"], str)
            self.assertGreater(len(meta["section"]), 0)

    # -------------------------------------------------------------------------
    # Task 3: Environment Configuration
    # -------------------------------------------------------------------------

    def test_task3_environment_configuration_loading(self):
        """Verify embedder reads configuration from environment and does not hardcode."""
        env_model = os.getenv("EMBEDDING_MODEL")
        env_base_url = os.getenv("OPENAI_BASE_URL")

        self.assertEqual(self.embedder.model, env_model)
        self.assertEqual(self.embedder.base_url, env_base_url)

    def test_task3_missing_model_raises_value_error(self):
        """Verify initialization fails informatively if model configuration is missing."""
        orig_env = os.environ.get("EMBEDDING_MODEL")
        try:
            if "EMBEDDING_MODEL" in os.environ:
                del os.environ["EMBEDDING_MODEL"]
            with self.assertRaises(ValueError):
                CorpusChunkEmbedder(model=None)
        finally:
            if orig_env is not None:
                os.environ["EMBEDDING_MODEL"] = orig_env

    # -------------------------------------------------------------------------
    # Task 4: Verification Output
    # -------------------------------------------------------------------------

    def test_task4_verification_metrics(self):
        """Confirm verification summary reports total chunks, vector length, and trimmed values."""
        self.assertTrue(self.summary.dimension_uniform)
        self.assertEqual(self.summary.vector_length, 384)
        self.assertEqual(self.summary.total_chunks_embedded, len(self.records))
        self.assertGreaterEqual(self.summary.unique_documents, 4)

        # Trimmed values check
        self.assertEqual(len(self.summary.sample_trimmed_values), 8)
        self.assertTrue(all(isinstance(v, float) for v in self.summary.sample_trimmed_values))

        # Sample preview check
        preview = self.summary.sample_metadata_preview
        self.assertIn("source_document", preview)
        self.assertIn("section", preview)
        self.assertIn("page_number", preview)
        self.assertIn("chunk_index", preview)

    # -------------------------------------------------------------------------
    # Task 5: Exported Corpus Artifacts
    # -------------------------------------------------------------------------

    def test_task5_exported_artifacts_exist_and_valid(self):
        """Verify embedded corpus JSON and Markdown verification artifacts exist and are valid."""
        json_path = PROJECT_ROOT / "outputs" / "embedded_corpus_chunks.json"
        md_path = PROJECT_ROOT / "outputs" / "embedding_generation_verification.md"

        self.embedder.export_embedded_corpus(self.records, json_path, md_path)

        self.assertTrue(json_path.exists(), f"Missing JSON artifact: {json_path}")
        self.assertTrue(md_path.exists(), f"Missing Markdown artifact: {md_path}")

        # Check JSON integrity
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("metadata", data)
        self.assertIn("embedded_chunks", data)
        self.assertEqual(len(data["embedded_chunks"]), len(self.records))

        first_chunk = data["embedded_chunks"][0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("source_text", first_chunk)
        self.assertIn("metadata", first_chunk)
        self.assertIn("vector_length", first_chunk)
        self.assertIn("trimmed_vector", first_chunk)
        self.assertIn("embedding", first_chunk)
        self.assertEqual(len(first_chunk["trimmed_vector"]), 8)
        self.assertEqual(first_chunk["vector_length"], 384)

        # Check Markdown integrity
        md_text = md_path.read_text(encoding="utf-8")
        self.assertIn("Total Chunks Embedded", md_text)
        self.assertIn("Vector Length (Dimension)", md_text)
        self.assertIn("Sample Embedded Chunk & Metadata Linkage", md_text)
        self.assertIn("Detailed Embedded Corpus Ledger", md_text)


if __name__ == "__main__":
    unittest.main()
