"""Unit tests for RegulSense Vector Database Module.

Validates:
1. Task 1: Vector database setup, client reachability, and heartbeat diagnostics.
2. Task 2: Correctly sized collection creation with calibrated vector dimension and cosine metric.
3. Task 3: VectorRecord schema enforcement, metadata sanitization, and serialization.
4. Task 4: Record insertion, readback round-trip fidelity, vector coordinate preservation, and readback verification.
5. Task 5: Existence and validity of exported setup report Markdown and JSON artifacts.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.vector_db import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_VECTOR_DIMENSION,
    ReadbackVerificationResult,
    VectorDatabaseManager,
    VectorRecord,
    run_vector_db_setup_and_readback,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVectorDatabase(unittest.TestCase):
    """Test suite for ChromaDB setup, collection sizing, record schema, and readback verification."""

    def setUp(self):
        """Initializes an ephemeral in-memory vector database manager for isolated testing."""
        self.vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name="test_regulsense_collection",
            vector_dimension=384,
            distance_metric="cosine",
        )
        self.sample_embedding = [float(0.001 * (i + 1)) for i in range(384)]
        self.sample_record = VectorRecord(
            id="test_circular_dor_2024_108_chunk_001",
            embedding=self.sample_embedding,
            document="Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship.",
            metadata={
                "source_document": "circular_dor_2024_108.txt",
                "document_id": "circular_dor_2024_108",
                "section": "2. Customer Due Diligence (CDD) Requirements",
                "page_number": 1,
                "chunk_index": 0,
                "token_count": 22,
                "file_type": ".txt",
            },
        )

    # -------------------------------------------------------------------------
    # Task 1: Vector Database Setup & Reachability
    # -------------------------------------------------------------------------

    def test_task1_database_is_reachable_and_heartbeat_valid(self):
        """Confirm vector database client is reachable and returns a valid heartbeat."""
        self.assertTrue(self.vdb.is_reachable(), "Vector database client must be reachable")
        hb = self.vdb.get_heartbeat()
        self.assertIsInstance(hb, int)
        self.assertGreater(hb, 0, "Heartbeat timestamp must be a positive integer")

    def test_task1_persistent_client_initialization(self):
        """Confirm persistent ChromaDB client initializes properly in a temporary directory."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            p_vdb = VectorDatabaseManager(
                persist_directory=temp_dir,
                in_memory=False,
            )
            self.assertTrue(p_vdb.is_reachable())
            self.assertEqual(p_vdb.persist_directory, Path(temp_dir))
            self.assertTrue(Path(temp_dir).exists())

    # -------------------------------------------------------------------------
    # Task 2: Create a Correctly Sized Collection
    # -------------------------------------------------------------------------

    def test_task2_collection_created_with_correct_dimension_and_metric(self):
        """Verify collection is created with configured vector dimension and cosine distance metric."""
        collection = self.vdb.get_or_create_collection()
        self.assertEqual(collection.name, "test_regulsense_collection")

        meta = collection.metadata
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("hnsw:space"), "cosine")
        self.assertEqual(meta.get("dimension"), 384)
        self.assertIn("all-minilm", meta.get("embedding_model", ""))

    def test_task2_dimension_mismatch_raises_value_error(self):
        """Verify inserting a vector with mismatched dimensions raises ValueError."""
        mismatched_vector = [0.1] * 128  # 128 dims instead of 384
        invalid_record = VectorRecord(
            id="bad_dim_chunk",
            embedding=mismatched_vector,
            document="Test content",
            metadata={"source": "test.txt"},
        )
        with self.assertRaises(ValueError):
            self.vdb.insert_record(invalid_record)

    # -------------------------------------------------------------------------
    # Task 3: Stored Record Schema Design
    # -------------------------------------------------------------------------

    def test_task3_record_schema_fields_and_sanitization(self):
        """Verify VectorRecord enforces required fields and sanitizes complex metadata."""
        complex_metadata = {
            "source_document": "sample.pdf",
            "page_number": 3,
            "is_active": True,
            "nested_list": ["tag1", "tag2"],
            "nested_dict": {"clause": 4},
            "none_val": None,
        }
        rec = VectorRecord(
            id="rec_complex",
            embedding=self.sample_embedding,
            document="Sample regulatory text",
            metadata=complex_metadata,
        )

        # Primitives preserved
        self.assertEqual(rec.metadata["source_document"], "sample.pdf")
        self.assertEqual(rec.metadata["page_number"], 3)
        self.assertEqual(rec.metadata["is_active"], True)
        self.assertEqual(rec.metadata["none_val"], "")

        # Complex types serialized to JSON string for ChromaDB compatibility
        self.assertIsInstance(rec.metadata["nested_list"], str)
        self.assertEqual(json.loads(rec.metadata["nested_list"]), ["tag1", "tag2"])
        self.assertIsInstance(rec.metadata["nested_dict"], str)
        self.assertEqual(json.loads(rec.metadata["nested_dict"]), {"clause": 4})

        # Test dictionary serialization
        d = rec.to_dict()
        self.assertEqual(d["id"], "rec_complex")
        self.assertEqual(d["vector_length"], 384)
        self.assertEqual(d["document"], "Sample regulatory text")
        self.assertIn("metadata", d)
        self.assertIn("embedding", d)

    # -------------------------------------------------------------------------
    # Task 4: Insert and Read Back a Test Record
    # -------------------------------------------------------------------------

    def test_task4_insert_and_readback_fidelity(self):
        """Confirm test record is inserted and read back with exact fidelity."""
        inserted_id = self.vdb.insert_record(self.sample_record)
        self.assertEqual(inserted_id, self.sample_record.id)

        # Read back record
        retrieved = self.vdb.get_record(self.sample_record.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, self.sample_record.id)
        self.assertEqual(retrieved.document, self.sample_record.document)
        self.assertEqual(len(retrieved.embedding), 384)

        # Check metadata
        for k, v in self.sample_record.metadata.items():
            self.assertEqual(retrieved.metadata.get(k), v)

        # Check vector coordinate fidelity: cosine similarity must be 1.000000
        u = np.asarray(self.sample_record.embedding)
        v = np.asarray(retrieved.embedding)
        sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_task4_verify_readback_method(self):
        """Confirm verify_readback executes full audit and reports verification_passed=True."""
        verification = self.vdb.verify_readback(self.sample_record)
        self.assertTrue(verification.verification_passed)
        self.assertTrue(verification.id_matched)
        self.assertTrue(verification.document_matched)
        self.assertTrue(verification.metadata_matched)
        self.assertTrue(verification.vector_dimension_matched)
        self.assertAlmostEqual(verification.cosine_similarity_to_original, 1.0, places=4)
        self.assertEqual(verification.expected_dimension, 384)
        self.assertEqual(verification.readback_dimension, 384)
        self.assertIn("PASSED", verification.status)

    def test_task4_get_nonexistent_record_returns_none(self):
        """Confirm querying a non-existent record ID returns None gracefully."""
        res = self.vdb.get_record("nonexistent_chunk_id_xyz")
        self.assertIsNone(res)

    # -------------------------------------------------------------------------
    # Task 5: Export Setup and Readback Artifacts
    # -------------------------------------------------------------------------

    def test_task5_export_setup_artifacts(self):
        """Verify Markdown and JSON setup reports are created with all required fields."""
        verification = self.vdb.verify_readback(self.sample_record)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            md_path = Path(temp_dir) / "setup_report.md"
            json_path = Path(temp_dir) / "readback.json"

            self.vdb.export_setup_artifacts(
                test_record=self.sample_record,
                verification=verification,
                output_markdown_path=md_path,
                output_json_path=json_path,
            )

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            # Validate JSON content
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("database_configuration", data)
            self.assertIn("readback_verification", data)
            self.assertIn("readback_record_sample", data)
            self.assertEqual(data["database_configuration"]["vector_dimension"], 384)
            self.assertEqual(data["database_configuration"]["distance_metric"], "cosine")
            self.assertTrue(data["database_configuration"]["is_reachable"])
            self.assertEqual(data["readback_record_sample"]["id"], self.sample_record.id)

            # Validate Markdown content
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Vector Database Setup & Collection Configuration Report", md_text)
            self.assertIn("Executive Setup & Readback Audit", md_text)
            self.assertIn("Stored Record Schema Architecture", md_text)
            self.assertIn("Readback Verification Proof", md_text)
            self.assertIn("HNSW:space = cosine", md_text)
            self.assertIn("384", md_text)


if __name__ == "__main__":
    unittest.main()
