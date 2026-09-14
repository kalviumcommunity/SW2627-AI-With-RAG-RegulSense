"""Unit tests for RegulSense Corpus Indexer Module.

Validates:
1. Task 1: Insertion of all corpus embeddings into ChromaDB collection.
2. Task 2: Verification of tripartite schema storing dense vectors, verbatim text, and provenance metadata.
3. Task 3: Exact count confirmation and reconciliation against upstream pipelines (15 chunks).
4. Task 4: Spot-check integrity audits verifying ID, text fidelity, metadata match, vector dimension, and cosine similarity.
5. Task 5: Generation and validity of Markdown and JSON summary artifacts.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.corpus_indexer import (
    CorpusIndexer,
    IndexingRunSummary,
    SpotCheckResult,
    run_corpus_indexing,
)
from src.vector_db import VectorDatabaseManager, VectorRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCorpusIndexer(unittest.TestCase):
    """Test suite for corpus indexing, metadata binding, count confirmation, and spot-check audits."""

    def setUp(self):
        """Initializes an in-memory vector database and CorpusIndexer for isolated testing."""
        self.vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name="test_indexing_collection",
            vector_dimension=384,
            distance_metric="cosine",
        )
        self.indexer = CorpusIndexer(
            vector_db=self.vdb,
            collection_name="test_indexing_collection",
        )

        # Create mock embedded corpus data for testing
        self.mock_chunks = [
            {
                "chunk_id": f"test_doc_001_chunk_{i:03d}",
                "source_text": f"This is verbatim regulatory clause text for chunk {i}.",
                "embedding": [float(0.001 * (j + i + 1)) for j in range(384)],
                "metadata": {
                    "source": f"C:\\data\\test_doc_001.txt",
                    "filename": "test_doc_001.txt",
                    "source_document": "test_doc_001.txt",
                    "document_id": "test_doc_001",
                    "file_type": ".txt",
                    "section": f"Section {i}: Mandatory Standards",
                    "page_number": 1 + (i // 2),
                    "chunk_index": i,
                    "total_chunks": 3,
                    "token_count": 25 + i * 5,
                    "strategy": "token_aware_300_overlap_50",
                    "relative_path": "data\\test_doc_001.txt",
                },
            }
            for i in range(3)
        ]

    # -------------------------------------------------------------------------
    # Task 1 & 2: Load, Prepare, and Insert Records with Text and Metadata
    # -------------------------------------------------------------------------

    def test_task2_prepare_vector_record_schema_and_metadata(self):
        """Confirm prepare_vector_record binds dense vector, text, and metadata correctly."""
        raw_chunk = self.mock_chunks[0]
        rec = self.indexer.prepare_vector_record(raw_chunk)

        self.assertIsInstance(rec, VectorRecord)
        self.assertEqual(rec.id, raw_chunk["chunk_id"])
        self.assertEqual(rec.document, raw_chunk["source_text"])
        self.assertEqual(len(rec.embedding), 384)

        # Metadata verification (Task 2)
        meta = rec.metadata
        self.assertEqual(meta["source_document"], "test_doc_001.txt")
        self.assertEqual(meta["filename"], "test_doc_001.txt")
        self.assertEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["section"], "Section 0: Mandatory Standards")
        self.assertEqual(meta["page_number"], 1)
        self.assertEqual(meta["document_id"], "test_doc_001")
        self.assertEqual(meta["token_count"], 25)
        self.assertEqual(meta["file_type"], ".txt")
        self.assertIn("indexed_at", meta)

    def test_task1_and_task2_insert_all_corpus_embeddings(self):
        """Confirm index_corpus inserts all prepared chunks with text and metadata into collection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"embedded_chunks": self.mock_chunks}, f)
            temp_path = Path(f.name)

        try:
            count, inserted_ids, errors = self.indexer.index_corpus(
                chunks_json_path=temp_path,
                recreate_collection=True,
            )

            self.assertEqual(count, 3)
            self.assertEqual(len(inserted_ids), 3)
            self.assertEqual(len(errors), 0)

            # Check collection count
            col = self.vdb.get_or_create_collection(self.indexer.collection_name)
            self.assertEqual(col.count(), 3)

            # Read back first record to verify stored data (Task 2)
            stored = self.vdb.get_record(self.mock_chunks[0]["chunk_id"])
            self.assertIsNotNone(stored)
            self.assertEqual(stored.id, self.mock_chunks[0]["chunk_id"])
            self.assertEqual(stored.document, self.mock_chunks[0]["source_text"])
            self.assertEqual(len(stored.embedding), 384)
            self.assertEqual(stored.metadata["source_document"], "test_doc_001.txt")
            self.assertEqual(stored.metadata["chunk_index"], 0)
            self.assertEqual(stored.metadata["section"], "Section 0: Mandatory Standards")
            self.assertEqual(stored.metadata["page_number"], 1)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # -------------------------------------------------------------------------
    # Task 3: Confirm Indexed Count
    # -------------------------------------------------------------------------

    def test_task3_confirm_indexed_count_match(self):
        """Confirm count validation confirms match when stored records equal expected chunks."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({
                "metadata": {"total_chunks_embedded": 3},
                "embedded_chunks": self.mock_chunks,
            }, f)
            temp_path = Path(f.name)

        try:
            self.indexer.index_corpus(chunks_json_path=temp_path, recreate_collection=True)
            result = self.indexer.confirm_indexed_count(embedded_json_path=temp_path)

            self.assertTrue(result["count_matches"])
            self.assertEqual(result["stored_count"], 3)
            self.assertEqual(result["expected_count"], 3)
            self.assertIn("MATCH", result["status_message"])
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_task3_confirm_indexed_count_mismatch(self):
        """Confirm count validation detects mismatch when stored count differs from expected."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({
                "metadata": {"total_chunks_embedded": 10},
                "embedded_chunks": self.mock_chunks,  # Only 3 chunks
            }, f)
            temp_path = Path(f.name)

        try:
            self.indexer.index_corpus(chunks_json_path=temp_path, recreate_collection=True)
            result = self.indexer.confirm_indexed_count(embedded_json_path=temp_path)

            self.assertFalse(result["count_matches"])
            self.assertEqual(result["stored_count"], 3)
            self.assertEqual(result["expected_count"], 10)
            self.assertIn("MISMATCH", result["status_message"])
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # -------------------------------------------------------------------------
    # Task 4: Spot-Check Stored Integrity
    # -------------------------------------------------------------------------

    def test_task4_spot_check_integrity_passes(self):
        """Verify spot-check audit confirms ID, text, metadata, vector dim, and cosine similarity."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"embedded_chunks": self.mock_chunks}, f)
            temp_path = Path(f.name)

        try:
            self.indexer.index_corpus(chunks_json_path=temp_path, recreate_collection=True)
            spot_checks = self.indexer.spot_check_stored_integrity(
                source_chunks=self.mock_chunks,
                target_chunk_ids=[self.mock_chunks[0]["chunk_id"], self.mock_chunks[1]["chunk_id"]],
            )

            self.assertEqual(len(spot_checks), 2)
            for sc in spot_checks:
                self.assertTrue(sc.id_matched)
                self.assertTrue(sc.text_matched)
                self.assertTrue(sc.metadata_matched)
                self.assertTrue(sc.vector_dimension_matched)
                self.assertEqual(sc.expected_dimension, 384)
                self.assertEqual(sc.readback_dimension, 384)
                self.assertAlmostEqual(sc.cosine_similarity, 1.0, places=4)
                self.assertEqual(len(sc.mismatches), 0)
                self.assertIn("PASSED", sc.status)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_task4_spot_check_missing_record_detected(self):
        """Verify spot-check detects records missing from the vector collection."""
        spot_checks = self.indexer.spot_check_stored_integrity(
            source_chunks=self.mock_chunks,
            target_chunk_ids=["nonexistent_chunk_id_999"],
        )
        self.assertEqual(len(spot_checks), 1)
        self.assertFalse(spot_checks[0].id_matched)
        self.assertIn("FAILED", spot_checks[0].status)

    # -------------------------------------------------------------------------
    # Task 5: Full Pipeline Execution and Export
    # -------------------------------------------------------------------------

    def test_task5_run_indexing_pipeline_and_export_artifacts(self):
        """Verify full pipeline produces valid IndexingRunSummary and exports Markdown and JSON."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            c_path = Path(temp_dir) / "embedded_chunks.json"
            c_path.write_text(
                json.dumps({
                    "metadata": {"total_chunks_embedded": 3},
                    "embedded_chunks": self.mock_chunks,
                }),
                encoding="utf-8",
            )

            summary = self.indexer.run_indexing_pipeline(
                chunks_json_path=c_path,
                recreate_collection=True,
                spot_check_all=True,
            )

            self.assertEqual(summary.overall_status, "SUCCESS")
            self.assertEqual(summary.final_indexed_count, 3)
            self.assertTrue(summary.count_matches_expected)
            self.assertEqual(summary.failures_count, 0)
            self.assertTrue(summary.all_spot_checks_passed)

            # Export artifacts
            md_path = Path(temp_dir) / "summary.md"
            json_path = Path(temp_dir) / "summary.json"

            self.indexer.export_summary_artifacts(
                summary=summary,
                output_markdown_path=md_path,
                output_json_path=json_path,
            )

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            # Verify JSON structure
            j_data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(j_data["overall_status"], "SUCCESS")
            self.assertEqual(j_data["final_indexed_count"], 3)
            self.assertTrue(j_data["count_matches_expected"])
            self.assertEqual(len(j_data["spot_checks"]), 3)

            # Verify Markdown structure
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Regulatory Corpus Vector Indexing Summary Report", md_text)
            self.assertIn("Executive Indexing Audit & Count Reconciliation", md_text)
            self.assertIn("Stored Record Schema Architecture", md_text)
            self.assertIn("Stored Record Spot-Check Integrity Audit", md_text)
            self.assertIn("EXACT MATCH", md_text)

    # -------------------------------------------------------------------------
    # Actual Corpus Integration Test
    # -------------------------------------------------------------------------

    def test_actual_corpus_data_compatibility(self):
        """Confirm that actual embedded_corpus_chunks.json (15 chunks) is parsed without error."""
        actual_path = PROJECT_ROOT / "outputs" / "embedded_corpus_chunks.json"
        if not actual_path.exists():
            self.skipTest("outputs/embedded_corpus_chunks.json does not exist")

        chunks = self.indexer.load_embedded_chunks(actual_path)
        self.assertEqual(len(chunks), 15)

        records = self.indexer.prepare_all_records(chunks)
        self.assertEqual(len(records), 15)

        for r in records:
            self.assertEqual(len(r.embedding), 384)
            self.assertIn("source_document", r.metadata)
            self.assertIn("chunk_index", r.metadata)
            self.assertIn("section", r.metadata)
            self.assertIn("page_number", r.metadata)


if __name__ == "__main__":
    unittest.main()
