"""Unit tests for RegulSense Full Corpus Ingestion Pipeline & Completeness Validation.

Validates:
1. End-to-end execution over the full corpus (PDF, TXT, HTML, MD).
2. Reporting accuracy of ingestion summary metrics.
3. Mathematical completeness reconciliation (Discovered == Ingested + Failures) and zero silent drops.
4. Correctness of sample chunks, boundaries, and 13-field metadata tagging.
5. Integrity of exported markdown summary and JSON chunk artifacts.
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.chunker import TextChunk, TokenAwareChunker
from src.document_loader import Document, DocumentLoader
from src.ingestion_pipeline import (
    CompletenessReconciliationResult,
    CompletenessValidationError,
    CorpusCompletenessValidator,
    DocumentIngestionRecord,
    IngestionPipeline,
    IngestionPipelineRunResult,
    SkippedOrFailedRecord,
    export_chunks_json,
    generate_ingestion_summary_markdown,
)
from src.text_cleaner import TextCleaner


class TestCorpusPreparationPipeline(unittest.TestCase):
    """Test suite for full corpus ingestion pipeline and completeness validation."""

    @classmethod
    def setUpClass(cls):
        """Run the ingestion pipeline on the actual data/ corpus."""
        cls.corpus_dir = Path("data")
        cls.pipeline = IngestionPipeline(chunk_size=300, chunk_overlap=50)
        cls.run_result = cls.pipeline.run(cls.corpus_dir, recursive=True)

    # -------------------------------------------------------------------------
    # Task 1: Run the full pipeline
    # -------------------------------------------------------------------------

    def test_task1_full_pipeline_ingestion(self):
        """Verify full pipeline ingests all supported regulatory documents across data/."""
        # Must discover at least 6 files (.gitkeep + 5 regulatory docs)
        self.assertGreaterEqual(self.run_result.reconciliation.total_discovered_files, 6)
        # Must successfully ingest the 5 regulatory documents
        self.assertGreaterEqual(self.run_result.total_documents_ingested, 5)

        ingested_filenames = {r.filename for r in self.run_result.ingested_records}
        expected_files = {
            "circular_dor_2024_108.txt",
            "cyber_resilience_framework.pdf",
            "digital_lending_compliance_note.html",
            "guidelines_cdd_pml_rules.md",
            "sample_regulatory_circular.txt",
        }
        for expected in expected_files:
            self.assertIn(expected, ingested_filenames, f"Expected file '{expected}' missing from ingested corpus!")

        # Total chunks created must be non-zero
        self.assertGreaterEqual(self.run_result.total_chunks, 10)

    def test_task1_multi_format_coverage(self):
        """Verify all target formats (.txt, .pdf, .html, .md) were ingested."""
        formats = {r.file_type.lower() for r in self.run_result.ingested_records}
        self.assertIn(".txt", formats)
        self.assertIn(".pdf", formats)
        self.assertIn(".html", formats)
        self.assertIn(".md", formats)

    # -------------------------------------------------------------------------
    # Task 2: Report the ingestion summary
    # -------------------------------------------------------------------------

    def test_task2_ingestion_summary_tallies(self):
        """Verify summary accurately tallies documents, chunks, and characters."""
        rec = self.run_result.reconciliation
        self.assertEqual(rec.successfully_ingested_count, len(self.run_result.ingested_records))
        self.assertEqual(rec.recorded_failures_or_skipped_count, len(self.run_result.skipped_or_failed_records))
        self.assertEqual(rec.total_chunks_created, len(self.run_result.all_chunks))

        # Check total characters and tokens are non-zero positive numbers
        self.assertGreater(rec.total_raw_chars, 10000)
        self.assertGreater(rec.total_cleaned_chars, 10000)
        self.assertGreater(rec.total_corpus_tokens, 2000)
        self.assertLessEqual(rec.total_cleaned_chars, rec.total_raw_chars)

    def test_task2_skipped_files_recorded(self):
        """Verify non-document placeholder files (.gitkeep) are explicitly tracked in skipped ledger."""
        skipped_names = {r.filename for r in self.run_result.skipped_or_failed_records}
        self.assertIn(".gitkeep", skipped_names)
        for s in self.run_result.skipped_or_failed_records:
            self.assertTrue(s.error_type)
            self.assertTrue(s.error_message)
            self.assertEqual(s.stage, "LOADING")

    # -------------------------------------------------------------------------
    # Task 3: Validate completeness (Mathematical Reconciliation & Zero Silent Drops)
    # -------------------------------------------------------------------------

    def test_task3_mathematical_reconciliation_exact(self):
        """Verify strict equation: Discovered == Ingested + Skipped, Unaccounted == 0."""
        rec = self.run_result.reconciliation
        self.assertTrue(rec.is_reconciled)
        self.assertEqual(rec.unaccounted_count, 0)
        self.assertEqual(
            rec.total_discovered_files,
            rec.successfully_ingested_count + rec.recorded_failures_or_skipped_count,
        )

    def test_task3_silent_drop_detection_raises_error(self):
        """Verify validator raises CompletenessValidationError if a document is dropped from ledger."""
        discovered = self.pipeline.scan_directory(self.corpus_dir, recursive=True)
        # Drop one document record from the ledger
        tampered_ingested = list(self.run_result.ingested_records[:-1])
        with self.assertRaises(CompletenessValidationError):
            CorpusCompletenessValidator.validate(
                discovered_files=discovered,
                ingested_records=tampered_ingested,
                skipped_records=self.run_result.skipped_or_failed_records,
                all_chunks=self.run_result.all_chunks,
            )

    def test_task3_zero_chunk_document_raises_error(self):
        """Verify validator raises error if an ingested document produced 0 chunks."""
        discovered = [Path("data/sample_regulatory_circular.txt")]
        rec = DocumentIngestionRecord(
            source_path="data/sample_regulatory_circular.txt",
            relative_path="data/sample_regulatory_circular.txt",
            filename="sample_regulatory_circular.txt",
            file_type=".txt",
            file_size_bytes=100,
            raw_char_count=100,
            cleaned_char_count=100,
            chars_removed=0,
            reduction_pct=0.0,
            raw_word_count=15,
            cleaned_word_count=15,
            token_count=20,
            chunk_count=0,  # Zero chunks!
            chunks=[],
        )
        with self.assertRaises(CompletenessValidationError):
            CorpusCompletenessValidator.validate(
                discovered_files=discovered,
                ingested_records=[rec],
                skipped_records=[],
                all_chunks=[],
            )

    def test_task3_reconciliation_with_corrupt_and_unsupported_files(self):
        """Test reconciliation in a synthetic directory with mixed valid, empty, corrupt, and unsupported files."""
        temp_dir = Path(tempfile.mkdtemp(prefix="regulsense_recon_test_"))
        try:
            # 1 valid file
            (temp_dir / "doc1.txt").write_text("This is valid compliance text for bank audit.", encoding="utf-8")
            # 1 unsupported file
            (temp_dir / "archive.bin").write_bytes(b"\x00\x01\x02\x03")
            # 1 empty file
            (temp_dir / "empty.txt").write_text("", encoding="utf-8")

            pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20)
            res = pipeline.run(temp_dir, recursive=False)

            self.assertEqual(res.reconciliation.total_discovered_files, 3)
            self.assertEqual(res.reconciliation.successfully_ingested_count, 1)
            self.assertEqual(res.reconciliation.recorded_failures_or_skipped_count, 2)
            self.assertEqual(res.reconciliation.unaccounted_count, 0)
            self.assertTrue(res.reconciliation.is_reconciled)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Task 4: Inspect sample chunks
    # -------------------------------------------------------------------------

    def test_task4_sample_chunks_metadata_schema(self):
        """Verify every chunk has all 13 standard metadata fields populated."""
        required_fields = {
            "source",
            "filename",
            "document_id",
            "file_type",
            "section",
            "page_number",
            "chunk_index",
            "total_chunks",
            "char_start",
            "char_end",
            "char_count",
            "token_count",
            "strategy",
        }
        for chunk in self.run_result.all_chunks:
            for rf in required_fields:
                self.assertIn(rf, chunk.metadata, f"Field '{rf}' missing from chunk {chunk.chunk_id}")
                self.assertIsNotNone(chunk.metadata[rf])

    def test_task4_chunk_token_bounds(self):
        """Verify all chunks strictly observe the token budget limit (<= 300 tokens)."""
        for chunk in self.run_result.all_chunks:
            self.assertLessEqual(
                chunk.token_count,
                300,
                f"Chunk {chunk.chunk_id} exceeded token limit: {chunk.token_count} > 300",
            )
            self.assertGreater(chunk.token_count, 0)

    def test_task4_traceability_across_corpus(self):
        """Verify sample chunks from every ingested document can be traced to their origin."""
        seen_files = set()
        for chunk in self.run_result.all_chunks:
            fn = chunk.metadata.get("filename")
            if fn not in seen_files:
                seen_files.add(fn)
                trace = chunk.trace()
                self.assertTrue(
                    trace.is_verified,
                    f"Trace verification failed for sample chunk from '{fn}' (similarity: {trace.similarity_score})",
                )
                self.assertGreaterEqual(trace.similarity_score, 0.95)

    # -------------------------------------------------------------------------
    # Task 5: Generated Output Artifacts
    # -------------------------------------------------------------------------

    def test_task5_output_artifacts_exist_and_valid(self):
        """Verify summary markdown and JSON export files exist and contain valid data."""
        summary_path = Path("outputs/corpus_ingestion_summary.md")
        chunks_json_path = Path("outputs/corpus_ingested_chunks.json")

        self.assertTrue(summary_path.exists(), "corpus_ingestion_summary.md does not exist!")
        self.assertTrue(chunks_json_path.exists(), "corpus_ingested_chunks.json does not exist!")

        summary_content = summary_path.read_text(encoding="utf-8")
        self.assertIn("Full Corpus Ingestion & Completeness Audit Report", summary_content)
        self.assertIn("PASSED (100% RECONCILED", summary_content)
        self.assertIn("Mathematical Completeness Reconciliation", summary_content)

        with open(chunks_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        self.assertIn("metadata", json_data)
        self.assertIn("documents", json_data)
        self.assertIn("chunks", json_data)
        self.assertEqual(len(json_data["documents"]), 5)
        self.assertEqual(len(json_data["chunks"]), 15)
        self.assertTrue(json_data["metadata"]["reconciliation_passed"])


if __name__ == "__main__":
    unittest.main()
