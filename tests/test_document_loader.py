"""Unit tests for RegulSense Document Loader.

Verifies:
- Task 1: Multi-format loading into plain text (TXT, MD, HTML, PDF).
- Task 2: Graceful error handling (missing, corrupt, unsupported, empty files).
- Task 3: Source identity preservation and citation references.
- Task 4: Intake confirmation (text length and snippet samples).
- Task 5: Sample corpus end-to-end integration.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import (
    Document,
    DocumentLoader,
    DocumentLoadingResult,
    DocumentNotFoundError,
    UnsupportedFormatError,
    CorruptedDocumentError,
    confirm_intake,
)


class TestDocumentLoader(unittest.TestCase):
    """Comprehensive test suite for document loading, citation, and resilience."""

    def setUp(self):
        self.loader = DocumentLoader()
        self.sample_corpus_dir = PROJECT_ROOT / "data" / "sample_corpus"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="regulsense_loader_test_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Task 1 & 3: Multi-format Loading & Source Identity Preservation
    # -------------------------------------------------------------------------

    def test_load_txt_file(self):
        """Verify TXT file is loaded into plain text with metadata."""
        txt_path = self.sample_corpus_dir / "circular_dor_2024_108.txt"
        doc = self.loader.load_file(txt_path)

        self.assertIsNotNone(doc)
        self.assertIsInstance(doc, Document)
        self.assertIn("RESERVE BANK OF INDIA", doc.content)
        self.assertIn("Customer Due Diligence", doc.content)
        self.assertEqual(doc.file_type, ".txt")
        self.assertEqual(doc.filename, "circular_dor_2024_108.txt")
        self.assertTrue(doc.char_count > 0)
        self.assertTrue(doc.word_count > 0)
        self.assertIn("circular_dor_2024_108.txt", doc.citation())

    def test_load_markdown_file(self):
        """Verify Markdown file is loaded with title metadata preserved."""
        md_path = self.sample_corpus_dir / "guidelines_cdd_pml_rules.md"
        doc = self.loader.load_file(md_path)

        self.assertIsNotNone(doc)
        self.assertEqual(doc.file_type, ".md")
        self.assertEqual(doc.filename, "guidelines_cdd_pml_rules.md")
        self.assertIn("Politically Exposed Persons (PEP) Workflow", doc.content)
        self.assertIn("title", doc.metadata)
        self.assertIn("Internal Compliance Guidelines", doc.metadata["title"])
        self.assertEqual(doc.citation(), "[Source: guidelines_cdd_pml_rules.md]")

    def test_load_html_file(self):
        """Verify HTML file strips tags/scripts and extracts plain text and title."""
        html_path = self.sample_corpus_dir / "digital_lending_compliance_note.html"
        doc = self.loader.load_file(html_path)

        self.assertIsNotNone(doc)
        self.assertEqual(doc.file_type, ".html")
        self.assertEqual(doc.filename, "digital_lending_compliance_note.html")
        # Ensure HTML tags and styles are stripped
        self.assertNotIn("<style>", doc.content)
        self.assertNotIn("<body>", doc.content)
        self.assertNotIn("<h1>", doc.content)
        self.assertIn("Direct Fund Disbursal and Repayment", doc.content)
        self.assertIn("Key Fact Statement (KFS) Mandate", doc.content)
        self.assertIn("title", doc.metadata)
        self.assertIn("Regulatory Advisory: Digital Lending Guidelines", doc.metadata["title"])

    def test_load_pdf_file(self):
        """Verify PDF file extracts multi-page text and records page count."""
        pdf_path = self.sample_corpus_dir / "cyber_resilience_framework.pdf"
        doc = self.loader.load_file(pdf_path)

        self.assertIsNotNone(doc)
        self.assertEqual(doc.file_type, ".pdf")
        self.assertEqual(doc.filename, "cyber_resilience_framework.pdf")
        self.assertIn("DEPARTMENT OF CYBER SECURITY", doc.content)
        self.assertIn("Mandatory Two-Factor Authentication", doc.content)
        self.assertIn("Incident Reporting Timelines (6-Hour Rule)", doc.content)
        self.assertIn("page_count", doc.metadata)
        self.assertEqual(doc.metadata["page_count"], 2)
        self.assertIn("cyber_resilience_framework.pdf", doc.citation())
        self.assertIn("Pages: 2", doc.citation())

    # -------------------------------------------------------------------------
    # Task 2: Fault Tolerance and Graceful Error Handling
    # -------------------------------------------------------------------------

    def test_missing_file_handled_gracefully(self):
        """Verify that loading a non-existent file returns None without crashing."""
        missing_path = self.temp_dir / "does_not_exist.pdf"
        doc = self.loader.load_file(missing_path)
        self.assertIsNone(doc)

    def test_missing_file_raise_mode(self):
        """Verify that strict mode raises DocumentNotFoundError on missing file."""
        strict_loader = DocumentLoader(raise_on_error=True)
        missing_path = self.temp_dir / "ghost_file.txt"
        with self.assertRaises(DocumentNotFoundError) as ctx:
            strict_loader.load_file(missing_path)
        self.assertIn("ghost_file.txt", str(ctx.exception))

    def test_corrupt_pdf_handled_gracefully(self):
        """Verify that corrupted PDF binary is skipped with a clear error without crashing."""
        corrupt_pdf = self.temp_dir / "corrupted_document.pdf"
        corrupt_pdf.write_bytes(b"INVALID_HEADER_GARBAGE_BYTES_1234567890")

        # Non-raising mode should log warning and return None
        doc = self.loader.load_file(corrupt_pdf)
        self.assertIsNone(doc)

        # Strict mode raises CorruptedDocumentError
        strict_loader = DocumentLoader(raise_on_error=True)
        with self.assertRaises(CorruptedDocumentError) as ctx:
            strict_loader.load_file(corrupt_pdf)
        self.assertIn("corrupted_document.pdf", str(ctx.exception))

    def test_empty_zero_byte_file_handled_gracefully(self):
        """Verify that a zero-byte file is handled gracefully as corrupt/empty without crashing."""
        empty_file = self.temp_dir / "empty_compliance_circular.txt"
        empty_file.touch()

        doc = self.loader.load_file(empty_file)
        self.assertIsNone(doc)

    def test_unsupported_file_format_skipped_gracefully(self):
        """Verify that unsupported file extensions are rejected gracefully with a message."""
        bad_file = self.temp_dir / "unsupported_data.bin"
        bad_file.write_bytes(b"\x00\x01\x02\x03\x04")

        doc = self.loader.load_file(bad_file)
        self.assertIsNone(doc)

        strict_loader = DocumentLoader(raise_on_error=True)
        with self.assertRaises(UnsupportedFormatError) as ctx:
            strict_loader.load_file(bad_file)
        self.assertIn(".bin", str(ctx.exception))

    def test_load_directory_with_mixed_valid_and_invalid_files(self):
        """Verify directory loader ingests valid files while recording and skipping invalid items."""
        # Setup mixed folder
        valid_txt = self.temp_dir / "valid_rule.txt"
        valid_txt.write_text("Rule 1: Maintain anti-money laundering records for 5 years.", encoding="utf-8")

        valid_md = self.temp_dir / "valid_policy.md"
        valid_md.write_text("# AML Policy\n\nMust report CTRs to FIU-IND.", encoding="utf-8")

        corrupt_pdf = self.temp_dir / "broken.pdf"
        corrupt_pdf.write_bytes(b"NOT_A_VALID_PDF_FILE")

        unsupported_exe = self.temp_dir / "malware.exe"
        unsupported_exe.write_bytes(b"MZ\x90\x00")

        empty_file = self.temp_dir / "zero.html"
        empty_file.touch()

        # Run directory loader
        result = self.loader.load_directory(self.temp_dir)

        self.assertEqual(result.total_examined, 5)
        self.assertEqual(result.successful_count, 2)
        self.assertEqual(result.skipped_count, 3)

        # Confirm valid docs were ingested
        loaded_names = {d.filename for d in result.documents}
        self.assertEqual(loaded_names, {"valid_rule.txt", "valid_policy.md"})

        # Confirm errors contain structured information
        error_files = {err.get("filename") for err in result.errors}
        self.assertIn("broken.pdf", error_files)
        self.assertIn("malware.exe", error_files)
        self.assertIn("zero.html", error_files)

    # -------------------------------------------------------------------------
    # Task 4: Confirm Intake Verification
    # -------------------------------------------------------------------------

    def test_confirm_intake_output_formatting(self):
        """Verify confirm_intake generates complete formatted report with lengths and samples."""
        result = self.loader.load_directory(self.sample_corpus_dir)
        summary = confirm_intake(result, verbose=False)

        self.assertIn("RegulSense Document Intake Confirmation", summary)
        self.assertIn("Total Files Examined : 4", summary)
        self.assertIn("Successfully Loaded  : 4", summary)
        self.assertIn("Skipped / Failed     : 0", summary)

        # Check each file length and sample is in the output
        for doc in result.documents:
            self.assertIn(doc.filename, summary)
            self.assertIn(f"{doc.char_count:,} characters", summary)
            self.assertIn(doc.citation(), summary)
            self.assertIn(doc.sample(max_chars=180), summary)

    # -------------------------------------------------------------------------
    # Task 5: Sample Corpus End-to-End Ingestion
    # -------------------------------------------------------------------------

    def test_sample_corpus_all_four_formats_present_and_loaded(self):
        """Verify the sample corpus contains all 4 formats and all load successfully."""
        result = self.loader.load_directory(self.sample_corpus_dir)

        self.assertEqual(result.successful_count, 4)
        extensions = {d.file_type for d in result.documents}
        self.assertEqual(extensions, {".txt", ".pdf", ".html", ".md"})

        total_words = result.total_words_loaded
        self.assertGreater(total_words, 1000)


if __name__ == "__main__":
    unittest.main()
