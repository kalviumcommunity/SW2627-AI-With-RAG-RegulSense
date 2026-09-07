"""Unit tests for RegulSense Text Cleaning & Normalization Pipeline.

Verifies:
- Task 1: Boilerplate removal (headers, footers, page counters, nav breadcrumbs).
- Task 2: Normalization of whitespace, Unicode NFKC, encoding artifacts, and line wraps.
- Task 3: Uniform corpus-wide cleaning and metadata provenance tracking.
- Task 4: Before/after comparison and metrics evaluation.
- Task 5: Integration and documentation report generation.
"""

from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import Document, DocumentLoader
from src.text_cleaner import (
    BoilerplateRemover,
    LineWrapNormalizer,
    TextCleaner,
    UnicodeNormalizer,
    WhitespaceNormalizer,
    compare_text,
)


class TestTextCleaner(unittest.TestCase):
    """Test suite for the text cleaning and normalization engine."""

    def setUp(self):
        self.cleaner = TextCleaner()
        self.sample_corpus_dir = PROJECT_ROOT / "data" / "sample_corpus"

    # -------------------------------------------------------------------------
    # Task 1: Boilerplate Removal
    # -------------------------------------------------------------------------

    def test_remove_page_number_variants(self):
        """Verify that page numbers in multiple notations are stripped cleanly."""
        raw_text = (
            "Section 1: KYC Standards.\n"
            "Page 1 of 5\n"
            "All customer accounts must undergo CDD.\n"
            "- 2 -\n"
            "Transactions over INR 10 Lakhs require CTR reporting.\n"
            "PAGE 3/5\n"
            "Audit records must be retained for 5 years.\n"
            "--- Page 4 ---\n"
            "Final provisions."
        )
        cleaned = BoilerplateRemover.clean(raw_text)

        self.assertNotIn("Page 1 of 5", cleaned)
        self.assertNotIn("- 2 -", cleaned)
        self.assertNotIn("PAGE 3/5", cleaned)
        self.assertNotIn("--- Page 4 ---", cleaned)
        self.assertIn("Section 1: KYC Standards.", cleaned)
        self.assertIn("Final provisions.", cleaned)

    def test_remove_confidentiality_and_disclaimer_stamps(self):
        """Verify that confidentiality banners and classification headers are removed."""
        raw_text = (
            "CONFIDENTIAL - FOR INTERNAL USE ONLY\n"
            "DO NOT DISTRIBUTE\n"
            "Master Direction on Cyber Resilience.\n"
            "Department of Supervision • Reserve Bank of India • Confidential Internal Copy\n"
            "RESTRICTED CIRCULATION\n"
            "Mandatory dynamic 2FA applies."
        )
        cleaned = BoilerplateRemover.clean(raw_text)

        self.assertNotIn("CONFIDENTIAL - FOR INTERNAL USE ONLY", cleaned)
        self.assertNotIn("DO NOT DISTRIBUTE", cleaned)
        self.assertNotIn("Confidential Internal Copy", cleaned)
        self.assertNotIn("RESTRICTED CIRCULATION", cleaned)
        self.assertIn("Master Direction on Cyber Resilience.", cleaned)
        self.assertIn("Mandatory dynamic 2FA applies.", cleaned)

    def test_remove_web_navigation_and_breadcrumbs(self):
        """Verify web navigation links, breadcrumbs, and footer boilerplate are eliminated."""
        raw_text = (
            "Home > Regulations > Master Directions > 2024\n"
            "Skip to main content\n"
            "Navigation Menu\n"
            "Digital Lending Guidelines for Regulated Entities.\n"
            "Back to top\n"
            "Print this page\n"
            "Copyright © 2024 Reserve Bank of India. All rights reserved."
        )
        cleaned = BoilerplateRemover.clean(raw_text)

        self.assertNotIn("Home > Regulations", cleaned)
        self.assertNotIn("Skip to main content", cleaned)
        self.assertNotIn("Navigation Menu", cleaned)
        self.assertNotIn("Back to top", cleaned)
        self.assertNotIn("Print this page", cleaned)
        self.assertNotIn("All rights reserved", cleaned)
        self.assertIn("Digital Lending Guidelines for Regulated Entities.", cleaned)

    # -------------------------------------------------------------------------
    # Task 2: Unicode Normalization, Line Wraps & Whitespace
    # -------------------------------------------------------------------------

    def test_unicode_nfkc_smart_quotes_and_ligatures(self):
        """Verify Unicode NFKC normalization, smart quotes, em dashes, and ligatures."""
        raw_text = (
            "The “Master Direction”—issued with ‘strict’ compliance—mandates that\n"
            "PEPs provide corroborating ﬁnancial statements (see \u200ba\u200b threshold…)."
        )
        cleaned = UnicodeNormalizer.clean(raw_text)

        # Smart quotes replaced with straight quotes
        self.assertIn('"Master Direction"', cleaned)
        self.assertIn("'strict'", cleaned)
        # Em dash replaced with hyphen
        self.assertIn("-issued with", cleaned)
        # Ligature fi decomposed
        self.assertIn("financial", cleaned)
        self.assertNotIn("ﬁ", cleaned)
        # Ellipsis replaced
        self.assertIn("threshold...", cleaned)
        # Zero-width spaces removed
        self.assertNotIn("\u200b", cleaned)

    def test_broken_line_wrap_and_dehyphenation(self):
        """Verify de-hyphenation of split words and unwrap of soft line breaks."""
        raw_text = (
            "All regu-\n"
            "latory entities must deploy auto-\n"
            "mated trans-\n"
            "action monitoring systems to detect money laundering."
        )
        cleaned = LineWrapNormalizer.clean(raw_text)

        self.assertIn("regulatory", cleaned)
        self.assertIn("automated", cleaned)
        self.assertIn("transaction", cleaned)
        self.assertNotIn("regu-", cleaned)
        self.assertNotIn("trans-", cleaned)

    def test_line_wrap_preserves_markdown_structures(self):
        """Verify that line unwrapping does not destroy Markdown headings, lists, or tables."""
        raw_markdown = (
            "# Master Direction on KYC\n\n"
            "Key controls:\n"
            "- First control: verify Officially Valid Documents\n"
            "- Second control: identify Beneficial Owners with >10% stake\n"
            "- Third control: obtain DGM approval for PEPs\n\n"
            "| Risk Tier | Frequency |\n"
            "| Tier 1 | 10 years |\n"
            "| Tier 3 | 2 years |"
        )
        cleaned = LineWrapNormalizer.clean(raw_markdown)

        self.assertIn("# Master Direction on KYC", cleaned)
        self.assertIn("- First control: verify", cleaned)
        self.assertIn("- Second control: identify", cleaned)
        self.assertIn("| Risk Tier | Frequency |", cleaned)

    def test_whitespace_and_blank_line_collapsing(self):
        """Verify runaway multiple spaces and 3+ consecutive newlines are collapsed."""
        raw_text = (
            "Reporting    Threshold:      INR 10,00,000.\n\n\n\n\n\n\n"
            "Furnish CTR   reports by the   15th of the succeeding month.\n\n\n\n"
            "Audit trails must be maintained."
        )
        cleaned = WhitespaceNormalizer.clean(raw_text)

        self.assertIn("Reporting Threshold: INR 10,00,000.", cleaned)
        self.assertIn("Furnish CTR reports by the 15th of the succeeding month.", cleaned)
        # 7 blank lines collapsed to double newline
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("\n\n", cleaned)

    # -------------------------------------------------------------------------
    # Task 3: Uniform Corpus Application & Metadata Tracking
    # -------------------------------------------------------------------------

    def test_clean_document_metadata_enrichment(self):
        """Verify clean_document preserves citation and attaches reduction metrics."""
        doc = Document(
            content="CONFIDENTIAL\nPage 1 of 1\nRisk rating is Tier 1 for low-risk customers.",
            metadata={"source": "test_doc.txt", "filename": "test_doc.txt", "file_type": ".txt"},
        )
        cleaned_doc = self.cleaner.clean_document(doc)

        self.assertTrue(cleaned_doc.metadata.get("cleaned"))
        self.assertGreater(cleaned_doc.metadata["chars_removed"], 0)
        self.assertGreater(cleaned_doc.metadata["reduction_pct"], 0.0)
        self.assertEqual(cleaned_doc.citation(), "[Source: test_doc.txt]")
        self.assertIn("Risk rating is Tier 1 for low-risk customers.", cleaned_doc.content)
        self.assertNotIn("CONFIDENTIAL", cleaned_doc.content)
        self.assertNotIn("Page 1 of 1", cleaned_doc.content)

    def test_clean_corpus_uniformity(self):
        """Verify that batch cleaning across the sample corpus applies uniformly."""
        loader = DocumentLoader()
        load_result = loader.load_directory(self.sample_corpus_dir)
        self.assertEqual(load_result.successful_count, 4)

        corpus_result = self.cleaner.clean_corpus(load_result.documents)

        self.assertEqual(len(corpus_result.documents), 4)
        self.assertEqual(len(corpus_result.document_metrics), 4)
        self.assertGreater(corpus_result.total_raw_chars, 0)
        self.assertGreater(corpus_result.total_cleaned_chars, 0)
        self.assertGreaterEqual(corpus_result.total_chars_removed, 0)

        # Confirm all cleaned documents have updated metadata
        for d in corpus_result.documents:
            self.assertTrue(d.metadata["cleaned"])
            self.assertIn("chars_removed", d.metadata)
            self.assertIn("reduction_pct", d.metadata)

    # -------------------------------------------------------------------------
    # Task 4: Before / After Comparison Helper
    # -------------------------------------------------------------------------

    def test_compare_text_helper(self):
        """Verify compare_text calculates accurate transformation delta and snippets."""
        raw = "Page 1 of 1\nUnder Section 35A, banks must keep records."
        cleaned = "Under Section 35A, banks must keep records."

        comp = compare_text(raw, cleaned, title="Page Number Test")
        self.assertEqual(comp["title"], "Page Number Test")
        self.assertEqual(comp["raw_chars"], len(raw))
        self.assertEqual(comp["cleaned_chars"], len(cleaned))
        self.assertEqual(comp["chars_removed"], len(raw) - len(cleaned))
        self.assertGreater(comp["reduction_pct"], 0.0)
        self.assertIn("Under Section 35A", comp["cleaned_sample"])


if __name__ == "__main__":
    unittest.main()
