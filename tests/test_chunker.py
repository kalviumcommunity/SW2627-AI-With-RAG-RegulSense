"""Unit tests for RegulSense Document Chunking & Strategy Comparison.

Verifies:
- Task 1: Chunking using defined strategies (Fixed-Size, Paragraph, Recursive Structural).
- Task 2: Comparative execution of multiple strategies on identical documents.
- Task 3: Statistical reporting of chunk counts, average sizes, and distributions.
- Task 4: Contextual boundary integrity and strategy evaluation.
- Task 5: Sample chunk output and metadata preservation.
"""

from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import Document
from src.chunker import (
    BaseChunker,
    ChunkingComparator,
    ChunkingStats,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveStructuralChunker,
    TextChunk,
    count_tokens,
)


class TestChunker(unittest.TestCase):
    """Test suite for chunking strategies, comparison engine, and statistics."""

    def setUp(self):
        self.sample_text = (
            "1. Customer Due Diligence (CDD) Requirements\n"
            "Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship.\n\n"
            "2. Enhanced Due Diligence (EDD) for High-Risk Accounts\n"
            "Accounts classified as high-risk, including Politically Exposed Persons (PEPs), non-resident customers, and trusts, warrant enhanced scrutiny.\n"
            "(a) Approval from Senior Management: Establishing relationships with PEPs requires written approval from an officer not below the rank of Deputy General Manager.\n"
            "(b) Source of Funds Verification: The source of wealth and funds must be explicitly documented.\n\n"
            "3. Record Retention Obligations\n"
            "All transaction records and customer identification dossiers must be safely preserved for at least five years."
        )
        self.meta = {"filename": "circular_test.txt", "source": "/data/circular_test.txt"}

    # -------------------------------------------------------------------------
    # Task 1: Defined Chunking Strategies
    # -------------------------------------------------------------------------

    def test_fixed_size_chunker_tokens(self):
        """Verify FixedSizeChunker splits text into sliding token windows with overlap."""
        chunker = FixedSizeChunker(chunk_size=60, chunk_overlap=15, count_by="tokens")
        chunks = chunker.split_text(self.sample_text, metadata=self.meta)

        self.assertGreater(len(chunks), 1)
        for idx, ch in enumerate(chunks):
            self.assertIsInstance(ch, TextChunk)
            self.assertEqual(ch.chunk_index, idx)
            self.assertEqual(ch.total_chunks, len(chunks))
            self.assertLessEqual(ch.token_count, 65)  # Token window cap
            self.assertIn("circular_test_txt_fixed_", ch.chunk_id)
            self.assertIn("circular_test.txt", ch.citation())

    def test_fixed_size_chunker_chars(self):
        """Verify FixedSizeChunker character-based windowing."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=40, count_by="chars")
        chunks = chunker.split_text(self.sample_text, metadata=self.meta)

        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertLessEqual(ch.char_count, 205)

    def test_fixed_size_invalid_overlap_raises_error(self):
        """Verify that overlap >= chunk_size raises ValueError."""
        with self.assertRaises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

    def test_paragraph_chunker_semantic_boundaries(self):
        """Verify ParagraphChunker respects double-newline paragraph boundaries."""
        chunker = ParagraphChunker(max_chunk_size=100, min_chunk_size=20)
        chunks = chunker.split_text(self.sample_text, metadata=self.meta)

        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertIsInstance(ch, TextChunk)
            # Paragraph chunker should never cut mid-sentence
            self.assertTrue(ch.content.endswith("."))

    def test_recursive_structural_chunker_hierarchy(self):
        """Verify RecursiveStructuralChunker preserves structure while respecting budget."""
        chunker = RecursiveStructuralChunker(target_chunk_size=80, chunk_overlap=15)
        chunks = chunker.split_text(self.sample_text, metadata=self.meta)

        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertIsInstance(ch, TextChunk)
            self.assertIn("circular_test_txt_recursive_", ch.chunk_id)
            # Chunks must have non-zero token and char counts
            self.assertGreater(ch.token_count, 0)
            self.assertGreater(ch.char_count, 0)

    # -------------------------------------------------------------------------
    # Task 2 & 3: Multi-Strategy Comparison & Statistics
    # -------------------------------------------------------------------------

    def test_compare_multiple_strategies_on_same_document(self):
        """Verify ChunkingComparator runs multiple strategies and computes valid stats."""
        strategies = [
            FixedSizeChunker(chunk_size=80, chunk_overlap=15),
            ParagraphChunker(max_chunk_size=120),
            RecursiveStructuralChunker(target_chunk_size=100, chunk_overlap=20),
        ]
        comparator = ChunkingComparator()
        results = comparator.compare(self.sample_text, strategies, metadata=self.meta)

        self.assertEqual(len(results), 3)

        for strat_name, res in results.items():
            self.assertIn("chunks", res)
            self.assertIn("stats", res)
            stats: ChunkingStats = res["stats"]

            self.assertGreater(stats.total_chunks, 0)
            self.assertGreater(stats.avg_tokens, 0)
            self.assertGreater(stats.min_tokens, 0)
            self.assertGreaterEqual(stats.max_tokens, stats.min_tokens)
            self.assertGreaterEqual(stats.std_dev_tokens, 0.0)
            self.assertGreaterEqual(stats.sentence_boundary_integrity_pct, 0.0)

    def test_split_document_integration(self):
        """Verify BaseChunker.split_document integrates with Document objects."""
        doc = Document(
            content=self.sample_text,
            metadata={"filename": "rbi_circular.txt", "source": "/data/rbi_circular.txt"},
        )
        chunker = RecursiveStructuralChunker(target_chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_document(doc)

        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].metadata["filename"], "rbi_circular.txt")
        self.assertIn("[Source: rbi_circular.txt, Chunk: 1/", chunks[0].citation())

    def test_empty_text_handling(self):
        """Verify all chunkers handle empty string without crashing."""
        chunkers = [
            FixedSizeChunker(),
            ParagraphChunker(),
            RecursiveStructuralChunker(),
        ]
        for c in chunkers:
            self.assertEqual(c.split_text(""), [])
            self.assertEqual(c.split_text("   \n\n  "), [])

    # -------------------------------------------------------------------------
    # Task 5: Sample Corpus Benchmark Integration
    # -------------------------------------------------------------------------

    def test_sample_corpus_file_chunking(self):
        """Verify chunking against the realistic sample corpus file."""
        corpus_file = PROJECT_ROOT / "data" / "sample_corpus" / "circular_dor_2024_108.txt"
        if not corpus_file.exists():
            self.skipTest("Sample corpus file not found.")

        text = corpus_file.read_text(encoding="utf-8")
        chunker = RecursiveStructuralChunker(target_chunk_size=350, chunk_overlap=50)
        chunks = chunker.split_text(text, metadata={"filename": corpus_file.name})

        # Circular has ~987 tokens -> with target 350, should yield 3-5 chunks
        self.assertGreaterEqual(len(chunks), 3)
        self.assertLessEqual(len(chunks), 6)

        # Confirm sample preview works
        for ch in chunks:
            sample = ch.sample(max_chars=80)
            self.assertGreater(len(sample), 10)


if __name__ == "__main__":
    unittest.main()
