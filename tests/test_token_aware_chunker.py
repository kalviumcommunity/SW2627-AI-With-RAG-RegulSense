"""Unit tests for RegulSense Token-Aware Chunking & Boundary Overlap Preservation.

Verifies:
- Task 1: Size by tokens (measured and bounded using tiktoken cl100k_base).
- Task 2: Controlled overlap (adjacent chunks repeat last N tokens of previous chunk).
- Task 3: Overlap preserving boundary context (side-by-side comparison with vs without overlap).
- Task 4: Model context budget justification (fits within llama3 8k window with safety headroom).
- Task 5: Sample output generation, serialization, and report creation.
"""

import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tiktoken

from src.document_loader import Document, DocumentLoader
from src.chunker import (
    TextChunk,
    TokenAwareChunker,
    count_tokens,
    demonstrate_token_boundary_preservation,
    generate_token_budget_justification,
    run_token_aware_demonstration,
)


class TestTokenAwareChunker(unittest.TestCase):
    """Test suite for token-aware sizing, controlled token overlap, and boundary preservation."""

    def setUp(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.sample_text = (
            "1. Preliminary Directives and Scope\n"
            "The Reserve Bank of India hereby issues the updated Master Directions on Customer Due Diligence (CDD) "
            "and Anti-Money Laundering (AML) standards for all commercial banking entities.\n\n"
            "2. Customer Due Diligence (CDD) Requirements\n"
            "Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship "
            "or executing an occasional cross-border financial transaction: "
            "(a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers "
            "using authorized OVDs such as Passport, Aadhaar, PAN Card, or Voter ID. "
            "(b) Beneficial Ownership Identification: For corporate entities, banks shall determine the natural person who ultimately owns or controls a customer, "
            "holding at least 10 percent of shares or voting rights.\n\n"
            "3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs\n"
            "Accounts classified as high-risk, including Politically Exposed Persons (PEPs), warrant enhanced scrutiny. "
            "Establishing relationships with PEPs requires written approval from an officer not below the rank of Deputy General Manager. "
            "Source of wealth and funds must be explicitly documented with corroborating financial statements.\n\n"
            "4. Record Retention Obligations\n"
            "All transaction records and customer identification files must be safely preserved for at least five years after the business relationship ends."
        )
        self.meta = {"filename": "circular_test.txt", "source": "/data/circular_test.txt"}

    # -------------------------------------------------------------------------
    # Task 1: Size by Tokens
    # -------------------------------------------------------------------------

    def test_task1_chunk_size_measured_in_tokens(self):
        """Task 1: Verify chunk size is strictly measured and bounded in tokens, not characters."""
        chunk_size_tokens = 80
        chunker = TokenAwareChunker(chunk_size=chunk_size_tokens, chunk_overlap=15, encoder=self.enc)
        chunks = chunker.split_text(self.sample_text, metadata=self.meta)

        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertIsInstance(ch, TextChunk)
            # Token count must be <= chunk_size_tokens
            self.assertLessEqual(ch.token_count, chunk_size_tokens)
            # Tiktoken token length of decoded content must match or be <= chunk_size_tokens
            actual_tokens = len(self.enc.encode(ch.content))
            self.assertLessEqual(actual_tokens, chunk_size_tokens + 2)  # boundary whitespace leeway
            self.assertEqual(ch.metadata["token_count"], ch.token_count)
            self.assertEqual(ch.strategy, f"token_aware_{chunk_size_tokens}_overlap_15")

    def test_task1_token_sizes_differ_from_character_sizes(self):
        """Task 1: Contrast token-based size (80 tokens ~350 chars) vs character sizing."""
        chunker = TokenAwareChunker(chunk_size=50, chunk_overlap=10, encoder=self.enc)
        chunks = chunker.split_text(self.sample_text, metadata=self.meta)

        for ch in chunks:
            # 50 tokens is roughly 200-250 characters
            self.assertLessEqual(ch.token_count, 50)
            self.assertGreater(ch.char_count, 100)

    def test_task1_invalid_chunk_sizes_raise_error(self):
        """Task 1: Verify invalid parameters (zero size, negative overlap, overlap >= size) raise ValueError."""
        with self.assertRaises(ValueError):
            TokenAwareChunker(chunk_size=0, chunk_overlap=0)

        with self.assertRaises(ValueError):
            TokenAwareChunker(chunk_size=100, chunk_overlap=-5)

        with self.assertRaises(ValueError):
            TokenAwareChunker(chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ValueError):
            TokenAwareChunker(chunk_size=100, chunk_overlap=150)

    # -------------------------------------------------------------------------
    # Task 2: Add Controlled Overlap
    # -------------------------------------------------------------------------

    def test_task2_controlled_overlap_repeats_trailing_tokens(self):
        """Task 2: Verify chunk k repeats trailing N tokens of chunk k-1."""
        overlap = 20
        chunk_size = 100
        chunker = TokenAwareChunker(chunk_size=chunk_size, chunk_overlap=overlap, encoder=self.enc)
        chunks = chunker.split_text(self.sample_text, metadata=self.meta)

        self.assertGreater(len(chunks), 1)

        for idx in range(len(chunks) - 1):
            ch_a = chunks[idx]
            ch_b = chunks[idx + 1]

            overlap_info = chunker.inspect_overlap(ch_a, ch_b)
            self.assertTrue(overlap_info["is_overlapping"])
            self.assertGreaterEqual(overlap_info["shared_token_count"], 1)
            self.assertLessEqual(overlap_info["shared_token_count"], overlap + 2)
            self.assertGreater(len(overlap_info["shared_text"]), 0)

            # The shared text must exist in both chunks
            self.assertIn(overlap_info["shared_text"], ch_a.content)
            self.assertIn(overlap_info["shared_text"], ch_b.content)

    def test_task2_zero_overlap_produces_disjoint_chunks(self):
        """Task 2: When overlap=0, adjacent chunks have zero overlapping tokens."""
        chunker = TokenAwareChunker(chunk_size=100, chunk_overlap=0, encoder=self.enc)
        chunks = chunker.split_text(self.sample_text, metadata=self.meta)

        self.assertGreater(len(chunks), 1)
        tokens_a = self.enc.encode(chunks[0].content)
        tokens_b = self.enc.encode(chunks[1].content)

        # Trailing token of chunk 0 should not equal leading token of chunk 1
        self.assertNotEqual(tokens_a[-1], tokens_b[0])

    # -------------------------------------------------------------------------
    # Task 3: Show Overlap Preserving Boundary Context
    # -------------------------------------------------------------------------

    def test_task3_boundary_preservation_demonstration(self):
        """Task 3: Demonstrate an idea on a boundary is severed without overlap but preserved intact with overlap."""
        demo = demonstrate_token_boundary_preservation(
            document_text=self.sample_text,
            chunk_size=120,
            chunk_overlap=30,
            encoder=self.enc,
        )

        self.assertIn("no_overlap", demo)
        self.assertIn("with_overlap", demo)
        self.assertIn("boundary_analysis", demo)

        b_analysis = demo["boundary_analysis"]
        complete_idea = b_analysis["complete_boundary_idea"]
        self.assertGreater(len(complete_idea), 0)

        no_ov = demo["no_overlap"]
        with_ov = demo["with_overlap"]

        # Without overlap: chunk 0 ends and chunk 1 begins right at the boundary cut
        self.assertTrue(no_ov["severed_boundary"])
        # With overlap: shared tokens exist and overlap is reported
        self.assertGreaterEqual(with_ov["shared_tokens"], 1)

    # -------------------------------------------------------------------------
    # Task 4: Justify Size + Overlap for Model
    # -------------------------------------------------------------------------

    def test_task4_model_budget_justification(self):
        """Task 4: Verify chosen size (300) and overlap (50) fit within llama3 8k context window."""
        justification = generate_token_budget_justification(
            model_name="llama3:latest",
            context_window=8192,
            chunk_size=300,
            chunk_overlap=50,
            top_k=4,
        )

        self.assertEqual(justification["target_model"], "llama3:latest")
        self.assertEqual(justification["context_window_limit"], 8192)
        self.assertEqual(justification["chunk_size_tokens"], 300)
        self.assertEqual(justification["chunk_overlap_tokens"], 50)
        self.assertEqual(justification["total_retrieved_tokens"], 1200)

        # Context utilization must be well under 50%
        self.assertLess(justification["context_utilization_pct"], 50.0)
        # Safety headroom must be greater than 50%
        self.assertGreater(justification["headroom_tokens"], 4000)

        # Justification rationales must exist
        self.assertIn("why_300_tokens", justification["justification"])
        self.assertIn("why_50_tokens_overlap", justification["justification"])
        self.assertIn("context_budget_fit", justification["justification"])

    # -------------------------------------------------------------------------
    # Task 5: Sample Outputs & Serialization
    # -------------------------------------------------------------------------

    def test_task5_token_aware_demonstration_and_sample_outputs(self):
        """Task 5: Verify run_token_aware_demonstration generates valid JSON and markdown artifacts."""
        corpus_file = PROJECT_ROOT / "data" / "sample_corpus" / "circular_dor_2024_108.txt"
        if not corpus_file.exists():
            self.skipTest("Sample corpus file not found.")

        out_json = PROJECT_ROOT / "outputs" / "test_token_aware_sample.json"
        out_report = PROJECT_ROOT / "outputs" / "test_token_overlap_demo.md"

        chunks, demo, just = run_token_aware_demonstration(
            corpus_dir=str(PROJECT_ROOT / "data" / "sample_corpus"),
            benchmark_file=str(corpus_file),
            output_json=str(out_json),
            output_report=str(out_report),
            chunk_size=300,
            chunk_overlap=50,
        )

        self.assertGreater(len(chunks), 0)
        self.assertTrue(out_json.exists())
        self.assertTrue(out_report.exists())

        # Validate JSON content
        data = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(len(data), len(chunks))
        first_chunk = data[0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("token_count", first_chunk)
        self.assertLessEqual(first_chunk["token_count"], 300)
        self.assertIn("metadata", first_chunk)
        self.assertIn("token_overlap", first_chunk["metadata"])

        # Validate report markdown content
        report_text = out_report.read_text(encoding="utf-8")
        self.assertIn("Token-Aware Chunk Sizing & Boundary Overlap Benchmark", report_text)
        self.assertIn("llama3:latest", report_text)
        self.assertIn("CONFIRMED INTACT", report_text)

        # Cleanup test artifacts
        if out_json.exists():
            out_json.unlink()
        if out_report.exists():
            out_report.unlink()


if __name__ == "__main__":
    unittest.main()
