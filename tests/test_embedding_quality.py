"""Unit tests for RegulSense Embedding Quality Sanity Testing Engine.

Validates:
1. Task 1: Known relevance test case specifications, ground-truth mapping, and negative controls.
2. Task 2: Related regulatory chunks rank strictly above unrelated negative controls with positive separation margins.
3. Task 3: Detection, diagnostic logging, and architectural explanation of surprising/failing edge cases (e.g. PEP approval query).
4. Task 4: Compilation of sanity reports with test counts, passes, failures, top-ranked sources, and scores.
5. Task 5: Structural validity of exported Markdown and JSON sanity artifacts.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from src.embedding_quality_checker import (
    DEFAULT_SANITY_TEST_CASES,
    SURPRISING_CASE_ANALYSIS,
    EmbeddingQualityChecker,
    QualitySanitySummary,
    RelevanceTestCase,
    TestCaseEvaluation,
    run_embedding_quality_audit,
)
from src.similarity_search import QueryRankingResult, ScoredChunk, SimilaritySearchEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestEmbeddingQuality(unittest.TestCase):
    """Test suite for embedding quality sanity verification and failure diagnosis."""

    @classmethod
    def setUpClass(cls):
        """Initializes the search engine and quality checker once for test suite."""
        cls.engine = SimilaritySearchEngine()
        cls.checker = EmbeddingQualityChecker(search_engine=cls.engine)
        cls.summary = cls.checker.run_sanity_suite(top_k=5)

    # -------------------------------------------------------------------------
    # Task 1: Known Relevance Tests Definition
    # -------------------------------------------------------------------------

    def test_task1_known_relevance_test_cases_structure(self):
        """Verify the test suite contains structured test cases with designated targets and negative controls."""
        self.assertGreaterEqual(len(DEFAULT_SANITY_TEST_CASES), 4)

        for tc in DEFAULT_SANITY_TEST_CASES:
            self.assertIsInstance(tc.case_id, str)
            self.assertTrue(tc.case_id.startswith("SANITY-"))
            self.assertGreater(len(tc.query), 15)
            self.assertGreater(len(tc.category), 3)
            self.assertIsInstance(tc.expected_relevant_chunk_ids, list)
            self.assertGreater(len(tc.expected_relevant_chunk_ids), 0)
            self.assertIsInstance(tc.expected_unrelated_chunk_ids, list)
            self.assertGreater(len(tc.expected_unrelated_chunk_ids), 0)
            # Ensure target chunks and negative controls are completely disjoint
            overlap = set(tc.expected_relevant_chunk_ids).intersection(set(tc.expected_unrelated_chunk_ids))
            self.assertEqual(len(overlap), 0, f"Overlapping chunk IDs detected in {tc.case_id}")

    def test_task1_target_chunks_exist_in_embedded_corpus(self):
        """Confirm all expected relevant and unrelated chunk IDs exist in the active corpus."""
        corpus_ids = {c.get("chunk_id") for c in self.checker.corpus_chunks}
        for tc in DEFAULT_SANITY_TEST_CASES:
            for rel_id in tc.expected_relevant_chunk_ids:
                self.assertIn(
                    rel_id,
                    corpus_ids,
                    f"Expected relevant chunk '{rel_id}' in {tc.case_id} not found in embedded corpus!",
                )
            for unrel_id in tc.expected_unrelated_chunk_ids:
                self.assertIn(
                    unrel_id,
                    corpus_ids,
                    f"Expected unrelated chunk '{unrel_id}' in {tc.case_id} not found in embedded corpus!",
                )

    # -------------------------------------------------------------------------
    # Task 2: Confirm Related Ranks Above Unrelated
    # -------------------------------------------------------------------------

    def test_task2_related_chunks_rank_above_unrelated(self):
        """Confirm that for standard test cases, related chunks rank strictly higher than unrelated chunks."""
        eval_map = {e.case_id: e for e in self.summary.evaluations}

        # Validate standard passing test cases
        passing_cases = ["SANITY-01", "SANITY-02", "SANITY-03", "SANITY-04"]
        for case_id in passing_cases:
            e = eval_map[case_id]
            self.assertTrue(e.passed, f"Test case {case_id} failed unexpectedly: {e.diagnostic_notes}")
            self.assertEqual(e.status, "PASSED")
            # Best relevant score must exceed worst unrelated score
            self.assertGreater(
                e.best_relevant_score,
                e.worst_unrelated_score,
                f"Case {case_id}: Best relevant ({e.best_relevant_score}) must exceed worst unrelated ({e.worst_unrelated_score})",
            )
            # Separation margin must be positive and substantial (> 0.20)
            self.assertGreater(
                e.separation_margin,
                0.20,
                f"Case {case_id}: Separation margin ({e.separation_margin}) should be > 0.20",
            )
            # Best relevant chunk must be in Top 3
            self.assertLessEqual(e.best_relevant_rank, 3)

    def test_task2_positive_average_separation_margin(self):
        """Confirm the entire test suite achieves a strong average separation margin."""
        self.assertGreater(self.summary.average_separation_margin, 0.15)
        for e in self.summary.evaluations:
            # Every test case should have a positive margin against the negative control
            self.assertGreater(
                e.separation_margin,
                0.0,
                f"Case {e.case_id} has non-positive separation margin ({e.separation_margin})",
            )

    # -------------------------------------------------------------------------
    # Task 3: Identify Failing / Surprising Case
    # -------------------------------------------------------------------------

    def test_task3_surprising_case_identified_and_diagnosed(self):
        """Confirm that the PEP approval authority query is identified as a surprising retrieval case."""
        eval_map = {e.case_id: e for e in self.summary.evaluations}
        self.assertIn("SANITY-05", eval_map)

        pep_eval = eval_map["SANITY-05"]
        # The ground truth chunk is in section 3 (EDD / PEPs)
        self.assertIn("tokenaware_003", pep_eval.best_relevant_chunk_id)

        # In dense embedding, ground truth failed to reach Top 3 (placed at Rank 6)
        self.assertGreater(
            pep_eval.best_relevant_rank,
            3,
            f"Expected SANITY-05 ground truth to be pushed outside Top 3, got rank {pep_eval.best_relevant_rank}",
        )
        self.assertIn("EXPECTED_SURPRISE", pep_eval.status)
        self.assertIn("bi-encoder semantic dilution", pep_eval.diagnostic_notes.lower())

    def test_task3_architectural_explanation_note_integrity(self):
        """Verify the quality checker includes rigorous architectural explanation of the failure mode."""
        self.assertIsInstance(SURPRISING_CASE_ANALYSIS, str)
        self.assertGreater(len(SURPRISING_CASE_ANALYSIS), 400)

        lower_exp = SURPRISING_CASE_ANALYSIS.lower()
        self.assertIn("politically exposed person", lower_exp)
        self.assertIn("bi-encoder", lower_exp)
        self.assertTrue("cross-encoder" in lower_exp or "bm25" in lower_exp or "hybrid" in lower_exp)
        self.assertIn("preamble", lower_exp)

    # -------------------------------------------------------------------------
    # Task 4 & 5: Summarize Sanity Report & Export Artifacts
    # -------------------------------------------------------------------------

    def test_task4_sanity_summary_metrics(self):
        """Verify summary report tracks total tests, passes, failures, and top-ranked sources."""
        self.assertEqual(self.summary.total_tests, len(DEFAULT_SANITY_TEST_CASES))
        self.assertGreaterEqual(self.summary.passed_tests, 4)
        self.assertGreaterEqual(self.summary.failed_tests, 1)  # Includes SANITY-05
        self.assertGreater(self.summary.pass_rate_pct, 50.0)

        for e in self.summary.evaluations:
            self.assertIsInstance(e.top_1_chunk_id, str)
            self.assertIsInstance(e.top_1_document, str)
            self.assertIsInstance(e.top_1_section, str)
            self.assertGreater(e.top_1_score, 0.0)
            self.assertEqual(len(e.top_ranked_matches), 3)

    def test_task5_export_sanity_report_artifacts(self):
        """Verify that Markdown and JSON sanity reports are exported with valid structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            md_path = Path(temp_dir) / "sanity_report.md"
            json_path = Path(temp_dir) / "sanity_report.json"

            self.checker.export_sanity_artifacts(
                summary=self.summary,
                output_markdown_path=md_path,
                output_json_path=json_path,
            )

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            # Validate JSON content
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["total_tests"], len(DEFAULT_SANITY_TEST_CASES))
            self.assertIn("evaluations", data)
            self.assertIn("surprising_case_analysis", data)
            self.assertIn("average_separation_margin", data)

            # Validate Markdown content
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Embedding Quality & Retrieval Sanity Report", md_text)
            self.assertIn("Executive Sanity Verification Matrix", md_text)
            self.assertIn("Test-by-Test Retrieval Ledger", md_text)
            self.assertIn("Deep Dive: Analysis of Surprising & Borderline Retrieval Cases", md_text)
            self.assertIn("Key Architectural Takeaways for RegulSense", md_text)


if __name__ == "__main__":
    unittest.main()
