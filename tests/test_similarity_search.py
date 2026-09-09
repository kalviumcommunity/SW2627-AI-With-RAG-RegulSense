"""Unit tests for RegulSense Embedding Similarity Ranking Module.

Validates:
1. Task 1: Computing similarity metrics (Cosine Similarity and Euclidean Distance) with boundary checks.
2. Task 2: Comparing a query vector against corpus chunk embeddings.
3. Task 3: Ranking corpus chunks in descending order of similarity, isolating top and bottom matches.
4. Task 4: Validating mathematical and conceptual justification for choosing Cosine Similarity.
5. Task 5: Validating exported ranking artifacts (Markdown and JSON) containing scores, text, and metadata.
"""

import json
from pathlib import Path
import unittest

import numpy as np

from src.similarity_search import (
    DEFAULT_QUERY,
    METRIC_JUSTIFICATION,
    QueryRankingResult,
    ScoredChunk,
    SimilaritySearchEngine,
    compute_cosine_similarity,
    compute_euclidean_distance,
    run_similarity_ranking_demo,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestSimilaritySearch(unittest.TestCase):
    """Test suite for embedding similarity calculation, query ranking, and metric justification."""

    @classmethod
    def setUpClass(cls):
        """Initialize engine and compute rankings for default query once for test suite."""
        cls.engine = SimilaritySearchEngine()
        cls.corpus_chunks = cls.engine.load_corpus_chunks()
        cls.ranking = cls.engine.rank_chunks_for_query(DEFAULT_QUERY, cls.corpus_chunks)

    # -------------------------------------------------------------------------
    # Task 1: Compute Similarity Metric
    # -------------------------------------------------------------------------

    def test_task1_cosine_similarity_math_boundaries(self):
        """Verify cosine similarity mathematical boundary conditions."""
        # 1. Identical vectors -> 1.0
        v1 = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(compute_cosine_similarity(v1, v1), 1.0, places=5)

        # 2. Orthogonal vectors -> 0.0
        v_ortho_1 = [1.0, 0.0]
        v_ortho_2 = [0.0, 1.0]
        self.assertAlmostEqual(compute_cosine_similarity(v_ortho_1, v_ortho_2), 0.0, places=5)

        # 3. Opposite vectors -> -1.0
        v_opp_1 = [1.0, 2.0]
        v_opp_2 = [-1.0, -2.0]
        self.assertAlmostEqual(compute_cosine_similarity(v_opp_1, v_opp_2), -1.0, places=5)

        # 4. Zero vector -> 0.0 (graceful guard)
        v_zero = [0.0, 0.0]
        self.assertEqual(compute_cosine_similarity(v1, v_zero), 0.0)

        # 5. Length scaling invariance: scaling a vector does not alter cosine similarity
        v_scaled = [10.0, 20.0]
        v_base = [1.0, 2.0]
        self.assertAlmostEqual(compute_cosine_similarity(v_base, v_scaled), 1.0, places=5)

    def test_task1_euclidean_distance_properties(self):
        """Verify Euclidean distance metric properties."""
        v1 = [1.0, 2.0, 3.0]
        # Distance to self is 0
        self.assertAlmostEqual(compute_euclidean_distance(v1, v1), 0.0, places=5)
        # Distance is always non-negative
        v2 = [4.0, 5.0, 6.0]
        self.assertGreater(compute_euclidean_distance(v1, v2), 0.0)

    # -------------------------------------------------------------------------
    # Task 2: Compare Query Against Chunks
    # -------------------------------------------------------------------------

    def test_task2_query_embedding_and_chunk_comparison(self):
        """Confirm query is embedded and compared against all corpus chunks."""
        self.assertGreater(len(self.corpus_chunks), 0)
        self.assertEqual(len(self.ranking.ranked_chunks), len(self.corpus_chunks))
        self.assertEqual(self.ranking.vector_dimension, 384)

        # Every chunk must receive a score within [-1.0, 1.0]
        for item in self.ranking.ranked_chunks:
            self.assertGreaterEqual(item.score, -1.0)
            self.assertLessEqual(item.score, 1.0)
            self.assertIsInstance(item.source_text, str)
            self.assertGreater(len(item.source_text), 10)

    def test_task2_empty_query_raises_value_error(self):
        """Verify empty query string is rejected."""
        with self.assertRaises(ValueError):
            self.engine.embed_query("   ")

    # -------------------------------------------------------------------------
    # Task 3: Rank and Show Results
    # -------------------------------------------------------------------------

    def test_task3_chunks_ranked_in_descending_order(self):
        """Confirm chunks are sorted strictly in descending order of similarity score."""
        scores = [c.score for c in self.ranking.ranked_chunks]
        self.assertEqual(scores, sorted(scores, reverse=True), "Scores must be strictly non-increasing by rank")

        # Ranks must be sequential 1..N
        ranks = [c.rank for c in self.ranking.ranked_chunks]
        self.assertEqual(ranks, list(range(1, len(self.ranking.ranked_chunks) + 1)))

    def test_task3_top_vs_bottom_separation(self):
        """Confirm top similar chunk scores significantly higher than bottom least similar chunk."""
        top_chunk = self.ranking.top_chunks[0]
        bottom_chunk = self.ranking.bottom_chunks[-1]

        self.assertGreater(
            top_chunk.score,
            bottom_chunk.score,
            f"Top chunk score ({top_chunk.score}) must exceed bottom chunk score ({bottom_chunk.score})",
        )
        self.assertGreater(
            top_chunk.score - bottom_chunk.score,
            0.15,
            "Expected substantial discrimination margin between top and bottom chunks",
        )

        # Confirm top chunk is directly on-topic for CDD/KYC query
        self.assertIn("Customer Due Diligence", top_chunk.section)

    # -------------------------------------------------------------------------
    # Task 4: Justify the Metric
    # -------------------------------------------------------------------------

    def test_task4_metric_justification_content(self):
        """Verify the justification document details why Cosine Similarity was selected."""
        self.assertIsInstance(METRIC_JUSTIFICATION, str)
        self.assertGreater(len(METRIC_JUSTIFICATION), 300)

        # Must justify direction/angle vs magnitude
        self.assertTrue(
            "direction" in METRIC_JUSTIFICATION.lower() or "angle" in METRIC_JUSTIFICATION.lower(),
            "Justification must discuss angular direction",
        )
        self.assertIn("magnitude", METRIC_JUSTIFICATION.lower())

        # Must justify text/chunk length invariance
        self.assertTrue(
            "length" in METRIC_JUSTIFICATION.lower() or "norm" in METRIC_JUSTIFICATION.lower(),
            "Justification must discuss length/norm normalization",
        )

        # Must mention bounded range
        self.assertIn("-1.0", METRIC_JUSTIFICATION)
        self.assertIn("+1.0", METRIC_JUSTIFICATION)

    # -------------------------------------------------------------------------
    # Task 5: Output Artifact Integrity
    # -------------------------------------------------------------------------

    def test_task5_ranking_artifacts_exist_and_valid(self):
        """Verify similarity ranking Markdown report and JSON artifact exist and are valid."""
        md_path = PROJECT_ROOT / "outputs" / "similarity_ranking_results.md"
        json_path = PROJECT_ROOT / "outputs" / "similarity_ranking_results.json"

        self.engine.export_ranking_artifacts(self.ranking, md_path, json_path)

        self.assertTrue(md_path.exists(), f"Markdown report missing at {md_path}")
        self.assertTrue(json_path.exists(), f"JSON export missing at {json_path}")

        # Check Markdown content
        md_text = md_path.read_text(encoding="utf-8")
        self.assertIn("Top Most Similar Chunks", md_text)
        self.assertIn("Least Similar Chunks", md_text)
        self.assertIn("Full Corpus Ranking Ledger", md_text)
        self.assertIn("Justification for Choosing Cosine Similarity", md_text)

        # Check JSON content
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("query", data)
        self.assertIn("top_similar_chunks", data)
        self.assertIn("least_similar_chunks", data)
        self.assertIn("full_rankings", data)
        self.assertIn("metric_justification", data)
        self.assertEqual(len(data["full_rankings"]), len(self.corpus_chunks))

        # Check top item attributes
        top_item = data["top_similar_chunks"][0]
        self.assertIn("rank", top_item)
        self.assertIn("score", top_item)
        self.assertIn("chunk_id", top_item)
        self.assertIn("source_document", top_item)
        self.assertIn("section", top_item)
        self.assertIn("source_text", top_item)
        self.assertEqual(top_item["rank"], 1)


if __name__ == "__main__":
    unittest.main()
