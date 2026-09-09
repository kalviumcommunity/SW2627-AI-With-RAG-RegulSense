"""Unit tests for RegulSense Embeddings Fundamentals Module.

Validates:
1. Task 1: Embedding generation across sample texts including similar pairs and unrelated text.
2. Task 2: Vector dimension reporting and validation of strict length uniformity across all samples.
3. Task 3: Cosine similarity mathematics and verification that semantically similar pairs score higher
   than dissimilar pairs.
4. Task 4: Conceptual explanation note integrity (contrasting dense vectors vs. random IDs and keyword counts).
5. Task 5: Existence and validity of exported demonstration artifacts (Markdown and JSON).
"""

import json
from pathlib import Path
import unittest

import numpy as np

from src.embeddings import (
    DEFAULT_SAMPLE_TEXTS,
    EXPLANATION_NOTE,
    DimensionReport,
    EmbeddingGenerator,
    TextEmbeddingSample,
    compute_cosine_similarity,
    generate_demonstration_markdown,
    run_embedding_demonstration,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestEmbeddingsFundamentals(unittest.TestCase):
    """Test suite for embedding generation, dimension reporting, and similarity analysis."""

    @classmethod
    def setUpClass(cls):
        """Initialize generator and generate embeddings once for the test suite."""
        cls.generator = EmbeddingGenerator()
        cls.samples = cls.generator.process_samples()
        cls.dim_report = cls.generator.report_dimensions(cls.samples)
        cls.comparisons, cls.matrix = cls.generator.compare_sample_pairs(cls.samples)

    # -------------------------------------------------------------------------
    # Task 1: Generate Embeddings
    # -------------------------------------------------------------------------

    def test_task1_generate_embeddings_count_and_types(self):
        """Verify embeddings are generated for all configured sample texts with float coordinates."""
        self.assertGreaterEqual(len(self.samples), 3)
        self.assertEqual(len(self.samples), len(DEFAULT_SAMPLE_TEXTS))

        for sample in self.samples:
            self.assertIsInstance(sample.id, str)
            self.assertIsInstance(sample.text, str)
            self.assertGreater(len(sample.text), 10)
            self.assertIsInstance(sample.vector, list)
            self.assertGreater(len(sample.vector), 0)
            self.assertTrue(all(isinstance(x, float) for x in sample.vector))

    def test_task1_sample_texts_include_similar_and_unrelated(self):
        """Confirm sample texts include at least one similar pair and at least one unrelated text."""
        sample_ids = {s.id for s in self.samples}
        self.assertIn("aml_cdd_1", sample_ids)
        self.assertIn("aml_cdd_2", sample_ids)
        self.assertIn("unrelated_culinary", sample_ids)

    # -------------------------------------------------------------------------
    # Task 2: Report Vector Dimension & Verify Uniformity
    # -------------------------------------------------------------------------

    def test_task2_vector_dimension_report(self):
        """Verify that dimension report is accurate and every vector has the same length."""
        self.assertTrue(self.dim_report.is_uniform)
        self.assertGreater(self.dim_report.common_dimension, 0)
        self.assertEqual(self.dim_report.total_samples, len(self.samples))

        # Every sample vector must equal the reported common dimension
        for sample in self.samples:
            self.assertEqual(
                len(sample.vector),
                self.dim_report.common_dimension,
                f"Sample '{sample.id}' vector length does not match common dimension {self.dim_report.common_dimension}",
            )

    def test_task2_dimension_uniformity_failure_detection(self):
        """Verify report_dimensions detects non-uniform vectors if dimensions differ."""
        mismatched_samples = [
            TextEmbeddingSample(id="s1", category="Cat", text="A", dimension=384, vector=[0.1] * 384),
            TextEmbeddingSample(id="s2", category="Cat", text="B", dimension=128, vector=[0.1] * 128),
        ]
        report = EmbeddingGenerator.report_dimensions(mismatched_samples)
        self.assertFalse(report.is_uniform)
        self.assertEqual(report.common_dimension, -1)
        self.assertIn("FAILED", report.status)

    # -------------------------------------------------------------------------
    # Task 3: Compare Similar vs. Dissimilar Texts (Cosine Similarity)
    # -------------------------------------------------------------------------

    def test_task3_cosine_similarity_math(self):
        """Verify cosine similarity mathematical boundary conditions."""
        # Identical vectors -> 1.0
        v1 = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(compute_cosine_similarity(v1, v1), 1.0, places=5)

        # Orthogonal vectors -> 0.0
        v_ortho_a = [1.0, 0.0]
        v_ortho_b = [0.0, 1.0]
        self.assertAlmostEqual(compute_cosine_similarity(v_ortho_a, v_ortho_b), 0.0, places=5)

        # Opposite vectors -> -1.0
        v_opp_a = [1.0, 2.0]
        v_opp_b = [-1.0, -2.0]
        self.assertAlmostEqual(compute_cosine_similarity(v_opp_a, v_opp_b), -1.0, places=5)

        # Zero vector -> 0.0
        v_zero = [0.0, 0.0]
        self.assertEqual(compute_cosine_similarity(v1, v_zero), 0.0)

    def test_task3_similar_pair_scores_higher_than_dissimilar_pair(self):
        """Confirm that semantically similar pairs score higher than dissimilar pairs."""
        similar_comparisons = [c for c in self.comparisons if c.pair_type == "similar"]
        dissimilar_comparisons = [c for c in self.comparisons if c.pair_type == "dissimilar"]

        self.assertGreater(len(similar_comparisons), 0)
        self.assertGreater(len(dissimilar_comparisons), 0)

        # Check each similar pair vs each dissimilar pair
        for sim_comp in similar_comparisons:
            for dis_comp in dissimilar_comparisons:
                self.assertGreater(
                    sim_comp.cosine_similarity,
                    dis_comp.cosine_similarity,
                    f"Expected similar pair ({sim_comp.id_a} vs {sim_comp.id_b}: {sim_comp.cosine_similarity}) "
                    f"to score higher than dissimilar pair ({dis_comp.id_a} vs {dis_comp.id_b}: {dis_comp.cosine_similarity})",
                )

        # Check average scores
        avg_similar = sum(c.cosine_similarity for c in similar_comparisons) / len(similar_comparisons)
        avg_dissimilar = sum(c.cosine_similarity for c in dissimilar_comparisons) / len(dissimilar_comparisons)
        self.assertGreater(avg_similar, avg_dissimilar)
        self.assertGreater(avg_similar - avg_dissimilar, 0.20, "Expected significant semantic margin")

    def test_task3_similarity_matrix_properties(self):
        """Confirm symmetry and identity along diagonal for the full similarity matrix."""
        n = len(self.samples)
        for i in range(n):
            self.assertEqual(self.matrix[i][i], 1.0, "Diagonal elements must equal 1.0 (self-similarity)")
            for j in range(n):
                self.assertEqual(
                    self.matrix[i][j],
                    self.matrix[j][i],
                    f"Matrix must be symmetric: matrix[{i}][{j}] != matrix[{j}][{i}]",
                )

    # -------------------------------------------------------------------------
    # Task 4: Explain What Vectors Represent
    # -------------------------------------------------------------------------

    def test_task4_explanation_note_content(self):
        """Verify the conceptual note explains meaning vs. random IDs vs. keyword counts."""
        self.assertIsInstance(EXPLANATION_NOTE, str)
        self.assertGreater(len(EXPLANATION_NOTE), 200)

        # Must mention random IDs and why they lack geometric/spatial meaning
        self.assertIn("Random IDs", EXPLANATION_NOTE)
        self.assertTrue(
            "geometric" in EXPLANATION_NOTE.lower() or "spatial" in EXPLANATION_NOTE.lower(),
            "Explanation must explain geometric/spatial properties vs random IDs",
        )

        # Must mention keyword counts / Bag-of-Words / TF-IDF
        self.assertTrue(
            "keyword" in EXPLANATION_NOTE.lower() or "bag-of-words" in EXPLANATION_NOTE.lower(),
            "Explanation must contrast with keyword counts/frequencies",
        )

        # Must explain dense semantic representation
        self.assertTrue(
            "semantic" in EXPLANATION_NOTE.lower() and "meaning" in EXPLANATION_NOTE.lower(),
            "Explanation must explain dense semantic meaning",
        )

    # -------------------------------------------------------------------------
    # Task 5: Output Artifact Integrity
    # -------------------------------------------------------------------------

    def test_task5_output_artifacts_exist_and_valid(self):
        """Verify demonstration artifacts are written, valid, and contain required sections."""
        md_file = PROJECT_ROOT / "outputs" / "embedding_fundamentals_demonstration.md"
        json_file = PROJECT_ROOT / "outputs" / "sample_embeddings.json"

        # If not already generated, run demonstration
        if not md_file.exists() or not json_file.exists():
            run_embedding_demonstration(md_file, json_file)

        self.assertTrue(md_file.exists(), f"Markdown report not found at {md_file}")
        self.assertTrue(json_file.exists(), f"JSON export not found at {json_file}")

        # Check Markdown content
        md_text = md_file.read_text(encoding="utf-8")
        self.assertIn("Vector Space Dimensionality", md_text)
        self.assertIn("Dimension Uniformity Verification", md_text)
        self.assertIn("Similar Scores Higher than Dissimilar", md_text)
        self.assertIn("Full Pairwise Cosine Similarity Matrix", md_text)
        self.assertIn("What Embedding Vectors Actually Represent", md_text)

        # Check JSON content
        json_content = json.loads(json_file.read_text(encoding="utf-8"))
        self.assertIn("metadata", json_content)
        self.assertIn("dimension_report", json_content)
        self.assertIn("samples", json_content)
        self.assertIn("key_comparisons", json_content)
        self.assertIn("similarity_matrix", json_content)
        self.assertIn("conceptual_explanation", json_content)
        self.assertTrue(json_content["dimension_report"]["is_uniform"])
        self.assertEqual(len(json_content["samples"]), len(self.samples))


if __name__ == "__main__":
    unittest.main()
