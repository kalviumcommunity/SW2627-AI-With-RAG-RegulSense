"""Unit tests for RegulSense Retrieval Relevance Tuning & Evaluation Module.

Validates:
1. Task 1: Canonical benchmark test queries definition, ground truth targets, and domain metadata.
2. Task 2: Comparative retrieval configurations across k, hybrid weights, thresholds, and filters.
3. Task 3: Information Retrieval relevance measures (Hit Rate @ k, Top-1 Hit, MRR, Precision @ k).
4. Task 4: Automated selection and empirical justification of the best-performing settings.
5. Task 5: Exported Markdown and JSON tuning audit reports.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.filtered_retriever import FilteredRetriever
from src.retrieval_tuner import (
    BENCHMARK_TEST_QUERIES,
    DEFAULT_RETRIEVAL_CONFIGS,
    ConfigPerformanceSummary,
    QueryResultEvaluation,
    RetrievalConfig,
    RetrievalTuner,
    RetrievedChunkView,
    TestQuery,
    TuningExperimentReport,
)
from src.retriever import VectorRetriever
from src.vector_db import (
    DEFAULT_VECTOR_DIMENSION,
    VectorDatabaseManager,
    VectorRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRetrievalTuning(unittest.TestCase):
    """Test suite for retrieval relevance benchmarking and configuration optimization."""

    def setUp(self):
        """Sets up an in-memory ChromaDB instance and mock client for deterministic testing."""
        self.vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name="test_tuning_collection",
            vector_dimension=DEFAULT_VECTOR_DIMENSION,
            distance_metric="cosine",
        )

        # Populate with 6 synthetic regulatory chunks mirroring real corpus chunks
        self.mock_records = [
            VectorRecord(
                id="digital_lending_compliance_note_html_tokenaware_001",
                embedding=self._unit_vec(0.1, 0.9, 0.0),
                document=(
                    "RegulSense Regulatory Advisory: Digital Lending & Fair Practices Code. "
                    "All loan disbursals and repayments must be executed solely between the bank "
                    "account of the borrower and the regulated entity (RE). Key Fact Statement (KFS) mandate."
                ),
                metadata={
                    "source_document": "digital_lending_compliance_note.html",
                    "filename": "digital_lending_compliance_note.html",
                    "file_type": ".html",
                    "section": "Preamble / Document Header",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 60,
                },
            ),
            VectorRecord(
                id="digital_lending_compliance_note_html_tokenaware_002",
                embedding=self._unit_vec(0.15, 0.85, 0.05),
                document=(
                    "Code of Conduct for Recovery Agents: Regulated entities and their recovery agents "
                    "are strictly prohibited from contacting borrowers before 8:00 AM or after 7:00 PM. "
                    "Harassment, verbal intimidation, or persistent unsolicited calls are strictly prohibited."
                ),
                metadata={
                    "source_document": "digital_lending_compliance_note.html",
                    "filename": "digital_lending_compliance_note.html",
                    "file_type": ".html",
                    "section": "3. Code of Conduct for Recovery Agents",
                    "page_number": 1,
                    "chunk_index": 1,
                    "token_count": 65,
                },
            ),
            VectorRecord(
                id="cyber_resilience_framework_pdf_tokenaware_001",
                embedding=self._unit_vec(0.0, 0.1, 0.95),
                document=(
                    "Cyber Security Framework in Banks: Incident reporting baseline requirements. "
                    "All cyber security incidents of Severity 1 and 2 shall be reported to CERT-In and RBI "
                    "within 6 hours of detection without fail."
                ),
                metadata={
                    "source_document": "cyber_resilience_framework.pdf",
                    "filename": "cyber_resilience_framework.pdf",
                    "file_type": ".pdf",
                    "section": "Annex 1: Incident Reporting Architecture",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 58,
                },
            ),
            VectorRecord(
                id="circular_dor_2024_108_txt_tokenaware_002",
                embedding=self._unit_vec(0.9, 0.1, 0.0),
                document=(
                    "Customer Due Diligence (CDD) Requirements and V-CIP Verification. "
                    "The automated facial match system shall achieve a confidence score of not less than 95% "
                    "with liveliness check prior to account opening."
                ),
                metadata={
                    "source_document": "circular_dor_2024_108.txt",
                    "filename": "circular_dor_2024_108.txt",
                    "file_type": ".txt",
                    "section": "2. Customer Due Diligence (CDD) Requirements",
                    "page_number": 1,
                    "chunk_index": 1,
                    "token_count": 62,
                },
            ),
            VectorRecord(
                id="circular_dor_2024_108_txt_tokenaware_004",
                embedding=self._unit_vec(0.7, 0.3, 0.1),
                document=(
                    "Record Retention Obligations: Banks must maintain all necessary records of transactions "
                    "for a minimum period of 5 years from the date of transaction or business cessation."
                ),
                metadata={
                    "source_document": "circular_dor_2024_108.txt",
                    "filename": "circular_dor_2024_108.txt",
                    "file_type": ".txt",
                    "section": "5. Record Retention Obligations",
                    "page_number": 2,
                    "chunk_index": 3,
                    "token_count": 55,
                },
            ),
            VectorRecord(
                id="guidelines_cdd_pml_rules_md_tokenaware_001",
                embedding=self._unit_vec(0.65, 0.2, 0.5),
                document=(
                    "PML Rules 2005 Statutory Framework: Beneficial Ownership (BO) identification. "
                    "For companies, controlling ownership interest is defined as exceeding 10% of shares or capital."
                ),
                metadata={
                    "source_document": "guidelines_cdd_pml_rules.md",
                    "filename": "guidelines_cdd_pml_rules.md",
                    "file_type": ".md",
                    "section": "1. Statutory Framework and Definition of Beneficial Owner",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 54,
                },
            ),
        ]
        self.vdb.insert_records(self.mock_records)

        # Mock OpenAI Client
        self.mock_client = MagicMock()
        mock_embedding_resp = MagicMock()
        mock_data = MagicMock()
        # Default mock vector
        mock_data.embedding = self._unit_vec(0.15, 0.85, 0.05)
        mock_embedding_resp.data = [mock_data]
        self.mock_client.embeddings.create.return_value = mock_embedding_resp

        self.retriever = VectorRetriever(
            vector_db=self.vdb,
            collection_name="test_tuning_collection",
            openai_client=self.mock_client,
        )
        self.filtered_retriever = FilteredRetriever(
            retriever=self.retriever,
        )
        self.tuner = RetrievalTuner(
            filtered_retriever=self.filtered_retriever,
        )

    def _unit_vec(self, x: float, y: float, z: float) -> list:
        vec = [0.0] * DEFAULT_VECTOR_DIMENSION
        vec[0] = x
        vec[1] = y
        vec[2] = z
        norm = np.linalg.norm(vec)
        return [float(v / norm) for v in vec]

    # -------------------------------------------------------------------------
    # Task 1: Define Test Queries
    # -------------------------------------------------------------------------

    def test_task1_benchmark_test_queries_structure(self):
        """Verify the canonical benchmark queries are defined with required ground truth fields."""
        self.assertGreaterEqual(len(BENCHMARK_TEST_QUERIES), 5)
        seen_ids = set()

        for q in BENCHMARK_TEST_QUERIES:
            self.assertIsInstance(q, TestQuery)
            self.assertTrue(q.query_id.startswith("Q"))
            self.assertNotIn(q.query_id, seen_ids)
            seen_ids.add(q.query_id)

            self.assertGreater(len(q.query_text), 20)
            self.assertGreater(len(q.regulatory_domain), 3)
            self.assertGreater(len(q.expected_sources), 0)
            self.assertGreater(len(q.expected_chunk_ids), 0)
            self.assertGreater(len(q.target_keywords), 0)

            # Check dictionary serialization
            d = q.to_dict()
            self.assertEqual(d["query_id"], q.query_id)
            self.assertEqual(d["query_text"], q.query_text)
            self.assertEqual(d["expected_sources"], q.expected_sources)

    # -------------------------------------------------------------------------
    # Task 2: Compare Retrieval Settings
    # -------------------------------------------------------------------------

    def test_task2_retrieval_configurations_matrix(self):
        """Verify the configuration matrix contains distinct parameter combinations."""
        self.assertGreaterEqual(len(DEFAULT_RETRIEVAL_CONFIGS), 4)

        k_values = {c.k for c in DEFAULT_RETRIEVAL_CONFIGS}
        self.assertIn(2, k_values)
        self.assertIn(5, k_values)
        self.assertIn(3, k_values)

        hybrid_flags = {c.use_hybrid for c in DEFAULT_RETRIEVAL_CONFIGS}
        self.assertIn(True, hybrid_flags)
        self.assertIn(False, hybrid_flags)

        thresholds = [c.score_threshold for c in DEFAULT_RETRIEVAL_CONFIGS if c.score_threshold > 0.0]
        self.assertGreater(len(thresholds), 0)

        filter_flags = [c.apply_domain_filter for c in DEFAULT_RETRIEVAL_CONFIGS if c.apply_domain_filter]
        self.assertGreater(len(filter_flags), 0)

    # -------------------------------------------------------------------------
    # Task 3: Report Relevance Measures
    # -------------------------------------------------------------------------

    def test_task3_evaluate_query_metrics(self):
        """Verify evaluate_query accurately computes Hit@k, Top-1 Hit, MRR, and Precision@k."""
        q1 = BENCHMARK_TEST_QUERIES[0]
        cfg = RetrievalConfig(
            config_id="test_cfg",
            name="Test Config",
            description="Test description",
            k=2,
            use_hybrid=False,
            alpha=1.0,
            score_threshold=0.0,
            apply_domain_filter=False,
        )

        res = self.tuner.evaluate_query(query=q1, config=cfg)
        self.assertIsInstance(res, QueryResultEvaluation)
        self.assertEqual(res.query_id, "Q1")
        self.assertEqual(res.config_id, "test_cfg")
        self.assertEqual(len(res.retrieved_chunks), 2)

        # Because mock query vector aligns with chunk 002 (rank 1), top-1 hit should be true
        self.assertTrue(res.hit_at_k)
        self.assertTrue(res.top1_hit)
        self.assertEqual(res.reciprocal_rank, 1.0)
        self.assertGreater(res.precision_at_k, 0.0)

    def test_task3_score_threshold_noise_gating(self):
        """Verify setting a score threshold filters out sub-threshold chunks and increments noise count."""
        q1 = BENCHMARK_TEST_QUERIES[0]
        # Low threshold (keeps all)
        cfg_low = RetrievalConfig(
            config_id="cfg_low",
            name="Low Thresh",
            description="",
            k=5,
            score_threshold=0.1,
        )
        res_low = self.tuner.evaluate_query(query=q1, config=cfg_low)

        # High threshold (drops low matches)
        cfg_high = RetrievalConfig(
            config_id="cfg_high",
            name="High Thresh",
            description="",
            k=5,
            score_threshold=0.98,
        )
        res_high = self.tuner.evaluate_query(query=q1, config=cfg_high)

        self.assertGreater(res_high.noise_chunks_filtered, 0)
        self.assertLess(res_high.chunks_after_threshold, res_low.chunks_after_threshold)

    def test_task3_evaluate_configuration_summary(self):
        """Verify evaluate_configuration aggregates metrics correctly across queries."""
        cfg = DEFAULT_RETRIEVAL_CONFIGS[0]
        summary, evals = self.tuner.evaluate_configuration(
            config=cfg,
            queries=BENCHMARK_TEST_QUERIES[:2],
        )

        self.assertIsInstance(summary, ConfigPerformanceSummary)
        self.assertEqual(summary.config_id, cfg.config_id)
        self.assertEqual(summary.total_queries, 2)
        self.assertGreaterEqual(summary.hit_rate_at_k, 0.0)
        self.assertLessEqual(summary.hit_rate_at_k, 1.0)
        self.assertGreaterEqual(summary.top1_hit_rate, 0.0)
        self.assertLessEqual(summary.top1_hit_rate, 1.0)
        self.assertGreaterEqual(summary.mean_reciprocal_rank, 0.0)
        self.assertLessEqual(summary.mean_reciprocal_rank, 1.0)
        self.assertEqual(len(evals), 2)

    # -------------------------------------------------------------------------
    # Task 4: Choose and Justify Best Settings
    # -------------------------------------------------------------------------

    def test_task4_choose_best_settings_and_justification(self):
        """Verify automated selection picks the highest-utility configuration with a detailed justification."""
        summaries = [
            ConfigPerformanceSummary(
                config_id="cfg_baseline",
                config_name="Baseline Dense",
                k=2,
                settings_summary="k=2",
                total_queries=5,
                hit_rate_at_k=0.6,
                top1_hit_rate=0.4,
                mean_reciprocal_rank=0.5,
                mean_precision_at_k=0.3,
                mean_score=0.45,
                total_noise_chunks_filtered=0,
            ),
            ConfigPerformanceSummary(
                config_id="cfg_optimal",
                config_name="Production Optimal Hybrid",
                k=3,
                settings_summary="k=3, hybrid=True",
                total_queries=5,
                hit_rate_at_k=1.0,
                top1_hit_rate=1.0,
                mean_reciprocal_rank=1.0,
                mean_precision_at_k=0.85,
                mean_score=0.58,
                total_noise_chunks_filtered=3,
            ),
        ]

        best_id, best_name, justification = RetrievalTuner.choose_best_settings(summaries)
        self.assertEqual(best_id, "cfg_optimal")
        self.assertEqual(best_name, "Production Optimal Hybrid")
        self.assertIn("cfg_optimal", justification)
        self.assertIn("Hit Rate", justification)
        self.assertIn("Top-1 Hit Rate", justification)
        self.assertIn("Mean Reciprocal Rank", justification)

    # -------------------------------------------------------------------------
    # Task 5: Export Tuning Artifacts
    # -------------------------------------------------------------------------

    def test_task5_export_tuning_artifacts(self):
        """Verify export_tuning_artifacts generates valid Markdown and JSON files."""
        report = self.tuner.run_tuning_experiment(
            test_queries=BENCHMARK_TEST_QUERIES[:2],
            configurations=DEFAULT_RETRIEVAL_CONFIGS[:2],
        )

        self.assertIsInstance(report, TuningExperimentReport)
        self.assertEqual(len(report.test_queries), 2)
        self.assertEqual(len(report.configurations), 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_md = Path(tmp_dir) / "test_retrieval_tuning.md"
            tmp_json = Path(tmp_dir) / "test_retrieval_tuning.json"

            out_md, out_json = self.tuner.export_tuning_artifacts(
                report=report,
                output_markdown_path=tmp_md,
                output_json_path=tmp_json,
            )

            self.assertTrue(out_md.exists())
            self.assertTrue(out_json.exists())

            # Check JSON contents
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("timestamp", data)
            self.assertIn("collection_name", data)
            self.assertIn("test_queries", data)
            self.assertIn("configurations", data)
            self.assertIn("evaluations", data)
            self.assertIn("summaries", data)
            self.assertIn("best_config_id", data)
            self.assertIn("justification", data)

            # Check Markdown contents
            md_text = out_md.read_text(encoding="utf-8")
            self.assertIn("# RegulSense: Retrieval Relevance Tuning & Settings Optimization", md_text)
            self.assertIn("Executive Summary & Optimization Scorecard", md_text)
            self.assertIn("Benchmark Test Query Suite", md_text)
            self.assertIn("Per-Query Retrieval Breakdown Across Configurations", md_text)
            self.assertIn("Architectural Selection & Empirical Justification", md_text)
            self.assertIn("PRODUCTION_RETRIEVAL_CONFIG", md_text)


if __name__ == "__main__":
    unittest.main()
