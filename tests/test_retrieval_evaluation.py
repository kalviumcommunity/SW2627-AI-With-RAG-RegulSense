"""Unit tests for RegulSense Retrieval Evaluation & Failure Diagnostics Module.

Validates:
1. Task 1: Labelled query dataset structure, ground truth coverage, and difficulty tiers.
2. Task 2: Recall@k measurement across top-1, top-3, and top-5.
3. Task 3: Precision@k, MRR, and MAP calculation with chunk relevance verdicts.
4. Task 4: Failure inspection logic, root-cause categorization, and remediation prescription.
5. Task 5: Exported Markdown and JSON evaluation reports.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.retrieval_evaluator import (
    LABELLED_COMPLIANCE_QUERIES,
    EvaluationBenchmarkReport,
    LabelledQuery,
    QueryEvaluationResult,
    RetrievalEvaluator,
    RetrievedChunkEvaluation,
)
from src.retriever import VectorRetriever
from src.vector_db import (
    DEFAULT_VECTOR_DIMENSION,
    VectorDatabaseManager,
    VectorRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRetrievalEvaluation(unittest.TestCase):
    """Test suite for retrieval quality benchmarking, precision/recall metrics, and failure diagnostics."""

    def setUp(self):
        """Initializes in-memory ChromaDB vector store and mock retriever for fast, isolated tests."""
        self.vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name="test_evaluation_collection",
            vector_dimension=DEFAULT_VECTOR_DIMENSION,
            distance_metric="cosine",
        )

        # Seed in-memory database with synthetic regulatory chunks
        self.corpus_chunks = [
            VectorRecord(
                id="digital_lending_compliance_note_html_tokenaware_001",
                embedding=self._unit_vec(0.1, 0.8, 0.0),
                document="Regulatory Advisory on Digital Lending: Direct disbursal and Key Fact Statement.",
                metadata={
                    "source_document": "digital_lending_compliance_note.html",
                    "filename": "digital_lending_compliance_note.html",
                    "file_type": ".html",
                    "section": "Preamble / Document Header",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 55,
                },
            ),
            VectorRecord(
                id="digital_lending_compliance_note_html_tokenaware_002",
                embedding=self._unit_vec(0.15, 0.85, 0.05),
                document=(
                    "Recovery Agent Code of Conduct: Strictly prohibited from contacting borrowers "
                    "before 8:00 AM or after 7:00 PM. Harassment prohibited."
                ),
                metadata={
                    "source_document": "digital_lending_compliance_note.html",
                    "filename": "digital_lending_compliance_note.html",
                    "file_type": ".html",
                    "section": "3. Code of Conduct for Recovery Agents",
                    "page_number": 1,
                    "chunk_index": 1,
                    "token_count": 68,
                },
            ),
            VectorRecord(
                id="cyber_resilience_framework_pdf_tokenaware_001",
                embedding=self._unit_vec(0.0, 0.1, 0.9),
                document="Severity 1 cyber security incidents shall be reported to CERT-In and RBI within 6 hours.",
                metadata={
                    "source_document": "cyber_resilience_framework.pdf",
                    "filename": "cyber_resilience_framework.pdf",
                    "file_type": ".pdf",
                    "section": "Annex 1: Incident Reporting Architecture",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 60,
                },
            ),
            VectorRecord(
                id="circular_dor_2024_108_txt_tokenaware_002",
                embedding=self._unit_vec(0.9, 0.1, 0.0),
                document="V-CIP onboarding: Automated facial match system shall achieve confidence score not less than 95%.",
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
                document="Record retention obligations: Banks must maintain all necessary records for at least 5 years.",
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
                embedding=self._unit_vec(0.65, 0.2, 0.45),
                document="PML Rules Beneficial Ownership threshold: 10% controlling interest for corporate entities.",
                metadata={
                    "source_document": "guidelines_cdd_pml_rules.md",
                    "filename": "guidelines_cdd_pml_rules.md",
                    "file_type": ".md",
                    "section": "1. Statutory Framework and Definition of Beneficial Owner",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 58,
                },
            ),
        ]
        self.vdb.insert_records(self.corpus_chunks)

        # Mock OpenAI Client
        self.mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = self._unit_vec(0.15, 0.85, 0.05)
        mock_resp.data = [mock_data]
        self.mock_client.embeddings.create.return_value = mock_resp

        self.retriever = VectorRetriever(
            vector_db=self.vdb,
            collection_name="test_evaluation_collection",
            openai_client=self.mock_client,
        )
        self.evaluator = RetrievalEvaluator(
            vector_retriever=self.retriever,
        )

    def _unit_vec(self, x: float, y: float, z: float) -> list:
        vec = [0.0] * DEFAULT_VECTOR_DIMENSION
        vec[0] = x
        vec[1] = y
        vec[2] = z
        norm = np.linalg.norm(vec)
        return [float(v / norm) for v in vec]

    # -------------------------------------------------------------------------
    # Task 1: Labelled Query Dataset Specification
    # -------------------------------------------------------------------------

    def test_task1_labelled_query_dataset_properties(self):
        """Verify the labelled query dataset contains complete annotations across difficulty tiers."""
        self.assertGreaterEqual(len(LABELLED_COMPLIANCE_QUERIES), 8)

        query_types = {q.query_type for q in LABELLED_COMPLIANCE_QUERIES}
        self.assertIn("standard", query_types)
        self.assertIn("colloquial_phrasing", query_types)
        self.assertIn("cross_domain_ambiguity", query_types)
        self.assertIn("out_of_corpus", query_types)

        difficulties = {q.difficulty for q in LABELLED_COMPLIANCE_QUERIES}
        self.assertIn("Easy", difficulties)
        self.assertIn("Medium", difficulties)
        self.assertIn("Hard", difficulties)

        for q in LABELLED_COMPLIANCE_QUERIES:
            self.assertIsInstance(q, LabelledQuery)
            self.assertTrue(q.query_id.startswith("LQ"))
            self.assertGreater(len(q.query_text), 15)
            self.assertGreater(len(q.intent_description), 10)
            d = q.to_dict()
            self.assertEqual(d["query_id"], q.query_id)
            self.assertEqual(d["query_text"], q.query_text)

    # -------------------------------------------------------------------------
    # Task 2: Measure Recall@k
    # -------------------------------------------------------------------------

    def test_task2_measure_recall_at_k(self):
        """Verify Recall@k accurately measures the fraction of ground truth retrieved."""
        q1 = LABELLED_COMPLIANCE_QUERIES[0]  # Recovery hours (ground truth has 2 chunks)
        res = self.evaluator.evaluate_query(query=q1, max_k=5)

        self.assertIsInstance(res, QueryEvaluationResult)
        self.assertGreaterEqual(res.recall_at_1, 0.0)
        self.assertLessEqual(res.recall_at_1, 1.0)
        self.assertGreaterEqual(res.recall_at_3, res.recall_at_1)
        self.assertGreaterEqual(res.recall_at_5, res.recall_at_3)

    # -------------------------------------------------------------------------
    # Task 3: Report Precision, MRR, and Quality Signals
    # -------------------------------------------------------------------------

    def test_task3_measure_precision_mrr_and_map(self):
        """Verify Precision@k, MRR, and MAP computations."""
        q1 = LABELLED_COMPLIANCE_QUERIES[0]
        res = self.evaluator.evaluate_query(query=q1, max_k=5)

        self.assertGreaterEqual(res.precision_at_1, 0.0)
        self.assertLessEqual(res.precision_at_1, 1.0)
        self.assertGreaterEqual(res.precision_at_3, 0.0)
        self.assertLessEqual(res.precision_at_3, 1.0)
        self.assertGreaterEqual(res.reciprocal_rank, 0.0)
        self.assertLessEqual(res.reciprocal_rank, 1.0)
        self.assertGreaterEqual(res.average_precision, 0.0)
        self.assertLessEqual(res.average_precision, 1.0)

        # Check chunk evaluation verdicts
        verdicts = {c.relevance_verdict for c in res.retrieved_chunks}
        self.assertTrue(any(v in {"RELEVANT", "PERIPHERAL", "IRRELEVANT"} for v in verdicts))

    # -------------------------------------------------------------------------
    # Task 4: Inspect Failures & Root Cause Diagnostics
    # -------------------------------------------------------------------------

    def test_task4_failure_inspection_out_of_corpus(self):
        """Verify negative control / out-of-corpus query is flagged with Out-of-Corpus diagnosis."""
        q_negative = next(q for q in LABELLED_COMPLIANCE_QUERIES if q.query_type == "out_of_corpus")
        res = self.evaluator.evaluate_query(query=q_negative, max_k=3)

        self.assertTrue(res.is_failure)
        self.assertEqual(res.failure_category, "Out-of-Corpus Query")
        self.assertIn("Basel III", res.failure_diagnosis)
        self.assertIn("cutoff threshold", res.prescribed_remediation)

    def test_task4_inspect_failure_colloquial_gap(self):
        """Verify failure inspection categorizes colloquial vocabulary gaps when ground truth is missing."""
        q_colloquial = next(q for q in LABELLED_COMPLIANCE_QUERIES if q.query_type == "colloquial_phrasing")
        # Simulated retrieved chunks where ground truth was missed
        fake_retrieved = [
            RetrievedChunkEvaluation(
                chunk_id="other_chunk_001",
                rank=1,
                similarity_score=0.45,
                source_document="unrelated.txt",
                section="General",
                is_ground_truth=False,
                is_acceptable_source=False,
                relevance_verdict="IRRELEVANT",
                snippet="Unrelated text",
            ),
            RetrievedChunkEvaluation(
                chunk_id="other_chunk_002",
                rank=2,
                similarity_score=0.42,
                source_document="unrelated.txt",
                section="General",
                is_ground_truth=False,
                is_acceptable_source=False,
                relevance_verdict="IRRELEVANT",
                snippet="Unrelated text",
            ),
            RetrievedChunkEvaluation(
                chunk_id="other_chunk_003",
                rank=3,
                similarity_score=0.40,
                source_document="unrelated.txt",
                section="General",
                is_ground_truth=False,
                is_acceptable_source=False,
                relevance_verdict="IRRELEVANT",
                snippet="Unrelated text",
            ),
        ]

        is_fail, cat, diag, remed = self.evaluator.inspect_failure(
            query=q_colloquial,
            retrieved=fake_retrieved,
        )
        self.assertTrue(is_fail)
        self.assertEqual(cat, "Colloquial Vocabulary Gap")
        self.assertIn("layman terminology", diag)
        self.assertIn("query expansion", remed)

    # -------------------------------------------------------------------------
    # Task 5: Export Evaluation Results (Artifact Verification)
    # -------------------------------------------------------------------------

    def test_task5_export_evaluation_artifacts(self):
        """Verify export_evaluation_artifacts generates valid Markdown and JSON files with diagnostics."""
        report = self.evaluator.run_benchmark(
            queries=LABELLED_COMPLIANCE_QUERIES[:3],
            max_k=3,
        )

        self.assertIsInstance(report, EvaluationBenchmarkReport)
        self.assertEqual(report.total_queries, 3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_md = Path(tmp_dir) / "test_evaluation.md"
            tmp_json = Path(tmp_dir) / "test_evaluation.json"

            out_md, out_json = self.evaluator.export_evaluation_artifacts(
                report=report,
                output_markdown_path=tmp_md,
                output_json_path=tmp_json,
            )

            self.assertTrue(out_md.exists())
            self.assertTrue(out_json.exists())

            # Validate JSON
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("timestamp", data)
            self.assertIn("collection_name", data)
            self.assertIn("summary_metrics", data)
            self.assertIn("mean_recall_at_3", data["summary_metrics"])
            self.assertIn("mean_precision_at_3", data["summary_metrics"])
            self.assertIn("evaluations", data)
            self.assertIn("remediation_roadmap", data)

            # Validate Markdown
            md_content = out_md.read_text(encoding="utf-8")
            self.assertIn("# RegulSense: Systematic Retrieval Quality Evaluation & Failure Diagnostics Report", md_content)
            self.assertIn("Executive Summary & Aggregate Retrieval Metrics", md_content)
            self.assertIn("Labelled Benchmark Dataset Specification", md_content)
            self.assertIn("Query-by-Query Retrieval Performance Ledger", md_content)
            self.assertIn("Failure Inspection & Root Cause Diagnostics", md_content)
            self.assertIn("Remediation Roadmap for Production Deployment", md_content)


if __name__ == "__main__":
    unittest.main()
