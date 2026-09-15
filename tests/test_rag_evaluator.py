"""Unit and Integration Tests for RAG Evaluator and Quality Benchmark Suite.

Tests cover:
- Task 1: Test set preparation, validation, and schema consistency.
- Task 2: Correctness and grounding scoring (answerable and unanswerable cases).
- Task 3: Citation accuracy checks, fabrication detection, and claim-to-chunk verification.
- Task 4: Failure diagnostics, root cause classification, and summary aggregation.
- Task 5: Artifact serialization and report generation (Markdown and JSON).
"""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.citation_engine import CitationAuditReport, CitedAnswerOutput
from src.hallucination_guardrails import GuardrailExecutionResult, RetrievalQualityAssessment
from src.rag_evaluator import (
    CitationCheckResult,
    EvaluationResult,
    EvaluationSummary,
    EvaluationTestCase,
    RAGEvaluator,
    STANDARD_EVALUATION_TEST_SET,
    export_evaluation_artifacts,
)
from src.vector_db import RetrievedRecord


class TestEvaluationTestCaseAndSet(unittest.TestCase):
    """Tests for Task 1: Prepare a Test Set."""

    def test_standard_test_set_completeness(self):
        """Verify the test set contains expected diverse test cases."""
        self.assertGreaterEqual(len(STANDARD_EVALUATION_TEST_SET), 8)
        categories = {tc.category for tc in STANDARD_EVALUATION_TEST_SET}
        self.assertIn("Cyber Incident Reporting", categories)
        self.assertIn("Forensic Audit Trail Preservation", categories)
        self.assertIn("Customer Due Diligence (CDD)", categories)
        self.assertIn("AML Record Retention", categories)
        self.assertIn("Digital Lending Conduct", categories)
        self.assertIn("Out-of-Corpus Fallback & Refusal", categories)

    def test_test_case_schema(self):
        """Verify test case data structure and serialization."""
        tc = STANDARD_EVALUATION_TEST_SET[0]
        self.assertTrue(tc.test_id.startswith("EVAL-"))
        self.assertTrue(tc.is_answerable)
        self.assertGreater(len(tc.key_facts), 0)
        self.assertGreater(len(tc.expected_sources), 0)

        tc_dict = tc.to_dict()
        self.assertEqual(tc_dict["test_id"], tc.test_id)
        self.assertEqual(tc_dict["expected_sources"], tc.expected_sources)

    def test_unanswerable_test_case_present(self):
        """Verify at least one out-of-corpus unanswerable query is included."""
        unanswerable = [tc for tc in STANDARD_EVALUATION_TEST_SET if not tc.is_answerable]
        self.assertEqual(len(unanswerable), 1)
        self.assertEqual(unanswerable[0].test_id, "EVAL-08")
        self.assertEqual(unanswerable[0].expected_sources, [])


class TestScoringCorrectnessAndGrounding(unittest.TestCase):
    """Tests for Task 2: Score Correctness and Grounding."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_citation_engine = MagicMock()
        self.mock_guardrail = MagicMock()
        self.evaluator = RAGEvaluator(
            openai_client=self.mock_client,
            retriever=self.mock_retriever,
            citation_engine=self.mock_citation_engine,
            guardrail=self.mock_guardrail,
        )
        self.test_case = EvaluationTestCase(
            test_id="TEST-01",
            query="What are reporting timelines for cyber incidents?",
            category="Cyber Incident Reporting",
            expected_answer="Must report within 6 hours to RBI and CERT-In, followed by 7 business days for forensic analysis.",
            expected_sources=["cyber_resilience.pdf"],
            expected_chunk_ids=["chunk_001"],
            key_facts=["6 hours", "7 business days", "CERT-In", "CSITE"],
            is_answerable=True,
        )

    def test_score_correctness_high_coverage(self):
        """Verify high correctness score when all key facts are present."""
        ans = (
            "Banks must report incidents to RBI CSITE and CERT-In within 6 hours of detection. "
            "A comprehensive forensic analysis report must be submitted within 7 business days."
        )
        score, matched, missing = self.evaluator.score_correctness(ans, self.test_case)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(matched), 4)
        self.assertEqual(len(missing), 0)

    def test_score_correctness_partial_coverage(self):
        """Verify proportionate correctness score when some key facts are missing."""
        ans = "Banks must report cyber incidents to CERT-In within 6 hours of detection."
        score, matched, missing = self.evaluator.score_correctness(ans, self.test_case)
        self.assertLess(score, 1.0)
        self.assertIn("6 hours", matched)
        self.assertIn("CERT-In", matched)
        self.assertIn("7 business days", missing)

    def test_score_correctness_unanswerable_refusal(self):
        """Verify unanswerable query achieves 1.0 correctness when refusal is invoked."""
        unanswerable_tc = EvaluationTestCase(
            test_id="TEST-NEG",
            query="What are Basel III rural bank capital buffers?",
            category="Refusal",
            expected_answer="Refuse.",
            expected_sources=[],
            expected_chunk_ids=[],
            key_facts=["insufficient"],
            is_answerable=False,
        )
        refusal_ans = "The provided regulatory context does not contain sufficient information to answer this question reliably."
        score, matched, missing = self.evaluator.score_correctness(refusal_ans, unanswerable_tc)
        self.assertEqual(score, 1.0)
        self.assertEqual(matched, ["safe_refusal_triggered"])

    def test_score_correctness_unanswerable_hallucination_fails(self):
        """Verify unanswerable query achieves 0.0 correctness when system hallucinates an answer."""
        unanswerable_tc = EvaluationTestCase(
            test_id="TEST-NEG",
            query="What are Basel III rural bank capital buffers?",
            category="Refusal",
            expected_answer="Refuse.",
            expected_sources=[],
            expected_chunk_ids=[],
            key_facts=["insufficient"],
            is_answerable=False,
        )
        hallucinated_ans = "Regional rural banks must maintain a CET1 capital adequacy ratio of 9.0% with a 2.5% buffer."
        score, matched, missing = self.evaluator.score_correctness(hallucinated_ans, unanswerable_tc)
        self.assertEqual(score, 0.0)
        self.assertEqual(missing, ["safe_refusal_missing"])

    def test_score_grounding_grounded_answer(self):
        """Verify grounding score is 1.0 when assertions match retrieved context."""
        chunk = RetrievedRecord(
            rank=1,
            id="c1",
            similarity_score=0.8,
            distance=0.2,
            document="Any cyber security incident affecting customer-facing channels must be reported to CSITE and CERT-In within 6 hours.",
            metadata={"source_document": "cyber_resilience.pdf"},
        )
        ans = "Any cyber security incident affecting customer-facing channels must be reported within 6 hours."
        score, supported, unsupported = self.evaluator.score_grounding(ans, [chunk], is_refusal=False)
        self.assertEqual(score, 1.0)
        self.assertGreater(len(supported), 0)
        self.assertEqual(len(unsupported), 0)

    def test_score_grounding_safe_refusal(self):
        """Verify safe refusal is automatically granted 1.0 grounding score."""
        score, supported, unsupported = self.evaluator.score_grounding(
            "The provided regulatory context does not contain sufficient information to answer.",
            [],
            is_refusal=True,
        )
        self.assertEqual(score, 1.0)


class TestCitationAccuracy(unittest.TestCase):
    """Tests for Task 3: Check Citation Accuracy."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.evaluator = RAGEvaluator(openai_client=self.mock_client)

    def test_check_citation_accuracy_valid(self):
        """Verify precision is 1.0 when citations map to chunks containing supporting evidence."""
        chunk1 = RetrievedRecord(
            rank=1,
            id="cyber_001",
            similarity_score=0.75,
            distance=0.25,
            document="Incidents must be reported to CERT-In within 6 hours of detection.",
            metadata={"source_document": "cyber_resilience_framework.pdf"},
        )
        registry = {
            "[1]": {
                "source_document": "cyber_resilience_framework.pdf",
                "chunk_id": "cyber_001",
                "verbatim_text": chunk1.document,
            }
        }
        ans = "Banks must report cyber incidents to CERT-In within 6 hours of detection [1]."

        precision, checks = self.evaluator.check_citation_accuracy(ans, registry, [chunk1])
        self.assertEqual(precision, 1.0)
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].is_valid_marker)
        self.assertFalse(checks[0].is_fabricated)
        self.assertTrue(checks[0].supports_claim)

    def test_check_citation_accuracy_fabricated_marker(self):
        """Verify fabricated marker not in candidate chunks or registry is flagged."""
        chunk1 = RetrievedRecord(
            rank=1,
            id="cyber_001",
            similarity_score=0.75,
            distance=0.25,
            document="Incidents must be reported within 6 hours.",
            metadata={"source_document": "cyber_resilience.pdf"},
        )
        ans = "Banks must report cyber incidents within 6 hours [99]."

        precision, checks = self.evaluator.check_citation_accuracy(ans, {}, [chunk1])
        self.assertEqual(precision, 0.0)
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].is_fabricated)
        self.assertFalse(checks[0].supports_claim)
        self.assertIn("Fabricated citation", checks[0].audit_notes)


class TestFailureDiagnosticsAndAggregation(unittest.TestCase):
    """Tests for Task 4: Summarize Quality and Failures."""

    def setUp(self):
        self.evaluator = RAGEvaluator(openai_client=MagicMock())

    def test_diagnose_failure_hallucination(self):
        """Verify failure diagnostic for hallucinating an unanswerable query."""
        tc = EvaluationTestCase(
            test_id="EVAL-08",
            query="Out of corpus query",
            category="Refusal",
            expected_answer="Refuse.",
            expected_sources=[],
            expected_chunk_ids=[],
            key_facts=["insufficient"],
            is_answerable=False,
        )
        verdict, failure_cat, root_cause, advice = self.evaluator.diagnose_failure(
            test_case=tc,
            correctness=0.0,
            grounding=0.2,
            citation_precision=0.0,
            source_recall=0.0,
            is_refusal=False,
        )
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(failure_cat, "HALLUCINATED_ANSWER")
        self.assertIn("failed to trigger safe refusal", root_cause)

    def test_diagnose_failure_fabricated_citation(self):
        """Verify failure diagnostic for fabricated citations."""
        tc = EvaluationTestCase(
            test_id="EVAL-01",
            query="Query",
            category="Category",
            expected_answer="Ans",
            expected_sources=["doc.pdf"],
            expected_chunk_ids=["c1"],
            key_facts=["fact"],
            is_answerable=True,
        )
        verdict, failure_cat, root_cause, advice = self.evaluator.diagnose_failure(
            test_case=tc,
            correctness=0.9,
            grounding=0.9,
            citation_precision=0.2,
            source_recall=1.0,
            is_refusal=False,
        )
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(failure_cat, "FABRICATED_CITATIONS")

    def test_diagnose_pass(self):
        """Verify PASS verdict when metrics meet high quality thresholds."""
        tc = EvaluationTestCase(
            test_id="EVAL-01",
            query="Query",
            category="Category",
            expected_answer="Ans",
            expected_sources=["doc.pdf"],
            expected_chunk_ids=["c1"],
            key_facts=["fact"],
            is_answerable=True,
        )
        verdict, failure_cat, root_cause, advice = self.evaluator.diagnose_failure(
            test_case=tc,
            correctness=1.0,
            grounding=1.0,
            citation_precision=1.0,
            source_recall=1.0,
            is_refusal=False,
        )
        self.assertEqual(verdict, "PASS")
        self.assertIsNone(failure_cat)


class TestArtifactExport(unittest.TestCase):
    """Tests for Task 5: Export Evaluation Results."""

    def test_export_evaluation_artifacts(self):
        """Verify export_evaluation_artifacts writes valid Markdown and JSON files."""
        result1 = EvaluationResult(
            test_id="EVAL-01",
            query="What are reporting timelines for cyber incidents?",
            category="Cyber Incident Reporting",
            is_answerable=True,
            generated_answer="Incidents must be reported within 6 hours to RBI and CERT-In [1].",
            correctness_score=1.0,
            grounding_score=1.0,
            citation_precision=1.0,
            overall_score=1.0,
            matched_key_facts=["6 hours", "CERT-In"],
            missing_key_facts=[],
            retrieved_chunk_ids=["chunk_001"],
            retrieved_sources=["cyber_resilience_framework.pdf"],
            expected_sources=["cyber_resilience_framework.pdf"],
            source_recall=1.0,
            citation_checks=[
                CitationCheckResult(
                    marker="[1]",
                    cited_chunk_id="chunk_001",
                    source_document="cyber_resilience_framework.pdf",
                    is_valid_marker=True,
                    is_fabricated=False,
                    supports_claim=True,
                    verbatim_evidence_snippet="reported within 6 hours",
                    claim_sentence="Incidents must be reported within 6 hours [1].",
                    audit_notes="Verified.",
                )
            ],
            verdict="PASS",
            latency_seconds=1.2,
            is_refusal=False,
        )

        summary = EvaluationSummary(
            total_tests=1,
            passed_tests=1,
            flagged_tests=0,
            failed_tests=0,
            pass_rate=1.0,
            mean_correctness=1.0,
            mean_grounding=1.0,
            mean_citation_precision=1.0,
            mean_source_recall=1.0,
            mean_overall_score=1.0,
            safe_refusal_accuracy=1.0,
            fabricated_citations_count=0,
            failure_breakdown={},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            md_path, json_path = export_evaluation_artifacts(
                results=[result1],
                summary=summary,
                output_dir=tmp_path,
            )

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            # Verify Markdown content
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("End-to-End RAG System Evaluation & Quality Benchmark Report", md_text)
            self.assertIn("EVAL-01", md_text)
            self.assertIn("Cyber Incident Reporting", md_text)
            self.assertIn("Overall Pass Rate", md_text)

            # Verify JSON schema
            json_text = json_path.read_text(encoding="utf-8")
            data = json.loads(json_text)
            self.assertEqual(data["summary"]["total_tests"], 1)
            self.assertEqual(data["summary"]["pass_rate"], 1.0)
            self.assertEqual(len(data["results"]), 1)
            self.assertEqual(data["results"][0]["test_id"], "EVAL-01")


if __name__ == "__main__":
    unittest.main()
