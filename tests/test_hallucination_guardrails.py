"""Unit tests for RegulSense Hallucination Guardrails & Retrieval Quality Enforcement.

Validates:
1. Task 1: Detecting weak retrieval (empty results, low similarity score, too few chunks).
2. Task 2: Returning safe refusal when context is weak, preventing hallucinations.
3. Task 3: Enforcing relevance thresholds and retrieval quality diagnostic checks.
4. Task 4: Preserving confident, cited grounded answers when strong context exists.
5. Task 5: Serializing sample refusal and answer reports to disk.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.citation_engine import CitationAuditReport, CitationEngine, CitedAnswerOutput
from src.hallucination_guardrails import (
    GuardrailExecutionResult,
    GuardrailThresholdConfig,
    HallucinationGuardrail,
    RetrievalQualityAssessment,
    export_guardrail_artifacts,
    generate_guardrails_markdown_report,
)
from src.retriever import RetrievalRunResult, VectorRetriever
from src.vector_db import RetrievedRecord


class TestHallucinationGuardrails(unittest.TestCase):
    """Test suite for HallucinationGuardrail retrieval quality checks and refusal protocols."""

    def setUp(self):
        """Sets up test mocks and sample candidate records."""
        self.mock_client = MagicMock()
        self.mock_retriever = MagicMock(spec=VectorRetriever)
        self.mock_citation_engine = MagicMock(spec=CitationEngine)

        # High-relevance candidate chunk (Severity 1 incident reporting)
        self.strong_chunks = [
            RetrievedRecord(
                rank=1,
                id="cyber_001",
                similarity_score=0.7850,
                distance=0.2150,
                document="Severity 1 incidents must be reported to CERT-In and RBI CSITE within 6 hours.",
                metadata={"source_document": "cyber_resilience.pdf", "section": "Sec 2", "page_number": 1},
            ),
            RetrievedRecord(
                rank=2,
                id="cyber_002",
                similarity_score=0.6420,
                distance=0.3580,
                document="Initial reports must be followed by a comprehensive forensic report in 7 days.",
                metadata={"source_document": "cyber_resilience.pdf", "section": "Sec 3", "page_number": 2},
            ),
        ]

        # Low-relevance candidate chunks (Out-of-corpus query)
        self.weak_chunks = [
            RetrievedRecord(
                rank=1,
                id="cdd_001",
                similarity_score=0.4120,
                distance=0.5880,
                document="General provisions for customer due diligence and identity verification.",
                metadata={"source_document": "circular_dor.txt", "section": "Sec 1", "page_number": 1},
            ),
            RetrievedRecord(
                rank=2,
                id="cdd_002",
                similarity_score=0.3850,
                distance=0.6150,
                document="Low-risk customers require standard OVD verification.",
                metadata={"source_document": "circular_dor.txt", "section": "Sec 2", "page_number": 2},
            ),
        ]

    def test_evaluate_retrieval_quality_empty_results(self):
        """Verifies Task 1: Detects empty retrieval results and flags failure."""
        guardrail = HallucinationGuardrail(
            openai_client=self.mock_client,
            config=GuardrailThresholdConfig(min_similarity_threshold=0.50),
        )

        assessment = guardrail.evaluate_retrieval_quality(
            query="Cryptocurrency mining rules in Antarctica",
            chunks=[],
        )

        self.assertFalse(assessment.is_sufficient)
        self.assertEqual(assessment.status, "EMPTY_RESULTS")
        self.assertEqual(assessment.top_score, 0.0)
        self.assertEqual(assessment.qualifying_chunk_count, 0)
        self.assertIn("returned 0 candidate documents", assessment.refusal_reason)

    def test_evaluate_retrieval_quality_low_similarity_score(self):
        """Verifies Task 1 & 3: Flags weak context when top similarity score falls below threshold."""
        guardrail = HallucinationGuardrail(
            openai_client=self.mock_client,
            config=GuardrailThresholdConfig(min_similarity_threshold=0.50, min_top_score_threshold=0.50),
        )

        assessment = guardrail.evaluate_retrieval_quality(
            query="What are the Basel III capital buffer ratios?",
            chunks=self.weak_chunks,  # Top score is 0.4120
        )

        self.assertFalse(assessment.is_sufficient)
        self.assertEqual(assessment.status, "LOW_SIMILARITY_SCORE")
        self.assertAlmostEqual(assessment.top_score, 0.4120)
        self.assertEqual(assessment.qualifying_chunk_count, 0)
        self.assertIn("below the minimum required confidence threshold", assessment.refusal_reason)

    def test_evaluate_retrieval_quality_too_few_qualifying_chunks(self):
        """Verifies Task 1 & 3: Flags context when qualifying chunks count is below minimum required."""
        # 1 strong chunk (0.65), 1 weak chunk (0.40) with min_chunks_above_threshold=2
        mixed_chunks = [
            {"similarity_score": 0.6500, "chunk_id": "c1", "text": "Valid text"},
            {"similarity_score": 0.4000, "chunk_id": "c2", "text": "Weak text"},
        ]
        guardrail = HallucinationGuardrail(
            openai_client=self.mock_client,
            config=GuardrailThresholdConfig(min_similarity_threshold=0.50, min_chunks_above_threshold=2),
        )

        assessment = guardrail.evaluate_retrieval_quality(
            query="Query requiring multiple sources",
            chunks=mixed_chunks,
        )

        self.assertFalse(assessment.is_sufficient)
        self.assertEqual(assessment.status, "TOO_FEW_RELEVANT_CHUNKS")
        self.assertEqual(assessment.qualifying_chunk_count, 1)
        self.assertIn("minimum required: 2", assessment.refusal_reason)

    def test_evaluate_retrieval_quality_sufficient_context(self):
        """Verifies Task 3 & 4: Validates sufficient context when candidate chunks pass threshold."""
        guardrail = HallucinationGuardrail(
            openai_client=self.mock_client,
            config=GuardrailThresholdConfig(min_similarity_threshold=0.50, min_chunks_above_threshold=1),
        )

        assessment = guardrail.evaluate_retrieval_quality(
            query="What is the 6-hour incident reporting rule?",
            chunks=self.strong_chunks,
        )

        self.assertTrue(assessment.is_sufficient)
        self.assertEqual(assessment.status, "SUFFICIENT_CONTEXT")
        self.assertAlmostEqual(assessment.top_score, 0.7850)
        self.assertEqual(assessment.qualifying_chunk_count, 2)
        self.assertIsNone(assessment.refusal_reason)

    def test_build_safe_refusal_formatting(self):
        """Verifies Task 2: Refusal message contains clear diagnostic details without hallucinated facts."""
        guardrail = HallucinationGuardrail(openai_client=self.mock_client)
        assessment = RetrievalQualityAssessment(
            is_sufficient=False,
            status="LOW_SIMILARITY_SCORE",
            top_score=0.4215,
            mean_qualifying_score=0.0,
            qualifying_chunk_count=0,
            total_retrieved_count=3,
            threshold_used=0.50,
            min_chunks_required=1,
            refusal_reason="Similarity score 0.4215 below threshold 0.5000.",
        )

        refusal = guardrail.build_safe_refusal(query="Inquiry", assessment=assessment)
        self.assertIn("does not contain sufficient information", refusal)
        self.assertIn("LOW_SIMILARITY_SCORE", refusal)
        self.assertIn("0.4215", refusal)
        self.assertIn("0.5000", refusal)

    def test_execute_safe_refusal_bypasses_llm(self):
        """Verifies Task 2: When retrieval is weak, system returns safe refusal without calling model generation."""
        guardrail = HallucinationGuardrail(
            openai_client=self.mock_client,
            retriever=self.mock_retriever,
            citation_engine=self.mock_citation_engine,
            config=GuardrailThresholdConfig(min_similarity_threshold=0.50),
        )

        result = guardrail.execute(
            query="Out of corpus query",
            chunks=self.weak_chunks,  # Top score 0.4120 < 0.50
        )

        self.assertEqual(result.action, "REFUSE")
        self.assertTrue(result.is_refusal)
        self.assertEqual(len(result.citations), 0)
        self.assertIsNone(result.cited_output)
        self.assertIn("LOW_SIMILARITY_SCORE", result.answer)
        # Citation engine must NOT be invoked when context is weak!
        self.mock_citation_engine.generate_cited_answer.assert_not_called()

    def test_execute_confident_grounded_answer(self):
        """Verifies Task 4: Confident grounded answers are generated when strong context exists."""
        mock_audit = CitationAuditReport(
            total_citations_found=2,
            unique_markers_cited=["[1]", "[2]"],
            verified_citations_count=2,
            fabricated_citations_count=0,
            unsupported_claims_count=0,
            citation_precision=1.0,
            claim_verifications=[],
            fabricated_markers=[],
            overall_verdict="PASS",
            audit_notes="Verified.",
        )
        mock_cited_output = CitedAnswerOutput(
            query="Severity 1 reporting timeline?",
            answer="Under [1], banks must report within 6 hours. Under [2], forensic analysis is due in 7 days.",
            citation_registry={"[1]": {}, "[2]": {}},
            audit_report=mock_audit,
            is_fallback=False,
            has_fabricated_citations=False,
            prompt_tokens=450,
            completion_tokens=45,
            latency_seconds=0.4,
            model="llama3:latest",
        )
        self.mock_citation_engine.generate_cited_answer.return_value = mock_cited_output

        guardrail = HallucinationGuardrail(
            openai_client=self.mock_client,
            citation_engine=self.mock_citation_engine,
            config=GuardrailThresholdConfig(min_similarity_threshold=0.50),
        )

        result = guardrail.execute(
            query="Severity 1 reporting timeline?",
            chunks=self.strong_chunks,
        )

        self.assertEqual(result.action, "ANSWER")
        self.assertFalse(result.is_refusal)
        self.assertIn("[1]", result.answer)
        self.assertIn("[1]", result.citations)
        self.assertIn("[2]", result.citations)
        self.assertEqual(result.quality_assessment.status, "SUFFICIENT_CONTEXT")
        self.mock_citation_engine.generate_cited_answer.assert_called_once()

    def test_custom_threshold_configuration(self):
        """Verifies Task 3: Sensitivity can be adjusted via custom thresholds."""
        strict_config = GuardrailThresholdConfig(min_similarity_threshold=0.85)
        guardrail = HallucinationGuardrail(openai_client=self.mock_client, config=strict_config)

        # Strong chunk score 0.7850 passes standard 0.50 threshold but fails strict 0.85 threshold
        assessment = guardrail.evaluate_retrieval_quality("Query", self.strong_chunks)
        self.assertFalse(assessment.is_sufficient)
        self.assertEqual(assessment.status, "LOW_SIMILARITY_SCORE")

    def test_serialization_and_artifact_export(self):
        """Verifies Task 5: Markdown and JSON artifacts are created with comparative scenarios."""
        assessment_pass = RetrievalQualityAssessment(
            is_sufficient=True,
            status="SUFFICIENT_CONTEXT",
            top_score=0.82,
            mean_qualifying_score=0.82,
            qualifying_chunk_count=1,
            total_retrieved_count=1,
            threshold_used=0.50,
            min_chunks_required=1,
        )
        confident_case = GuardrailExecutionResult(
            query="Confident query",
            action="ANSWER",
            answer="Grounded answer citing [1].",
            quality_assessment=assessment_pass,
            is_refusal=False,
            citations=["[1]"],
            latency_seconds=0.5,
            model="llama3:latest",
        )

        assessment_weak = RetrievalQualityAssessment(
            is_sufficient=False,
            status="LOW_SIMILARITY_SCORE",
            top_score=0.38,
            mean_qualifying_score=0.0,
            qualifying_chunk_count=0,
            total_retrieved_count=2,
            threshold_used=0.50,
            min_chunks_required=1,
            refusal_reason="Low score",
        )
        weak_case = GuardrailExecutionResult(
            query="Weak query",
            action="REFUSE",
            answer="Refusal due to weak score.",
            quality_assessment=assessment_weak,
            is_refusal=True,
            citations=[],
            latency_seconds=0.1,
            model="llama3:latest",
        )

        assessment_empty = RetrievalQualityAssessment(
            is_sufficient=False,
            status="EMPTY_RESULTS",
            top_score=0.0,
            mean_qualifying_score=0.0,
            qualifying_chunk_count=0,
            total_retrieved_count=0,
            threshold_used=0.50,
            min_chunks_required=1,
            refusal_reason="Empty",
        )
        empty_case = GuardrailExecutionResult(
            query="Empty query",
            action="REFUSE",
            answer="Refusal due to empty chunks.",
            quality_assessment=assessment_empty,
            is_refusal=True,
            citations=[],
            latency_seconds=0.05,
            model="llama3:latest",
        )

        md = generate_guardrails_markdown_report(confident_case, weak_case, empty_case)
        self.assertIn("RegulSense: Hallucination Guardrails", md)
        self.assertIn("Confident Grounded Answer", md)
        self.assertIn("Safe Refusal Case: Low Similarity Score", md)
        self.assertIn("Safe Refusal Case: Empty Retrieval", md)
        self.assertIn("Retrieval Quality Diagnostic Comparison", md)

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path, json_path = export_guardrail_artifacts(
                confident_case=confident_case,
                weak_score_refusal_case=weak_case,
                empty_refusal_case=empty_case,
                output_dir=tmp_dir,
            )
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["confident_case"]["action"], "ANSWER")
            self.assertEqual(data["weak_score_refusal_case"]["action"], "REFUSE")
            self.assertEqual(data["empty_refusal_case"]["action"], "REFUSE")


if __name__ == "__main__":
    unittest.main()
