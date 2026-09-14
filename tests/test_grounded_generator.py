"""Unit tests for RegulSense Grounded Answer Generation, Source Accuracy & Retrieval Comparison.

Validates:
1. Task 1: Generating answers from injected retrieved context.
2. Task 2: Checking source accuracy against retrieved chunks and detecting unsupported claims.
3. Task 3: Missing-context fallback when no supporting context is available.
4. Task 4: Comparing outputs generated with and without retrieval.
5. Task 5: Serialization and report export integrity.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.context_assembler import (
    AugmentedPrompt,
    ContextAssembler,
    InjectedChunk,
    SourceMarker,
    TokenBudgetLedger,
    TokenBudgetSpec,
)
from src.grounded_generator import (
    AccuracyVerificationReport,
    BaselineGenerationOutput,
    GroundedGenerationOutput,
    GroundedGenerator,
    RetrievalComparisonReport,
    export_grounded_generation_artifacts,
    generate_grounded_markdown_report,
)
from src.retriever import RetrievalRunResult, VectorRetriever
from src.vector_db import RetrievedRecord


class TestGroundedGenerator(unittest.TestCase):
    """Test suite for GroundedGenerator, source verification, and comparison."""

    def setUp(self):
        """Sets up mock client and sample chunks."""
        self.mock_client = MagicMock()
        self.mock_retriever = MagicMock(spec=VectorRetriever)
        self.mock_assembler = MagicMock(spec=ContextAssembler)

        self.sample_chunks = [
            RetrievedRecord(
                rank=1,
                id="cyber_resilience_chunk_001",
                similarity_score=0.8850,
                distance=0.1150,
                document="Any Severity 1 cyber security incident must be reported to the RBI CSITE and CERT-In within 6 hours of detection. Initial reports must be followed by a comprehensive forensic analysis report within 7 business days.",
                metadata={
                    "source_document": "cyber_resilience_framework.pdf",
                    "section": "2. Incident Reporting Timelines (6-Hour Rule)",
                    "page_number": 1,
                    "token_count": 35,
                },
            ),
        ]

    def test_generate_grounded_answer_mock(self):
        """Verifies Task 1: Grounded generation properly invokes LLM with assembled messages and parses answer."""
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "Under RBI CSITE guidelines [1], banks must report any Severity 1 cyber security "
            "incident to CERT-In and RBI within 6 hours of detection."
        )
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 320
        mock_usage.completion_tokens = 35
        mock_usage.total_tokens = 355
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        self.mock_client.chat.completions.create.return_value = mock_resp

        generator = GroundedGenerator(
            openai_client=self.mock_client,
            retriever=self.mock_retriever,
            model="llama3:latest",
        )

        output = generator.generate_grounded_answer(
            query="When must Severity 1 cyber incidents be reported?",
            chunks=self.sample_chunks,
        )

        self.assertIsInstance(output, GroundedGenerationOutput)
        self.assertIn("within 6 hours", output.answer)
        self.assertIn("[1]", output.source_citations)
        self.assertFalse(output.is_fallback)
        self.assertEqual(output.prompt_tokens, 320)
        self.assertEqual(output.completion_tokens, 35)
        self.assertTrue(output.accuracy_report.is_grounded)
        self.assertEqual(output.accuracy_report.verdict, "PASS")

    def test_verify_source_accuracy_pass(self):
        """Verifies Task 2: Factual claims completely present in source text achieve 100% fidelity score."""
        generator = GroundedGenerator(openai_client=self.mock_client)
        answer = (
            "According to [1], Severity 1 incidents must be reported to CERT-In and the RBI "
            "CSITE cell within 6 hours. A comprehensive report is due in 7 business days."
        )

        report = generator.verify_source_accuracy(
            answer=answer,
            chunks=self.sample_chunks,
            query="What are the reporting timelines?",
        )

        self.assertTrue(report.is_grounded)
        self.assertEqual(report.verdict, "PASS")
        self.assertEqual(report.fidelity_score, 1.0)
        self.assertEqual(len(report.unsupported_claims), 0)
        self.assertIn("[1]", report.citation_markers_present)
        self.assertTrue(any("6 hours" in c for c in report.verified_claims))
        self.assertTrue(any("7 business days" in c for c in report.verified_claims))

    def test_verify_source_accuracy_detects_unsupported_claim(self):
        """Verifies Task 2: Accurately flags hallucinated timelines or claims absent from context."""
        generator = GroundedGenerator(openai_client=self.mock_client)
        # Answer contains hallucinated "72 hours" and "24 hours" not in chunk text (which only says 6 hours)
        fabricated_answer = (
            "According to [1], banks have 72 hours to notify the supervisor, and must finish "
            "investigation within 24 hours."
        )

        report = generator.verify_source_accuracy(
            answer=fabricated_answer,
            chunks=self.sample_chunks,
            query="What are the reporting timelines?",
        )

        self.assertIn("72 hours", report.unsupported_claims)
        self.assertIn("24 hours", report.unsupported_claims)
        self.assertLess(report.fidelity_score, 0.50)
        self.assertEqual(report.verdict, "FLAGGED")

    def test_missing_context_fallback_invocation(self):
        """Verifies Task 3: Missing context triggers strict fallback refusal without hallucinating."""
        mock_choice = MagicMock()
        mock_choice.message.content = "The provided regulatory context does not contain sufficient information to answer this question."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 280
        mock_usage.completion_tokens = 18
        mock_usage.total_tokens = 298
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        self.mock_client.chat.completions.create.return_value = mock_resp

        generator = GroundedGenerator(openai_client=self.mock_client)
        output = generator.generate_with_missing_context_fallback(
            query="What are the Basel III capital buffer ratios?"
        )

        self.assertTrue(output.is_fallback)
        self.assertIn("not contain sufficient information", output.answer.lower())
        self.assertTrue(output.accuracy_report.is_grounded)
        self.assertEqual(output.accuracy_report.verdict, "PASS")

    def test_compare_with_and_without_retrieval(self):
        """Verifies Task 4: Executes same query with and without retrieval and generates comparison."""
        # 1. Grounded mock response
        mock_grounded_choice = MagicMock()
        mock_grounded_choice.message.content = "Under RBI Master Direction [1], Severity 1 incidents must be reported within 6 hours."
        mock_grounded_resp = MagicMock()
        mock_grounded_resp.choices = [mock_grounded_choice]
        mock_grounded_resp.usage = MagicMock(prompt_tokens=300, completion_tokens=20, total_tokens=320)

        # 2. Baseline ungrounded mock response
        mock_baseline_choice = MagicMock()
        mock_baseline_choice.message.content = "Banks should generally notify cybersecurity authorities promptly, often within 24 to 72 hours under standard IT guidelines."
        mock_baseline_resp = MagicMock()
        mock_baseline_resp.choices = [mock_baseline_choice]
        mock_baseline_resp.usage = MagicMock(prompt_tokens=40, completion_tokens=25, total_tokens=65)

        self.mock_client.chat.completions.create.side_effect = [
            mock_grounded_resp,
            mock_baseline_resp,
        ]

        self.mock_retriever.retrieve.return_value = RetrievalRunResult(
            query_text="Cyber report deadline",
            k=1,
            retrieved_count=1,
            chunks=self.sample_chunks,
            top_score=0.885,
            lowest_score=0.885,
            mean_score=0.885,
            total_tokens_retrieved=35,
            source_documents=["cyber_resilience_framework.pdf"],
        )

        generator = GroundedGenerator(
            openai_client=self.mock_client,
            retriever=self.mock_retriever,
        )

        comparison = generator.compare_with_and_without_retrieval(
            query="What is the mandatory reporting timeframe for Severity 1 cyber incidents?",
            top_k=1,
        )

        self.assertIsInstance(comparison, RetrievalComparisonReport)
        self.assertTrue(comparison.specificity_contrast["grounded_contains_exact_6_hour_rule"])
        self.assertFalse(comparison.specificity_contrast["baseline_contains_exact_6_hour_rule"])
        self.assertIn("HIGH RISK", comparison.hallucination_risk_assessment)
        self.assertIn("source citations", comparison.summary)

    def test_serialization_and_markdown_report(self):
        """Verifies Task 5: JSON serialization and Markdown report export."""
        acc_rep = AccuracyVerificationReport(
            is_grounded=True,
            fidelity_score=1.0,
            citation_markers_present=["[1]"],
            citations_count=1,
            verified_claims=["6 hours"],
            unsupported_claims=[],
            verdict="PASS",
            audit_notes="Audit verified.",
        )
        grounded_out = GroundedGenerationOutput(
            query="Sample query",
            answer="Answer citing [1]",
            injected_chunks=[],
            source_citations=["[1]"],
            accuracy_report=acc_rep,
            is_fallback=False,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            latency_seconds=0.5,
            model="llama3:latest",
        )
        fallback_out = GroundedGenerationOutput(
            query="Negative query",
            answer="The provided regulatory context does not contain sufficient information to answer this question.",
            injected_chunks=[],
            source_citations=[],
            accuracy_report=acc_rep,
            is_fallback=True,
            prompt_tokens=80,
            completion_tokens=15,
            total_tokens=95,
            latency_seconds=0.3,
            model="llama3:latest",
        )
        baseline_out = BaselineGenerationOutput(
            query="Sample query",
            answer="Generic baseline answer",
            prompt_tokens=30,
            completion_tokens=15,
            total_tokens=45,
            latency_seconds=0.2,
            model="llama3:latest",
        )
        comp_rep = RetrievalComparisonReport(
            query="Sample query",
            with_retrieval=grounded_out,
            without_retrieval=baseline_out,
            specificity_contrast={
                "grounded_source_citations": ["[1]"],
                "grounded_cites_circular_authority": True,
                "grounded_contains_exact_6_hour_rule": True,
                "baseline_cites_exact_circular": False,
                "baseline_contains_exact_6_hour_rule": False,
                "grounded_word_count": 5,
                "baseline_word_count": 4,
                "grounded_latency_seconds": 0.5,
                "baseline_latency_seconds": 0.2,
            },
            hallucination_risk_assessment="High risk without retrieval.",
            summary="Retrieval grounded.",
        )

        md = generate_grounded_markdown_report(grounded_out, fallback_out, comp_rep)
        self.assertIn("# RegulSense: Grounded Answer Generation", md)
        self.assertIn("Claim Fidelity Score", md)
        self.assertIn("Missing-Context Fallback Protocol Demonstration", md)
        self.assertIn("Side-by-Side Comparison", md)

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path, json_path = export_grounded_generation_artifacts(
                grounded_output=grounded_out,
                fallback_output=fallback_out,
                comparison_report=comp_rep,
                output_dir=tmp_dir,
            )
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())


if __name__ == "__main__":
    unittest.main()
