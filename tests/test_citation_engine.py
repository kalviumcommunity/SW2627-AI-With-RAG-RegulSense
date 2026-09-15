"""Unit tests for RegulSense Verifiable Source Citation Mapping & Anti-Fabrication Engine.

Validates:
1. Task 1: Attaching in-text source references ([1], [2]).
2. Task 2: Mapping citations back to real documents, chunk IDs, sections, and pages.
3. Task 3: Verifying cited sources against original retrieved text.
4. Task 4: Detecting fabricated citations and returning clean fallback when context is missing.
5. Task 5: Serialization and report generation integrity.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.citation_engine import (
    CitationAuditReport,
    CitationEngine,
    CitationMetadataRecord,
    CitedAnswerOutput,
    ClaimVerificationDetail,
    export_citation_artifacts,
    generate_citation_markdown_report,
)
from src.retriever import RetrievalRunResult, VectorRetriever
from src.vector_db import RetrievedRecord


class TestCitationEngine(unittest.TestCase):
    """Test suite for CitationEngine, citation mapping, and anti-fabrication auditing."""

    def setUp(self):
        """Sets up mock records and engine instance."""
        self.mock_client = MagicMock()
        self.mock_retriever = MagicMock(spec=VectorRetriever)

        self.mock_chunks = [
            RetrievedRecord(
                rank=1,
                id="cyber_resilience_chunk_001",
                similarity_score=0.8850,
                distance=0.1150,
                document="Any Severity 1 cyber security incident must be reported to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection. Initial reports must be followed by a comprehensive forensic analysis report within 7 business days.",
                metadata={
                    "source_document": "cyber_resilience_framework.pdf",
                    "section": "2. Incident Reporting Timelines (6-Hour Rule)",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 35,
                },
            ),
            RetrievedRecord(
                rank=2,
                id="circular_dor_chunk_004",
                similarity_score=0.7420,
                distance=0.2580,
                document="Transaction records and account dossiers shall be preserved for a minimum period of five years from the date of cessation of the transaction.",
                metadata={
                    "source_document": "circular_dor_2024_108.txt",
                    "section": "5. Record Retention Obligations",
                    "page_number": 3,
                    "chunk_index": 3,
                    "token_count": 25,
                },
            ),
        ]

    def test_build_citation_registry(self):
        """Verifies Task 2: Registry maps citation markers ([1], [2]) directly to physical metadata."""
        engine = CitationEngine(openai_client=self.mock_client)
        registry = engine.build_citation_registry(self.mock_chunks)

        self.assertEqual(len(registry), 2)
        self.assertIn("[1]", registry)
        self.assertIn("[2]", registry)

        rec1 = registry["[1]"]
        self.assertIsInstance(rec1, CitationMetadataRecord)
        self.assertEqual(rec1.marker_id, "[1]")
        self.assertEqual(rec1.source_document, "cyber_resilience_framework.pdf")
        self.assertEqual(rec1.chunk_id, "cyber_resilience_chunk_001")
        self.assertEqual(rec1.chunk_index, 0)
        self.assertEqual(rec1.section, "2. Incident Reporting Timelines (6-Hour Rule)")
        self.assertEqual(rec1.page_number, 1)
        self.assertAlmostEqual(rec1.similarity_score, 0.8850)
        self.assertIn("within 6 hours", rec1.verbatim_text)

        rec2 = registry["[2]"]
        self.assertEqual(rec2.source_document, "circular_dor_2024_108.txt")
        self.assertEqual(rec2.chunk_index, 3)

    def test_extract_claims_and_citations(self):
        """Verifies Task 1: Extracts claim sentences paired with their cited markers."""
        engine = CitationEngine(openai_client=self.mock_client)
        sample_answer = (
            "Severity 1 cyber incidents must be reported to CERT-In within 6 hours [1]. "
            "Forensic reports are required within 7 business days [1]. "
            "Customer account records must be preserved for five years [2]."
        )

        pairs = engine.extract_claims_and_citations(sample_answer)

        self.assertEqual(len(pairs), 3)
        self.assertIn("[1]", pairs[0][1])
        self.assertIn("within 6 hours", pairs[0][0])
        self.assertIn("[1]", pairs[1][1])
        self.assertIn("7 business days", pairs[1][0])
        self.assertIn("[2]", pairs[2][1])
        self.assertIn("five years", pairs[2][0])

    def test_extract_claims_with_section_submarkers(self):
        """Verifies Task 1: Handles citations with subsections like [1, Section 2] and trailing periods."""
        engine = CitationEngine(openai_client=self.mock_client)
        sample_answer = (
            "According to the Master Direction [1], cyber incidents must be reported within 6 hours [1, Section 2].\n"
            "Initial reports must be followed by forensic analysis within 7 business days [1, Section 2]."
        )

        pairs = engine.extract_claims_and_citations(sample_answer)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][1], ["[1]"])
        self.assertEqual(pairs[1][1], ["[1]"])

    def test_extract_claims_with_source_filename(self):
        """Verifies Task 1: Handles direct source filename citations like [cyber_resilience_framework.pdf]."""
        engine = CitationEngine(openai_client=self.mock_client)
        registry = engine.build_citation_registry(self.mock_chunks)
        sample_answer = (
            "Under [cyber_resilience_framework.pdf], banks must report Severity 1 incidents within 6 hours. "
            "Customer records are kept for five years [circular_dor_2024_108.txt]."
        )

        pairs = engine.extract_claims_and_citations(sample_answer)
        self.assertEqual(len(pairs), 2)
        self.assertIn("[cyber_resilience_framework.pdf]", pairs[0][1])
        self.assertIn("[circular_dor_2024_108.txt]", pairs[1][1])

        # Verify filename markers are resolved and audited without fabrication error
        audit = engine.verify_citations(answer=sample_answer, registry=registry)
        self.assertEqual(audit.overall_verdict, "PASS")
        self.assertEqual(audit.fabricated_citations_count, 0)
        self.assertEqual(len(audit.claim_verifications), 2)
        self.assertTrue(audit.claim_verifications[0].is_verified)

    def test_verify_claim_against_chunk_modular(self):
        """Verifies Task 3: Direct call to verify_claim_against_chunk extracts verbatim evidence spans."""
        engine = CitationEngine(openai_client=self.mock_client)
        rec = CitationMetadataRecord(
            marker_id="[1]",
            index=1,
            source_document="cyber_resilience_framework.pdf",
            chunk_id="cyber_001",
            chunk_index=0,
            section="2. Incident Reporting Timelines (6-Hour Rule)",
            page_number=1,
            similarity_score=0.88,
            verbatim_text="Any Severity 1 cyber security incident must be reported to the RBI within 6 hours of detection.",
        )

        claim = "Banks must report any Severity 1 incident within 6 hours [1]."
        detail = engine.verify_claim_against_chunk(claim, "[1]", rec)

        self.assertIsInstance(detail, ClaimVerificationDetail)
        self.assertTrue(detail.is_verified)
        self.assertIsNotNone(detail.matched_verbatim_span)
        self.assertTrue(
            "Severity 1" in detail.matched_verbatim_span or "6 hours" in detail.matched_verbatim_span
        )
        self.assertGreater(detail.confidence_score, 0.5)

    def test_verify_citations_pass(self):
        """Verifies Task 3: Claim sentences matched against chunk verbatim text return VERIFIED status."""
        engine = CitationEngine(openai_client=self.mock_client)
        registry = engine.build_citation_registry(self.mock_chunks)

        answer = (
            "Under [1], banks must report Severity 1 incidents within 6 hours. "
            "Under [2], transaction records must be retained for five years."
        )

        audit = engine.verify_citations(answer=answer, registry=registry)

        self.assertIsInstance(audit, CitationAuditReport)
        self.assertEqual(audit.overall_verdict, "PASS")
        self.assertEqual(audit.citation_precision, 1.0)
        self.assertEqual(audit.fabricated_citations_count, 0)
        self.assertEqual(len(audit.claim_verifications), 2)

        # Check claim 1 details
        c1 = audit.claim_verifications[0]
        self.assertEqual(c1.cited_marker, "[1]")
        self.assertEqual(c1.source_document, "cyber_resilience_framework.pdf")
        self.assertTrue(c1.is_verified)
        self.assertEqual(c1.status, "VERIFIED")
        self.assertIsNotNone(c1.matched_verbatim_span)

        # Check claim 2 details
        c2 = audit.claim_verifications[1]
        self.assertEqual(c2.cited_marker, "[2]")
        self.assertEqual(c2.source_document, "circular_dor_2024_108.txt")
        self.assertTrue(c2.is_verified)
        self.assertEqual(c2.status, "VERIFIED")

    def test_detect_fabricated_markers(self):
        """Verifies Task 4: Detects hallucinated citation markers not present in the context registry."""
        engine = CitationEngine(openai_client=self.mock_client)
        registry = engine.build_citation_registry(self.mock_chunks)  # Contains [1] and [2]

        # Answer hallucinates marker [99] and [4]
        answer_with_hallucinated_citations = (
            "Banks must report incidents within 6 hours [1]. "
            "Capital buffer ratios must be 12.5% under Basel III [99]. "
            "Additional liquid assets are required [4]."
        )

        audit = engine.verify_citations(answer=answer_with_hallucinated_citations, registry=registry)

        self.assertIn("[99]", audit.fabricated_markers)
        self.assertIn("[4]", audit.fabricated_markers)
        self.assertEqual(audit.fabricated_citations_count, 2)
        self.assertEqual(audit.overall_verdict, "REJECTED")
        self.assertLess(audit.citation_precision, 1.0)

    def test_no_source_fallback_zero_fabricated_citations(self):
        """Verifies Task 4: Missing context returns refusal without fabricating any citations."""
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "The provided regulatory context does not contain sufficient information to answer this question."
        )
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=250, completion_tokens=15)
        self.mock_client.chat.completions.create.return_value = mock_resp

        engine = CitationEngine(openai_client=self.mock_client)

        output = engine.generate_cited_answer(
            query="What are the Basel III capital requirements?",
            chunks=[],  # No supporting chunks
        )

        self.assertTrue(output.is_fallback)
        self.assertFalse(output.has_fabricated_citations)
        self.assertEqual(output.audit_report.fabricated_citations_count, 0)
        self.assertEqual(output.audit_report.overall_verdict, "PASS")
        self.assertEqual(len(output.citation_registry), 0)

    def test_no_source_fallback_with_out_of_corpus_refusal(self):
        """Verifies Task 4: Candidate chunks present but model issues uncited refusal fallback."""
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "The provided regulatory context does not contain sufficient information to answer this question. "
            "It only covers cyber incident reporting and record retention."
        )
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=400, completion_tokens=25)
        self.mock_client.chat.completions.create.return_value = mock_resp

        engine = CitationEngine(openai_client=self.mock_client)

        output = engine.generate_cited_answer(
            query="What are the Basel III capital requirements?",
            chunks=self.mock_chunks,  # Candidate chunks provided, but irrelevant
        )

        self.assertTrue(output.is_fallback)
        self.assertFalse(output.has_fabricated_citations)
        self.assertEqual(output.audit_report.fabricated_citations_count, 0)
        self.assertEqual(output.audit_report.overall_verdict, "PASS")

    def test_generate_cited_answer_end_to_end_mock(self):
        """Verifies full cited answer generation, metadata mapping, and auditing."""
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "According to [1], banks must notify CERT-In within 6 hours of detecting a Severity 1 incident. "
            "Furthermore, account records must be preserved for five years [2]."
        )
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=350, completion_tokens=35)
        self.mock_client.chat.completions.create.return_value = mock_resp

        engine = CitationEngine(openai_client=self.mock_client)

        output = engine.generate_cited_answer(
            query="What are the reporting and retention rules?",
            chunks=self.mock_chunks,
        )

        self.assertIsInstance(output, CitedAnswerOutput)
        self.assertFalse(output.is_fallback)
        self.assertFalse(output.has_fabricated_citations)
        self.assertIn("[1]", output.citation_registry)
        self.assertIn("[2]", output.citation_registry)
        self.assertEqual(output.audit_report.overall_verdict, "PASS")
        self.assertEqual(output.audit_report.citation_precision, 1.0)

    def test_serialization_and_markdown_export(self):
        """Verifies Task 5: JSON serialization and Markdown artifact generation."""
        rec = CitationMetadataRecord(
            marker_id="[1]",
            index=1,
            source_document="doc1.pdf",
            chunk_id="c1",
            chunk_index=0,
            section="Sec 1",
            page_number=1,
            similarity_score=0.9,
            verbatim_text="Incident reported within 6 hours.",
        )
        detail = ClaimVerificationDetail(
            claim_sentence="Report within 6 hours [1].",
            cited_marker="[1]",
            source_document="doc1.pdf",
            section="Sec 1",
            chunk_id="c1",
            is_verified=True,
            status="VERIFIED",
            matched_verbatim_span="within 6 hours",
            matched_keywords=["6", "hours"],
            confidence_score=0.95,
        )
        audit = CitationAuditReport(
            total_citations_found=1,
            unique_markers_cited=["[1]"],
            verified_citations_count=1,
            fabricated_citations_count=0,
            unsupported_claims_count=0,
            citation_precision=1.0,
            claim_verifications=[detail],
            fabricated_markers=[],
            overall_verdict="PASS",
            audit_notes="Verified.",
        )
        cited_out = CitedAnswerOutput(
            query="Sample query",
            answer="Answer citing [1]",
            citation_registry={"[1]": rec.to_dict()},
            audit_report=audit,
            is_fallback=False,
            has_fabricated_citations=False,
            prompt_tokens=100,
            completion_tokens=20,
            latency_seconds=0.4,
            model="llama3:latest",
        )
        fallback_out = CitedAnswerOutput(
            query="Negative query",
            answer="The provided regulatory context does not contain sufficient information to answer this question.",
            citation_registry={},
            audit_report=audit,
            is_fallback=True,
            has_fabricated_citations=False,
            prompt_tokens=80,
            completion_tokens=15,
            latency_seconds=0.2,
            model="llama3:latest",
        )

        md = generate_citation_markdown_report(cited_out, fallback_out)
        self.assertIn("# RegulSense: Verifiable Source Citation Mapping", md)
        self.assertIn("Citation-to-Metadata Registry Mapping", md)
        self.assertIn("Claim-by-Claim Evidentiary Verification Ledger", md)
        self.assertIn("`[1]`", md)

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path, json_path = export_citation_artifacts(
                cited_output=cited_out,
                fallback_output=fallback_out,
                output_dir=tmp_dir,
            )
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())


if __name__ == "__main__":
    unittest.main()
