"""Unit tests for RegulSense Two-Stage Retrieval and Chunk Re-Ranking Module.

Validates:
1. Task 1: Retrieving an expanded Stage 1 candidate pool (N > k).
2. Task 2: Re-ranking candidates using fine-grained cross-scoring and LLM scoring.
3. Task 3: Demonstrating improved top results where operative clauses are promoted to Rank #1.
4. Task 4: Before-and-after comparison ledgers displaying initial vector vs. re-ranked ranks and scores.
5. Task 5: Exporting comprehensive Markdown and JSON re-ranking audit reports.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.reranker import (
    CrossScorer,
    LLMScorer,
    RerankDemonstrationReport,
    RerankResult,
    ScoredCandidate,
    TwoStageRetriever,
    demonstrate_reranking,
    export_reranking_artifacts,
)
from src.retriever import VectorRetriever
from src.vector_db import (
    DEFAULT_VECTOR_DIMENSION,
    RetrievedRecord,
    VectorDatabaseManager,
    VectorRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestChunkReranker(unittest.TestCase):
    """Test suite for two-stage candidate retrieval and fine-grained chunk re-ranking."""

    def setUp(self):
        """Sets up an in-memory ChromaDB instance with synthetic regulatory chunks."""
        self.vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name="test_reranking_collection",
            vector_dimension=DEFAULT_VECTOR_DIMENSION,
            distance_metric="cosine",
        )

        # Create synthetic corpus chunks:
        # Chunk 1: Preamble / General Advisory header (broad keywords, but no specific hours)
        # Chunk 2: Operative Section 3: Recovery Agent Conduct (explicit 8:00 AM - 7:00 PM, harassment prohibition)
        # Chunk 3: Cyber Security Incident Notification (Severity 1 within 6 hours)
        # Chunk 4: Cyber Security SOC 24x7 continuous monitoring
        # Chunk 5: AML KYC V-CIP 95% facial match
        self.corpus_chunks = [
            VectorRecord(
                id="digital_lending_compliance_note_html_tokenaware_001",
                embedding=self._make_unit_vector(0.3, 0.8, 0.0),
                document=(
                    "Regulatory Advisory: Digital Lending Guidelines & Consumer Protection. "
                    "RegulSense Regulatory Advisory: Digital Lending & Fair Practices Code. "
                    "Reference: RBI/2022-23/DOR.CRE.REC.42/21.04.048/2022-23. "
                    "Target Entities: Commercial Banks, Digital Lending Platforms, Lending Service Providers. "
                    "All loan disbursals and repayments must be executed solely between the bank account of the borrower."
                ),
                metadata={
                    "source_document": "digital_lending_compliance_note.html",
                    "filename": "digital_lending_compliance_note.html",
                    "file_type": ".html",
                    "section": "Preamble / Document Header",
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": 65,
                },
            ),
            VectorRecord(
                id="digital_lending_compliance_note_html_tokenaware_002",
                embedding=self._make_unit_vector(0.25, 0.75, 0.05),
                document=(
                    "3. Code of Conduct for Recovery Agents: Regulated entities and their engaged recovery agents "
                    "are strictly prohibited from contacting borrowers before 8:00 AM or after 7:00 PM. "
                    "Harassment, verbal intimidation, or unauthorized contact with borrowers' relatives are strictly prohibited. "
                    "Violations will lead to immediate regulatory sanction and license revocation."
                ),
                metadata={
                    "source_document": "digital_lending_compliance_note.html",
                    "filename": "digital_lending_compliance_note.html",
                    "file_type": ".html",
                    "section": "3. Code of Conduct for Recovery Agents",
                    "page_number": 1,
                    "chunk_index": 1,
                    "token_count": 72,
                },
            ),
            VectorRecord(
                id="cyber_resilience_framework_pdf_tokenaware_001",
                embedding=self._make_unit_vector(0.05, 0.1, 0.85),
                document=(
                    "Cyber Security Framework in Banks: Incident reporting baseline architecture. "
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
                id="cyber_resilience_framework_pdf_tokenaware_002",
                embedding=self._make_unit_vector(0.05, 0.1, 0.90),
                document=(
                    "24x7x365 Security Operations Centre (SOC) equipped with continuous automated threat monitoring, "
                    "SIEM log telemetry, and quarterly penetration testing for bank infrastructure resilience."
                ),
                metadata={
                    "source_document": "cyber_resilience_framework.pdf",
                    "filename": "cyber_resilience_framework.pdf",
                    "file_type": ".pdf",
                    "section": "Annex 3: SOC Governance and Telemetry",
                    "page_number": 2,
                    "chunk_index": 1,
                    "token_count": 60,
                },
            ),
            VectorRecord(
                id="circular_dor_2024_108_txt_tokenaware_002",
                embedding=self._make_unit_vector(0.9, 0.1, 0.0),
                document=(
                    "Customer Due Diligence (CDD) Requirements and V-CIP Verification. "
                    "The automated facial match system shall achieve a confidence score of not less than 95% "
                    "with liveliness detection prior to account onboarding."
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
        ]
        self.vdb.insert_records(self.corpus_chunks)

        # Mock OpenAI Client for vector retriever
        self.mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_data = MagicMock()
        # Query vector close to chunk 001 (Preamble) so initial vector search puts chunk 001 at Rank #1
        mock_data.embedding = self._make_unit_vector(0.3, 0.8, 0.0)
        mock_resp.data = [mock_data]
        self.mock_client.embeddings.create.return_value = mock_resp

        self.retriever = VectorRetriever(
            vector_db=self.vdb,
            collection_name="test_reranking_collection",
            openai_client=self.mock_client,
        )
        self.two_stage = TwoStageRetriever(
            vector_retriever=self.retriever,
            use_llm_reranker=False,
        )

    def _make_unit_vector(self, x: float, y: float, z: float) -> list:
        vec = [0.0] * DEFAULT_VECTOR_DIMENSION
        vec[0] = x
        vec[1] = y
        vec[2] = z
        norm = np.linalg.norm(vec)
        return [float(v / norm) for v in vec]

    # -------------------------------------------------------------------------
    # Task 1: Retrieve a Larger Candidate Set
    # -------------------------------------------------------------------------

    def test_task1_retrieve_larger_candidate_set(self):
        """Verify Stage 1 retrieves an expanded candidate pool (N=5, larger than target k=2)."""
        candidates = self.two_stage.retrieve_candidates(
            query_text="What are recovery agent permitted hours and harassment rules?",
            candidate_count=5,
        )
        self.assertEqual(len(candidates), 5)
        self.assertIsInstance(candidates[0], RetrievedRecord)
        # Confirm initial ranking order is strictly descending by vector score
        scores = [c.similarity_score for c in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    # -------------------------------------------------------------------------
    # Task 2: Re-Rank Candidates
    # -------------------------------------------------------------------------

    def test_task2_cross_scorer_mathematical_bounds_and_entities(self):
        """Verify CrossScorer bounds scores in [0.0, 1.0] and extracts statutory entities."""
        query = "What are the permitted hours for recovery agents contacting borrowers?"
        doc = (
            "Recovery agents shall contact borrowers only between 8:00 AM and 7:00 PM. "
            "Harassment is strictly prohibited."
        )
        score, breakdown, note = CrossScorer.score_candidate(
            query=query,
            document_text=doc,
            section="3. Code of Conduct for Recovery Agents",
            initial_vector_score=0.75,
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIn("sentence_relevance", breakdown)
        self.assertIn("statutory_entity_match", breakdown)
        self.assertIn("answer_directness", breakdown)
        self.assertIn("initial_vector_score", breakdown)
        self.assertGreater(breakdown["answer_directness"], 0.5)

        # Check entity extraction
        entities = CrossScorer.extract_statutory_entities("within 6 hours of Severity 1 incident, 95% match")
        self.assertTrue(any("6 hours" in e for e in entities))
        self.assertTrue(any("95%" in e for e in entities))
        self.assertTrue(any("severity 1" in e for e in entities))

    def test_task2_llm_scorer_fallback_mode(self):
        """Verify LLMScorer gracefully falls back to CrossScorer if API call fails."""
        mock_failing_client = MagicMock()
        mock_failing_client.chat.completions.create.side_effect = RuntimeError("API unavailable")

        scorer = LLMScorer(openai_client=mock_failing_client)
        score, breakdown, note = scorer.score_candidate(
            query="Recovery hours",
            document_text="8:00 AM to 7:00 PM required hours.",
            section="Recovery Rules",
            initial_vector_score=0.5,
        )
        self.assertGreater(score, 0.0)
        self.assertIn("sentence_relevance", breakdown)

    # -------------------------------------------------------------------------
    # Task 3: Show Improved Top Results (Rank Promotion)
    # -------------------------------------------------------------------------

    def test_task3_re_ranking_promotes_operative_clause(self):
        """Verify re-ranking elevates specific operative clause over preamble chunk to Rank #1."""
        query = "What are the permitted hours for recovery agents contacting borrowers and what penalties apply for harassment?"
        # Stage 1 vector search puts chunk 001 (Preamble) at Rank 1 due to mock vector
        candidates = self.two_stage.retrieve_candidates(query_text=query, candidate_count=5)
        self.assertEqual(candidates[0].id, "digital_lending_compliance_note_html_tokenaware_001")

        # Stage 2 re-ranking scores candidates
        rerank_result = self.two_stage.rerank_candidates(
            query_text=query,
            candidates=candidates,
            final_k=2,
        )

        self.assertIsInstance(rerank_result, RerankResult)
        self.assertEqual(len(rerank_result.final_top_k), 2)
        self.assertTrue(rerank_result.top_result_improved)

        # Chunk 002 (with 8:00 AM - 7:00 PM and harassment sanctions) should now be Rank #1
        top_chunk = rerank_result.final_top_k[0]
        self.assertEqual(top_chunk.chunk_id, "digital_lending_compliance_note_html_tokenaware_002")
        self.assertEqual(top_chunk.final_rank, 1)
        self.assertGreater(top_chunk.rank_delta, 0)
        self.assertEqual(top_chunk.status, "PROMOTED")

        # Preamble chunk 001 should be demoted
        chunk_001 = next(c for c in rerank_result.all_candidates if c.chunk_id == "digital_lending_compliance_note_html_tokenaware_001")
        self.assertLess(chunk_001.rank_delta, 0)
        self.assertEqual(chunk_001.status, "DEMOTED")

    # -------------------------------------------------------------------------
    # Task 4: Compare Before and After
    # -------------------------------------------------------------------------

    def test_task4_before_and_after_comparison_structure(self):
        """Verify before-and-after comparison contains initial ranks, final ranks, deltas, and scores."""
        query = "Severity 1 cyber security incident notification deadline"
        res = self.two_stage.retrieve_and_rerank(query_text=query, candidate_count=4, final_k=2)

        self.assertEqual(res.candidate_count, 4)
        self.assertEqual(res.final_k, 2)
        for c in res.all_candidates:
            self.assertIsInstance(c, ScoredCandidate)
            self.assertGreaterEqual(c.initial_rank, 1)
            self.assertGreaterEqual(c.final_rank, 1)
            self.assertEqual(c.rank_delta, c.initial_rank - c.final_rank)
            self.assertIn(c.status, {"PROMOTED", "DEMOTED", "STABLE"})
            self.assertGreaterEqual(c.rerank_score, 0.0)
            self.assertLessEqual(c.rerank_score, 1.0)
            self.assertGreater(len(c.document), 10)

        d = res.to_dict()
        self.assertIn("all_candidates", d)
        self.assertIn("final_top_k", d)
        self.assertIn("top_result_improved", d)
        self.assertIn("improvement_summary", d)

    # -------------------------------------------------------------------------
    # Task 5: Commit with Sample Output (Artifact Verification)
    # -------------------------------------------------------------------------

    def test_task5_export_reranking_artifacts(self):
        """Verify export_reranking_artifacts creates valid Markdown and JSON files with ledgers."""
        demo_report = demonstrate_reranking(retriever=self.two_stage)
        self.assertIn("case1_pep_approval_rank", demo_report.cases)
        self.assertIn("case2_recovery_agent_conduct", demo_report.cases)
        self.assertIn("case3_cyber_reporting_timeline", demo_report.cases)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_md = Path(tmp_dir) / "test_reranking.md"
            tmp_json = Path(tmp_dir) / "test_reranking.json"

            out_md, out_json = export_reranking_artifacts(
                demo_report=demo_report,
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
            self.assertIn("cases", data)
            self.assertEqual(data["candidate_count_n"], 10)
            self.assertEqual(data["final_k"], 3)

            # Validate Markdown
            md_content = out_md.read_text(encoding="utf-8")
            self.assertIn("# RegulSense: Two-Stage Retrieval & Chunk Re-Ranking Audit Report", md_content)
            self.assertIn("Before-and-After Comparison Ledger", md_content)
            self.assertIn("Final Top-3 Grounding Context Selected for LLM", md_content)
            self.assertIn("Production Architecture Guidelines", md_content)


if __name__ == "__main__":
    unittest.main()
