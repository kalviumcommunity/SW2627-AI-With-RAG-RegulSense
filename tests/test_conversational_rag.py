"""Unit and Integration Tests for Conversational RAG Engine.

Tests cover:
- Task 1: Tracking conversation history across multiple turns.
- Task 2: Rewriting follow-up questions into standalone queries.
- Task 3: Retrieving using rewritten queries with score uplift analysis.
- Task 4: Demonstrating multi-turn dialogue with contextual grounding.
- Task 5: Serializing and exporting conversational dialogue artifacts.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.citation_engine import CitationAuditReport, CitedAnswerOutput
from src.conversational_rag import (
    ConversationHistory,
    ConversationTurn,
    ConversationalRAG,
    QueryRewriter,
    QueryRewritingResult,
    export_conversational_artifacts,
)
from src.retriever import RetrievalRunResult
from src.vector_db import RetrievedRecord


class TestConversationHistory(unittest.TestCase):
    """Tests for Task 1: Track Conversation History."""

    def setUp(self):
        self.history = ConversationHistory(max_turns=3)

    def test_add_turn_and_length(self):
        """Verify adding turns increments history count."""
        self.assertEqual(len(self.history), 0)
        turn1 = ConversationTurn(
            turn_index=1,
            user_raw_query="What are Severity 1 reporting timelines?",
            rewritten_query="What are Severity 1 reporting timelines?",
            is_rewritten=False,
            rewriting_method="STANDALONE_PASSTHROUGH",
            rewriting_latency=0.001,
            raw_top_score=0.75,
            rewritten_top_score=0.75,
            score_uplift=0.0,
            raw_retrieved_chunk_ids=["chunk_001"],
            retrieved_chunk_ids=["chunk_001"],
            retrieved_documents=["cyber_resilience.pdf"],
            assistant_answer="Severity 1 incidents must be reported within 6 hours.",
            citations=["[1]"],
            total_latency_seconds=0.5,
        )
        self.history.add_turn(turn1)
        self.assertEqual(len(self.history), 1)
        self.assertEqual(self.history.turns[0].user_raw_query, "What are Severity 1 reporting timelines?")

    def test_get_recent_turns_windowing(self):
        """Verify windowing limits the number of returned turns."""
        for i in range(1, 5):
            turn = ConversationTurn(
                turn_index=i,
                user_raw_query=f"Query {i}",
                rewritten_query=f"Rewritten Query {i}",
                is_rewritten=False,
                rewriting_method="STANDALONE_PASSTHROUGH",
                rewriting_latency=0.01,
                raw_top_score=0.5,
                rewritten_top_score=0.5,
                score_uplift=0.0,
                raw_retrieved_chunk_ids=[],
                retrieved_chunk_ids=[],
                retrieved_documents=[],
                assistant_answer=f"Answer {i}",
                citations=[],
            )
            self.history.add_turn(turn)

        # max_turns is 3
        recent_default = self.history.get_recent_turns()
        self.assertEqual(len(recent_default), 3)
        self.assertEqual(recent_default[0].turn_index, 2)
        self.assertEqual(recent_default[2].turn_index, 4)

        # Explicit limit k=2
        recent_k2 = self.history.get_recent_turns(k=2)
        self.assertEqual(len(recent_k2), 2)
        self.assertEqual(recent_k2[0].turn_index, 3)
        self.assertEqual(recent_k2[1].turn_index, 4)

    def test_format_history_for_rewriter(self):
        """Verify plain text history formatting for query rewriter prompt."""
        turn = ConversationTurn(
            turn_index=1,
            user_raw_query="What are reporting timelines for cyber incidents?",
            rewritten_query="What are reporting timelines for cyber incidents?",
            is_rewritten=False,
            rewriting_method="STANDALONE_PASSTHROUGH",
            rewriting_latency=0.01,
            raw_top_score=0.65,
            rewritten_top_score=0.65,
            score_uplift=0.0,
            raw_retrieved_chunk_ids=[],
            retrieved_chunk_ids=[],
            retrieved_documents=[],
            assistant_answer="Banks must report within 6 hours to RBI and CERT-In.",
            citations=["[1]"],
        )
        self.history.add_turn(turn)
        formatted = self.history.format_history_for_rewriter()
        self.assertIn("Turn 1:", formatted)
        self.assertIn("User: What are reporting timelines for cyber incidents?", formatted)
        self.assertIn("Assistant: Banks must report within 6 hours to RBI and CERT-In.", formatted)

    def test_format_chat_messages(self):
        """Verify chat completion messages format with system prompt and history."""
        turn = ConversationTurn(
            turn_index=1,
            user_raw_query="Initial Q",
            rewritten_query="Initial Q",
            is_rewritten=False,
            rewriting_method="STANDALONE_PASSTHROUGH",
            rewriting_latency=0.01,
            raw_top_score=0.5,
            rewritten_top_score=0.5,
            score_uplift=0.0,
            raw_retrieved_chunk_ids=[],
            retrieved_chunk_ids=[],
            retrieved_documents=[],
            assistant_answer="Initial A",
            citations=[],
        )
        self.history.add_turn(turn)

        messages = self.history.format_chat_messages(
            system_prompt="You are RegulSense.",
            current_user_query="Follow-up Q",
        )
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0], {"role": "system", "content": "You are RegulSense."})
        self.assertEqual(messages[1], {"role": "user", "content": "Initial Q"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "Initial A"})
        self.assertEqual(messages[3], {"role": "user", "content": "Follow-up Q"})

    def test_clear_history(self):
        """Verify clear() resets the history."""
        turn = ConversationTurn(
            turn_index=1,
            user_raw_query="Q",
            rewritten_query="Q",
            is_rewritten=False,
            rewriting_method="STANDALONE_PASSTHROUGH",
            rewriting_latency=0.01,
            raw_top_score=0.5,
            rewritten_top_score=0.5,
            score_uplift=0.0,
            raw_retrieved_chunk_ids=[],
            retrieved_chunk_ids=[],
            retrieved_documents=[],
            assistant_answer="A",
            citations=[],
        )
        self.history.add_turn(turn)
        self.assertEqual(len(self.history), 1)
        self.history.clear()
        self.assertEqual(len(self.history), 0)


class TestQueryRewriter(unittest.TestCase):
    """Tests for Task 2: Rewrite Follow-up Questions."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.rewriter = QueryRewriter(openai_client=self.mock_client)
        self.history = ConversationHistory()

    def test_has_follow_up_cues(self):
        """Verify detection of pronouns, demonstratives, and follow-up cues."""
        self.assertTrue(self.rewriter.has_follow_up_cues("What must be submitted after that initial report?"))
        self.assertTrue(self.rewriter.has_follow_up_cues("How long do we need to store it?"))
        self.assertTrue(self.rewriter.has_follow_up_cues("What are the supervisory penalties for them?"))
        self.assertTrue(self.rewriter.has_follow_up_cues("What about the deadline?"))
        self.assertFalse(self.rewriter.has_follow_up_cues("Customer Due Diligence Master Direction provisions."))

    def test_turn_one_standalone_passthrough(self):
        """Verify initial turn is passed through without LLM call."""
        result = self.rewriter.rewrite("What are the reporting timelines for cyber incidents?", self.history)
        self.assertFalse(result.is_rewritten)
        self.assertEqual(result.method, "STANDALONE_PASSTHROUGH")
        self.assertEqual(result.rewritten_query, "What are the reporting timelines for cyber incidents?")
        self.mock_client.chat.completions.create.assert_not_called()

    def test_llm_query_rewriting_success(self):
        """Verify LLM query rewriting reformulates ambiguous follow-up."""
        # Add Turn 1 to history
        self.history.add_turn(
            ConversationTurn(
                turn_index=1,
                user_raw_query="What are reporting timelines for Severity 1 cyber security incidents?",
                rewritten_query="What are reporting timelines for Severity 1 cyber security incidents?",
                is_rewritten=False,
                rewriting_method="STANDALONE_PASSTHROUGH",
                rewriting_latency=0.01,
                raw_top_score=0.7,
                rewritten_top_score=0.7,
                score_uplift=0.0,
                raw_retrieved_chunk_ids=[],
                retrieved_chunk_ids=[],
                retrieved_documents=[],
                assistant_answer="Must report to RBI CSITE within 6 hours.",
                citations=["[1]"],
            )
        )

        mock_choice = MagicMock()
        mock_choice.message.content = "What comprehensive forensic report must be submitted after the initial 6-hour cyber incident notification?"
        self.mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        result = self.rewriter.rewrite("What must be submitted after that initial report?", self.history)
        self.assertTrue(result.is_rewritten)
        self.assertEqual(result.method, "LLM")
        self.assertIn("6-hour cyber incident", result.rewritten_query)
        self.mock_client.chat.completions.create.assert_called_once()

    def test_heuristic_fallback_when_llm_fails(self):
        """Verify deterministic heuristic fallback when LLM throws exception."""
        self.history.add_turn(
            ConversationTurn(
                turn_index=1,
                user_raw_query="What are cyber incident reporting timelines to RBI?",
                rewritten_query="What are cyber incident reporting timelines to RBI?",
                is_rewritten=False,
                rewriting_method="STANDALONE_PASSTHROUGH",
                rewriting_latency=0.01,
                raw_top_score=0.7,
                rewritten_top_score=0.7,
                score_uplift=0.0,
                raw_retrieved_chunk_ids=[],
                retrieved_chunk_ids=[],
                retrieved_documents=[],
                assistant_answer="Must report within 6 hours.",
                citations=["[1]"],
            )
        )

        self.mock_client.chat.completions.create.side_effect = RuntimeError("Ollama connection timeout")

        result = self.rewriter.rewrite("What must be submitted after that initial report?", self.history)
        self.assertTrue(result.is_rewritten)
        self.assertEqual(result.method, "HEURISTIC_FALLBACK")
        self.assertIn("forensic analysis report", result.rewritten_query)
        self.assertIn("cyber security incident", result.rewritten_query)


class TestConversationalRAGRetrievalAndDialogue(unittest.TestCase):
    """Tests for Task 3 & 4: Retrieve using rewritten query & Multi-turn execution."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_retriever = MagicMock()
        self.mock_citation_engine = MagicMock()
        self.mock_rewriter = MagicMock()

        self.rag = ConversationalRAG(
            openai_client=self.mock_client,
            retriever=self.mock_retriever,
            citation_engine=self.mock_citation_engine,
            query_rewriter=self.mock_rewriter,
        )

    def test_retrieve_with_comparison_score_uplift(self):
        """Verify Task 3: retrieve_with_comparison tracks score uplift between raw and rewritten query."""
        raw_record = RetrievedRecord(
            rank=1,
            id="circular_dor_004",
            similarity_score=0.34,
            distance=0.66,
            document="Irrelevant text",
            metadata={"source_document": "circular_dor_2024_108.txt"},
        )
        rewritten_record = RetrievedRecord(
            rank=1,
            id="cyber_resilience_001",
            similarity_score=0.64,
            distance=0.36,
            document="Forensic report within 7 business days",
            metadata={"source_document": "cyber_resilience_framework.pdf"},
        )

        self.mock_retriever.retrieve.side_effect = [
            RetrievalRunResult(
                query_text="rewritten query",
                k=1,
                retrieved_count=1,
                chunks=[rewritten_record],
                top_score=0.64,
                lowest_score=0.64,
                mean_score=0.64,
                total_tokens_retrieved=20,
                source_documents=["cyber_resilience_framework.pdf"],
            ),
            RetrievalRunResult(
                query_text="raw query",
                k=1,
                retrieved_count=1,
                chunks=[raw_record],
                top_score=0.34,
                lowest_score=0.34,
                mean_score=0.34,
                total_tokens_retrieved=20,
                source_documents=["circular_dor_2024_108.txt"],
            ),
        ]

        rewritten_res, raw_res, metrics = self.rag.retrieve_with_comparison(
            raw_query="raw query",
            rewritten_query="rewritten query",
            top_k=1,
        )

        self.assertEqual(metrics["raw_top_score"], 0.34)
        self.assertEqual(metrics["rewritten_top_score"], 0.64)
        self.assertAlmostEqual(metrics["score_uplift"], 0.30, places=2)
        self.assertEqual(metrics["rewritten_documents"], ["cyber_resilience_framework.pdf"])

    def test_multi_turn_dialogue_flow(self):
        """Verify Task 4: end-to-end multi-turn dialogue orchestration and history accumulation."""
        chunk1 = RetrievedRecord(
            rank=1,
            id="cyber_resilience_001",
            similarity_score=0.72,
            distance=0.28,
            document="Severity 1 incidents must be reported in 6 hours.",
            metadata={"source_document": "cyber_resilience_framework.pdf"},
        )

        audit_report = CitationAuditReport(
            total_citations_found=1,
            unique_markers_cited=["[1]"],
            verified_citations_count=1,
            fabricated_citations_count=0,
            unsupported_claims_count=0,
            citation_precision=1.0,
            claim_verifications=[],
            fabricated_markers=[],
            overall_verdict="PASS",
            audit_notes="Verified.",
        )
        mock_output = CitedAnswerOutput(
            query="test",
            answer="Reporting timeline is 6 hours [1].",
            citation_registry={"[1]": {"source_document": "cyber_resilience_framework.pdf"}},
            audit_report=audit_report,
            is_fallback=False,
            has_fabricated_citations=False,
            prompt_tokens=100,
            completion_tokens=40,
            latency_seconds=0.3,
            model="llama3:latest",
        )

        self.mock_rewriter.rewrite.return_value = QueryRewritingResult(
            raw_query="What are reporting timelines?",
            rewritten_query="What are reporting timelines?",
            is_rewritten=False,
            method="STANDALONE_PASSTHROUGH",
            latency_seconds=0.01,
            reason="Standalone.",
        )
        self.mock_retriever.retrieve.return_value = RetrievalRunResult(
            query_text="test",
            k=1,
            retrieved_count=1,
            chunks=[chunk1],
            top_score=0.72,
            lowest_score=0.72,
            mean_score=0.72,
            total_tokens_retrieved=20,
            source_documents=["cyber_resilience_framework.pdf"],
        )
        self.mock_citation_engine.generate_cited_answer.return_value = mock_output

        # Turn 1
        turn1 = self.rag.ask("What are reporting timelines?")
        self.assertEqual(turn1.turn_index, 1)
        self.assertEqual(len(self.rag.history), 1)
        self.assertEqual(turn1.citations, ["[1]"])

        # Turn 2: Follow up
        self.mock_rewriter.rewrite.return_value = QueryRewritingResult(
            raw_query="What after that?",
            rewritten_query="What must be submitted after initial cyber report?",
            is_rewritten=True,
            method="LLM",
            latency_seconds=0.05,
            reason="Pronoun resolved.",
        )
        turn2 = self.rag.ask("What after that?")
        self.assertEqual(turn2.turn_index, 2)
        self.assertTrue(turn2.is_rewritten)
        self.assertEqual(len(self.rag.history), 2)


class TestArtifactExport(unittest.TestCase):
    """Tests for Task 5: Artifact Serialization."""

    def test_export_conversational_artifacts(self):
        """Verify export_conversational_artifacts generates valid Markdown and JSON files."""
        turns = [
            ConversationTurn(
                turn_index=1,
                user_raw_query="What are reporting timelines?",
                rewritten_query="What are reporting timelines?",
                is_rewritten=False,
                rewriting_method="STANDALONE_PASSTHROUGH",
                rewriting_latency=0.001,
                raw_top_score=0.70,
                rewritten_top_score=0.70,
                score_uplift=0.0,
                raw_retrieved_chunk_ids=["cyber_001"],
                retrieved_chunk_ids=["cyber_001"],
                retrieved_documents=["cyber_resilience_framework.pdf"],
                assistant_answer="Severity 1 incidents must be reported in 6 hours [1].",
                citations=["[1]"],
                total_latency_seconds=0.4,
            ),
            ConversationTurn(
                turn_index=2,
                user_raw_query="What must be submitted after that initial report?",
                rewritten_query="What forensic report must be submitted after the initial 6-hour cyber report?",
                is_rewritten=True,
                rewriting_method="LLM",
                rewriting_latency=0.05,
                raw_top_score=0.34,
                rewritten_top_score=0.64,
                score_uplift=0.30,
                raw_retrieved_chunk_ids=["circular_004"],
                retrieved_chunk_ids=["cyber_001"],
                retrieved_documents=["cyber_resilience_framework.pdf"],
                assistant_answer="A forensic analysis report within 7 business days [1].",
                citations=["[1]"],
                total_latency_seconds=0.6,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            md_file, json_file = export_conversational_artifacts(turns=turns, output_dir=temp_path)

            self.assertTrue(md_file.exists())
            self.assertTrue(json_file.exists())

            # Verify markdown content
            md_text = md_file.read_text(encoding="utf-8")
            self.assertIn("Conversational RAG Evaluation Report", md_text)
            self.assertIn("Turn 1", md_text)
            self.assertIn("Turn 2", md_text)
            self.assertIn("What must be submitted after that initial report?", md_text)
            self.assertIn("Score Uplift", md_text)

            # Verify JSON schema
            json_text = json_file.read_text(encoding="utf-8")
            data = json.loads(json_text)
            self.assertEqual(data["metadata"]["total_turns"], 2)
            self.assertEqual(data["metadata"]["rewritten_turns"], 1)
            self.assertEqual(len(data["turns"]), 2)
            self.assertEqual(data["turns"][1]["turn_index"], 2)
            self.assertTrue(data["turns"][1]["is_rewritten"])


if __name__ == "__main__":
    unittest.main()
