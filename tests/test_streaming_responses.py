"""Unit and integration test suite for Progressive Streaming Responses and Citation Inspection.

Tests:
1. CitationEngine.stream_cited_answer generator yields (sources, token deltas, done).
2. HallucinationGuardrail.execute_stream handles confident streaming and refusal streaming.
3. Backend FastAPI /api/v1/query/stream SSE streaming endpoint.
4. RegulSenseAPIClient.submit_query_stream client SSE parsing and error recovery.
5. Headless Streamlit AppTest execution with streaming response handlers.
"""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app
from src.api_client import ClientResponse, RegulSenseAPIClient
from src.citation_engine import CitationAuditReport, CitationEngine, CitedAnswerOutput
from src.hallucination_guardrails import GuardrailExecutionResult, HallucinationGuardrail


class TestCitationEngineStreaming(unittest.TestCase):
    """Unit tests for CitationEngine streaming generator."""

    @patch("openai.resources.chat.completions.Completions.create")
    def test_stream_cited_answer_yields_events(self, mock_create):
        """Verifies stream_cited_answer yields sources first, then tokens, then done."""
        # Setup mock stream chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Under RBI guidelines [1], "))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="banks must report within 6 hours."))]

        mock_create.return_value = [chunk1, chunk2]

        engine = CitationEngine()
        from src.vector_db import RetrievedRecord
        mock_chunk = RetrievedRecord(
            rank=1,
            id="test_chunk_001",
            similarity_score=0.85,
            distance=0.15,
            document="Under RBI guidelines, banks must report within 6 hours.",
            metadata={
                "source_document": "rbi_test.txt",
                "section": "Reporting Timelines",
                "page_number": 1,
            },
        )

        events = list(engine.stream_cited_answer(
            query="What is the reporting timeline?",
            chunks=[mock_chunk],
        ))

        self.assertGreaterEqual(len(events), 3)

        # First event must be "sources"
        self.assertEqual(events[0][0], "sources")
        self.assertIn("[1]", events[0][1])

        # Intermediate events must be "token"
        tokens = [item[1] for item in events if item[0] == "token"]
        self.assertEqual(len(tokens), 2)
        self.assertIn("Under RBI guidelines [1], ", tokens[0])

        # Final event must be "done"
        self.assertEqual(events[-1][0], "done")
        done_output = events[-1][1]
        self.assertIsInstance(done_output, CitedAnswerOutput)
        self.assertIn("[1]", done_output.audit_report.unique_markers_cited)


class TestGuardrailStreaming(unittest.TestCase):
    """Unit tests for HallucinationGuardrail streaming generator."""

    def test_execute_stream_refusal(self):
        """Verifies execute_stream yields refusal tokens when context is weak."""
        guardrail = HallucinationGuardrail()
        events = list(guardrail.execute_stream(
            query="What is the capital reserve for lunar banks?",
            chunks=[],  # Empty chunks trigger immediate refusal
        ))

        # First event is empty sources
        self.assertEqual(events[0][0], "sources")
        self.assertEqual(events[0][1], {})

        # Intermediate events are refusal tokens
        token_events = [item for item in events if item[0] == "token"]
        self.assertGreater(len(token_events), 0)

        # Final event is done with is_refusal=True
        self.assertEqual(events[-1][0], "done")
        res = events[-1][1]
        self.assertIsInstance(res, GuardrailExecutionResult)
        self.assertTrue(res.is_refusal)
        self.assertEqual(res.action, "REFUSE")


class TestBackendStreamingEndpoint(unittest.TestCase):
    """Integration tests for FastAPI /api/v1/query/stream endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_query_stream_validation_error(self):
        """Verifies short/empty query returns HTTP 422 validation error."""
        resp = self.client.post("/api/v1/query/stream", json={"question": "hi"})
        self.assertEqual(resp.status_code, 422)

    @patch.object(HallucinationGuardrail, "execute_stream")
    def test_query_stream_sse_success(self, mock_execute_stream):
        """Verifies SSE format containing sources, token, and done events."""
        mock_res = GuardrailExecutionResult(
            query="What is incident reporting time?",
            action="ANSWER",
            answer="Report in 6 hours [1].",
            quality_assessment=None,
            is_refusal=False,
            citations=["[1]"],
            latency_seconds=0.25,
            model="llama3:latest",
        )
        mock_execute_stream.return_value = [
            ("sources", {"[1]": {"source_document": "rbi.txt", "chunk_id": "c1", "similarity_score": 0.8}}, []),
            ("token", "Report in "),
            ("token", "6 hours [1]."),
            ("done", mock_res),
        ]

        resp = self.client.post(
            "/api/v1/query/stream",
            json={"question": "What is incident reporting time?"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("content-type"), "text/event-stream; charset=utf-8")

        lines = resp.text.strip().split("\n\n")
        self.assertGreaterEqual(len(lines), 3)

        # Parse first line (sources event)
        first_evt = json.loads(lines[0].replace("data: ", ""))
        self.assertEqual(first_evt["event"], "sources")
        self.assertEqual(len(first_evt["data"]["sources"]), 1)

        # Parse token lines
        token_lines = [l for l in lines if '"event": "token"' in l]
        self.assertEqual(len(token_lines), 2)

        # Parse last line (done event)
        last_evt = json.loads(lines[-1].replace("data: ", ""))
        self.assertEqual(last_evt["event"], "done")
        self.assertEqual(last_evt["data"]["status"], "success")


class TestAPIClientStreaming(unittest.TestCase):
    """Unit tests for RegulSenseAPIClient.submit_query_stream."""

    def setUp(self):
        self.client = RegulSenseAPIClient(
            base_url="http://mock-api:8000",
            enable_direct_fallback=False,
        )

    def test_client_stream_validation_rejection(self):
        """Verifies client rejects empty or short questions before making request."""
        events_empty = list(self.client.submit_query_stream(""))
        self.assertEqual(len(events_empty), 1)
        self.assertEqual(events_empty[0]["type"], "error")
        self.assertEqual(events_empty[0]["error_code"], "EMPTY_QUESTION")

        events_short = list(self.client.submit_query_stream("no"))
        self.assertEqual(len(events_short), 1)
        self.assertEqual(events_short[0]["type"], "error")
        self.assertEqual(events_short[0]["error_code"], "QUESTION_TOO_SHORT")

    @patch("requests.post")
    def test_client_submit_query_stream_success(self, mock_post):
        """Verifies client parses SSE lines and yields typed event dictionaries."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            'data: {"event": "sources", "data": {"sources": [{"marker": "[1]", "source_document": "rbi.txt"}]}}',
            'data: {"event": "token", "data": {"token": "Notify "}}',
            'data: {"event": "token", "data": {"token": "CERT-In within 6 hours."}}',
            'data: {"event": "done", "data": {"status": "success", "answer": "Notify CERT-In within 6 hours.", "citations": ["[1]"]}}',
        ]
        mock_post.return_value = mock_resp

        events = list(self.client.submit_query_stream("What is the CERT-In deadline?"))
        self.assertEqual(len(events), 4)

        self.assertEqual(events[0]["type"], "sources")
        self.assertEqual(events[0]["sources"][0]["marker"], "[1]")

        self.assertEqual(events[1]["type"], "token")
        self.assertEqual(events[1]["token"], "Notify ")

        self.assertEqual(events[2]["type"], "token")
        self.assertEqual(events[2]["token"], "CERT-In within 6 hours.")

        self.assertEqual(events[3]["type"], "done")
        self.assertEqual(events[3]["citations"], ["[1]"])

    @patch("requests.post")
    def test_client_stream_connection_error_recovery(self, mock_post):
        """Verifies connection failure yields BACKEND_OFFLINE error event gracefully."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        events = list(self.client.submit_query_stream("Any valid question here?"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["error_code"], "BACKEND_OFFLINE")
        self.assertIn("unreachable", events[0]["message"].lower())


class TestChatAppStreamingHeadless(unittest.TestCase):
    """Headless AppTest execution verifying the Streamlit app loads and renders."""

    def test_chat_app_renders_with_streaming(self):
        """Verifies that the Streamlit app initializes with the streaming loop without crashing."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1 not available in environment.")

        app_file = str(PROJECT_ROOT / "src" / "chat_app.py")
        at = AppTest.from_file(app_file, default_timeout=15.0)

        with patch.object(
            RegulSenseAPIClient,
            "check_health",
            return_value=ClientResponse(
                success=True,
                status_code=200,
                data={
                    "status": "healthy",
                    "vector_db_reachable": True,
                    "collection_name": "regulsense_regulatory_chunks",
                    "chat_model": "llama3:latest",
                    "embedding_model": "all-minilm",
                },
                latency_seconds=0.01,
            ),
        ):
            at.run()
            self.assertFalse(at.exception, f"App threw unexpected exception: {at.exception}")
            self.assertIn("messages", at.session_state)


if __name__ == "__main__":
    unittest.main()
