"""Unit and integration test suite for RegulSense Streamlit Chat Interface.

Tests:
1. RegulSenseAPIClient methods (health check, query, upload, error handling).
2. Client-side input validation and error envelopes.
3. Source formatting and metadata parsing.
4. Safe guardrail refusal response handling.
5. Streamlit AppTest headless execution and session state lifecycle.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import ClientResponse, RegulSenseAPIClient


class TestRegulSenseAPIClient(unittest.TestCase):
    """Tests for the RegulSenseAPIClient networking and error handling."""

    def setUp(self):
        self.client = RegulSenseAPIClient(
            base_url="http://mock-regulsense:8000",
            default_timeout=5.0,
            enable_direct_fallback=False,
        )

    def test_client_initialization(self):
        """Verifies default properties and base URL trimming."""
        self.assertEqual(self.client.base_url, "http://mock-regulsense:8000")
        self.assertEqual(self.client.default_timeout, 5.0)
        self.assertFalse(self.client.enable_direct_fallback)

    def test_input_validation_empty_query(self):
        """Verifies client rejects empty or whitespace queries before making network calls."""
        res_empty = self.client.submit_query("")
        self.assertFalse(res_empty.success)
        self.assertEqual(res_empty.error_code, "EMPTY_QUESTION")
        self.assertEqual(res_empty.status_code, 400)

        res_spaces = self.client.submit_query("   \n\t  ")
        self.assertFalse(res_spaces.success)
        self.assertEqual(res_spaces.error_code, "EMPTY_QUESTION")

    def test_input_validation_short_query(self):
        """Verifies client rejects queries shorter than 3 characters."""
        res = self.client.submit_query("hi")
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "QUESTION_TOO_SHORT")
        self.assertEqual(res.status_code, 400)

    @patch("requests.get")
    def test_health_check_success(self, mock_get):
        """Verifies successful health check parsing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "healthy",
            "timestamp": "2026-09-15T12:00:00Z",
            "vector_db_reachable": True,
            "collection_name": "regulsense_regulatory_chunks",
            "chat_model": "llama3:latest",
            "embedding_model": "all-minilm",
        }
        mock_get.return_value = mock_resp

        res = self.client.check_health()
        self.assertTrue(res.success)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.data["vector_db_reachable"])
        self.assertEqual(res.data["chat_model"], "llama3:latest")

    @patch("requests.get")
    def test_health_check_connection_error(self, mock_get):
        """Verifies connection failure returns status 503 with BACKEND_OFFLINE."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        res = self.client.check_health()
        self.assertFalse(res.success)
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.error_code, "BACKEND_OFFLINE")
        self.assertIn("unreachable", res.error_message.lower())

    @patch("requests.post")
    def test_submit_query_success_with_sources(self, mock_post):
        """Verifies successful query response parsing with sources and citations."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "answer": "Banks must notify CERT-In within 6 hours of detecting Severity 1 incidents [1].",
            "sources": [
                {
                    "marker": "[1]",
                    "source_document": "RBI_Cyber_Resilience_Framework.txt",
                    "chunk_id": "chunk_rbi_sec4_001",
                    "section": "Section 4: Incident Response",
                    "page_number": 12,
                    "similarity_score": 0.884,
                    "verbatim_text": "All scheduled commercial banks shall report cybersecurity incidents...",
                }
            ],
            "citations": ["[1]"],
            "metadata": {
                "latency_seconds": 0.42,
                "model": "llama3:latest",
                "top_similarity_score": 0.884,
            },
        }
        mock_post.return_value = mock_resp

        res = self.client.submit_query(
            question="What is the timeframe to report Severity 1 incidents to CERT-In?",
            top_k=3,
        )

        self.assertTrue(res.success)
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.is_refusal)
        self.assertIn("[1]", res.answer)
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0]["chunk_id"], "chunk_rbi_sec4_001")
        self.assertEqual(res.citations, ["[1]"])

    @patch("requests.post")
    def test_submit_query_safe_guardrail_refusal(self, mock_post):
        """Verifies guardrail refusal payload is correctly identified."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "refusal",
            "answer": "I don't have sufficient regulatory context to answer this query with confidence.",
            "sources": [],
            "citations": [],
            "metadata": {
                "is_refusal": True,
                "top_similarity_score": 0.21,
                "guardrail_status": "REFUSAL_LOW_SIMILARITY",
            },
        }
        mock_post.return_value = mock_resp

        res = self.client.submit_query(
            question="What are reserve requirements for lunar colony banks?",
        )

        self.assertTrue(res.success)
        self.assertTrue(res.is_refusal)
        self.assertIn("sufficient regulatory context", res.answer)
        self.assertEqual(len(res.sources), 0)

    @patch("requests.post")
    def test_submit_query_server_error_500(self, mock_post):
        """Verifies HTTP 500 error envelope is parsed without crashing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {
            "status": "error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected internal error occurred.",
        }
        mock_post.return_value = mock_resp

        res = self.client.submit_query(
            question="Any valid question here?",
        )

        self.assertFalse(res.success)
        self.assertEqual(res.status_code, 500)
        self.assertEqual(res.error_code, "INTERNAL_SERVER_ERROR")
        self.assertIn("unexpected internal error", res.error_message)

    @patch("requests.post")
    def test_upload_document_success(self, mock_post):
        """Verifies runtime upload endpoint call and response structure."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "filename": "RBI_UPI_Circular.txt",
            "file_size_bytes": 1024,
            "chunks_created": 4,
            "records_indexed": 4,
            "collection_name": "regulsense_regulatory_chunks",
            "searchable_immediately": True,
        }
        mock_post.return_value = mock_resp

        res = self.client.upload_document(
            file_bytes=b"Sample regulatory circular text content...",
            filename="RBI_UPI_Circular.txt",
        )

        self.assertTrue(res.success)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data["chunks_created"], 4)
        self.assertTrue(res.data["searchable_immediately"])


class TestChatAppHeadless(unittest.TestCase):
    """Tests Streamlit application initialization using streamlit.testing.v1.AppTest."""

    def test_app_initialization_headless(self):
        """Runs the Streamlit app headlessly and verifies default controls and initial message."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1 not available in environment.")

        app_file = str(PROJECT_ROOT / "src" / "chat_app.py")
        at = AppTest.from_file(app_file, default_timeout=15.0)

        # Mock health check so the app runs offline without network dependency
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

            # Ensure no unhandled exceptions occurred
            self.assertFalse(at.exception, f"App threw unexpected exception: {at.exception}")

            # Verify sidebar sliders and inputs rendered
            self.assertGreater(len(at.sidebar.slider), 0)
            self.assertEqual(at.sidebar.slider[0].value, 3)

            # Verify session state initialized
            self.assertIn("messages", at.session_state)
            self.assertGreaterEqual(len(at.session_state["messages"]), 1)
            self.assertEqual(at.session_state["messages"][0]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
