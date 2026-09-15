"""Unit and Integration Tests for RegulSense Backend API Service.

Tests cover:
- Task 1: Query endpoint routing and request dispatch.
- Task 2: Structured JSON response schemas with answers, sources, citations, and metadata.
- Task 3: Pydantic input validation and HTTP status code error handling (400, 422, 500).
- Task 4: Environment-driven configuration loading and secret redaction.
- Task 5: Sample request and response artifact export.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api import (
    AppConfig,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
    create_app,
    export_sample_api_artifacts,
)
from src.citation_engine import CitationAuditReport, CitedAnswerOutput
from src.hallucination_guardrails import GuardrailExecutionResult, RetrievalQualityAssessment
from src.vector_db import RetrievedRecord


class TestAppConfig(unittest.TestCase):
    """Tests for Task 4: Load Config from Environment."""

    def test_default_config_loading(self):
        """Verify AppConfig loads sensible default values."""
        cfg = AppConfig()
        self.assertIn("11434", cfg.openai_base_url)
        self.assertEqual(cfg.chat_model, os.getenv("CHAT_MODEL", "llama3:latest"))
        self.assertEqual(cfg.chroma_collection, "regulsense_regulatory_chunks")
        self.assertEqual(cfg.api_port, 8000)
        self.assertIsInstance(cfg.chroma_persist_dir, Path)

    def test_environment_variable_overrides(self):
        """Verify environment variables take precedence in AppConfig."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "http://custom-host:8080/v1",
                "CHAT_MODEL": "gpt-4o-mini",
                "API_PORT": "9000",
                "CHROMA_COLLECTION_NAME": "custom_regulatory_collection",
                "DEFAULT_TOP_K": "5",
                "MIN_SIMILARITY_THRESHOLD": "0.65",
            },
            clear=False,
        ):
            cfg = AppConfig()
            self.assertEqual(cfg.openai_base_url, "http://custom-host:8080/v1")
            self.assertEqual(cfg.chat_model, "gpt-4o-mini")
            self.assertEqual(cfg.api_port, 9000)
            self.assertEqual(cfg.chroma_collection, "custom_regulatory_collection")
            self.assertEqual(cfg.default_top_k, 5)
            self.assertEqual(cfg.min_similarity_threshold, 0.65)

    def test_secret_redaction(self):
        """Verify API keys are masked when exporting configuration dictionaries."""
        cfg = AppConfig(openai_api_key="secret-production-token-12345")
        data = cfg.to_dict(redact_secrets=True)
        self.assertEqual(data["openai_api_key"], "***REDACTED***")

        unredacted = cfg.to_dict(redact_secrets=False)
        self.assertEqual(unredacted["openai_api_key"], "secret-production-token-12345")


class TestAPIEndpoints(unittest.TestCase):
    """Tests for Tasks 1, 2, 3: Query Endpoint, Structured JSON, Validation & Error Handling."""

    def setUp(self):
        self.mock_guardrail = MagicMock()
        self.mock_db_manager = MagicMock()
        self.mock_db_manager.is_reachable.return_value = True

        self.config = AppConfig(app_env="test")
        self.app = create_app(
            config=self.config,
            guardrail=self.mock_guardrail,
            vector_db_manager=self.mock_db_manager,
        )
        self.client = TestClient(self.app)

    def test_health_check_endpoint(self):
        """Verify GET /api/v1/health and /health return 200 OK with connectivity metrics."""
        for path in ["/api/v1/health", "/health"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["vector_db_reachable"])
            self.assertEqual(data["collection_name"], "regulsense_regulatory_chunks")

    def test_config_endpoint_redaction(self):
        """Verify GET /api/v1/config returns redacted configuration safely."""
        response = self.client.get("/api/v1/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["config"]["openai_api_key"], "***REDACTED***")

    def test_successful_query_endpoint_structured_json(self):
        """Verify Task 1 & 2: POST /api/v1/query returns structured JSON with answer, sources, and metadata."""
        mock_output = CitedAnswerOutput(
            query="What are reporting timelines for cyber incidents?",
            answer="Incidents must be reported to CERT-In and RBI within 6 hours of detection [1].",
            citation_registry={
                "[1]": {
                    "source_document": "cyber_resilience_framework.pdf",
                    "chunk_id": "cyber_resilience_001",
                    "section": "2. Incident Reporting Timelines",
                    "page_number": 1,
                    "similarity_score": 0.7250,
                    "verbatim_text": "Report to RBI and CERT-In within 6 hours.",
                }
            },
            audit_report=CitationAuditReport(
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
            ),
            is_fallback=False,
            has_fabricated_citations=False,
            prompt_tokens=250,
            completion_tokens=60,
            latency_seconds=0.45,
            model="llama3:latest",
        )

        mock_assessment = RetrievalQualityAssessment(
            is_sufficient=True,
            status="SUFFICIENT_CONTEXT",
            top_score=0.7250,
            mean_qualifying_score=0.7250,
            qualifying_chunk_count=1,
            total_retrieved_count=3,
            threshold_used=0.50,
            min_chunks_required=1,
            qualifying_chunks=[],
        )

        self.mock_guardrail.execute.return_value = GuardrailExecutionResult(
            query="What are reporting timelines for cyber incidents?",
            action="ANSWER",
            answer=mock_output.answer,
            quality_assessment=mock_assessment,
            is_refusal=False,
            citations=["[1]"],
            cited_output=mock_output,
            latency_seconds=0.45,
            model="llama3:latest",
        )

        payload = {
            "question": "What are reporting timelines for cyber incidents?",
            "top_k": 3,
            "include_metadata": True,
        }

        # Test both primary endpoint and root alias
        for path in ["/api/v1/query", "/query"]:
            response = self.client.post(path, json=payload)
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertIn("6 hours", data["answer"])
            self.assertEqual(data["citations"], ["[1]"])

            # Verify sources array structure (Task 2)
            self.assertEqual(len(data["sources"]), 1)
            source = data["sources"][0]
            self.assertEqual(source["marker"], "[1]")
            self.assertEqual(source["source_document"], "cyber_resilience_framework.pdf")
            self.assertEqual(source["chunk_id"], "cyber_resilience_001")
            self.assertEqual(source["similarity_score"], 0.7250)

            # Verify metadata
            self.assertEqual(data["metadata"]["model"], "llama3:latest")
            self.assertEqual(data["metadata"]["top_k"], 3)
            self.assertFalse(data["metadata"]["is_refusal"])

    def test_safe_refusal_response_structure(self):
        """Verify out-of-corpus query returns structured refusal with status 'refusal'."""
        mock_assessment = RetrievalQualityAssessment(
            is_sufficient=False,
            status="LOW_SIMILARITY_SCORE",
            top_score=0.35,
            mean_qualifying_score=0.0,
            qualifying_chunk_count=0,
            total_retrieved_count=3,
            threshold_used=0.50,
            min_chunks_required=1,
            refusal_reason="Similarity below threshold.",
            qualifying_chunks=[],
        )

        self.mock_guardrail.execute.return_value = GuardrailExecutionResult(
            query="What are Basel III rural bank buffers?",
            action="REFUSE",
            answer="The provided regulatory context does not contain sufficient information to answer this question.",
            quality_assessment=mock_assessment,
            is_refusal=True,
            citations=[],
            cited_output=None,
            latency_seconds=0.05,
            model="llama3:latest",
        )

        response = self.client.post("/api/v1/query", json={"question": "What are Basel III rural bank buffers?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "refusal")
        self.assertIn("sufficient information", data["answer"].lower())
        self.assertEqual(data["sources"], [])
        self.assertTrue(data["metadata"]["is_refusal"])

    # -------------------------------------------------------------------------
    # Task 3: Input Validation and Error Handling Tests
    # -------------------------------------------------------------------------

    def test_missing_question_field_returns_422(self):
        """Verify missing question field returns HTTP 422 Unprocessable Entity."""
        response = self.client.post("/api/v1/query", json={})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")
        self.assertIn("detail", data)

    def test_empty_or_whitespace_question_returns_422(self):
        """Verify empty or whitespace-only question is rejected with 422."""
        for bad_q in ["", "   ", " \t \n "]:
            response = self.client.post("/api/v1/query", json={"question": bad_q})
            self.assertEqual(response.status_code, 422)
            data = response.json()
            self.assertEqual(data["error_code"], "VALIDATION_ERROR")

    def test_too_short_question_returns_422(self):
        """Verify questions shorter than 3 characters are rejected with 422."""
        response = self.client.post("/api/v1/query", json={"question": "ab"})
        self.assertEqual(response.status_code, 422)

    def test_too_long_question_returns_422(self):
        """Verify questions exceeding 2000 characters are rejected with 422."""
        giant_q = "What are the rules? " * 200  # > 3000 chars
        response = self.client.post("/api/v1/query", json={"question": giant_q})
        self.assertEqual(response.status_code, 422)

    def test_invalid_top_k_returns_422(self):
        """Verify top_k out of bounds (<= 0 or > 20) is rejected with 422."""
        for bad_k in [0, -5, 25, 100]:
            response = self.client.post(
                "/api/v1/query",
                json={"question": "Valid compliance question?", "top_k": bad_k},
            )
            self.assertEqual(response.status_code, 422)

    def test_server_error_handling_returns_500(self):
        """Verify internal server exceptions return HTTP 500 without leaking stack traces."""
        self.mock_guardrail.execute.side_effect = RuntimeError("Fatal ChromaDB storage crash")

        response = self.client.post("/api/v1/query", json={"question": "Valid compliance question?"})
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], "INTERNAL_SERVER_ERROR")
        self.assertIn("unexpected internal error", data["message"].lower())


class TestArtifactExport(unittest.TestCase):
    """Tests for Task 5: Sample Request and Response Artifacts."""

    def test_export_sample_api_artifacts(self):
        """Verify export_sample_api_artifacts generates valid JSON and Markdown files."""
        mock_guard = MagicMock()
        mock_vdb = MagicMock()
        mock_vdb.is_reachable.return_value = True

        mock_output = CitedAnswerOutput(
            query="Test query",
            answer="Sample answer [1].",
            citation_registry={"[1]": {"source_document": "cyber_resilience.pdf"}},
            audit_report=MagicMock(unique_markers_cited=["[1]"]),
            is_fallback=False,
            has_fabricated_citations=False,
            prompt_tokens=100,
            completion_tokens=20,
            latency_seconds=0.1,
            model="llama3:latest",
        )
        mock_guard.execute.return_value = GuardrailExecutionResult(
            query="Test query",
            action="ANSWER",
            answer="Sample answer [1].",
            quality_assessment=MagicMock(status="SUFFICIENT_CONTEXT", top_score=0.75, qualifying_chunks=[]),
            is_refusal=False,
            citations=["[1]"],
            cited_output=mock_output,
            latency_seconds=0.1,
            model="llama3:latest",
        )

        test_app = create_app(
            config=AppConfig(app_env="test"),
            guardrail=mock_guard,
            vector_db_manager=mock_vdb,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            req_f, res_f, md_f = export_sample_api_artifacts(app_instance=test_app, output_dir=tmp_path)

            self.assertTrue(req_f.exists())
            self.assertTrue(res_f.exists())
            self.assertTrue(md_f.exists())

            # Verify Request JSON
            req_data = json.loads(req_f.read_text(encoding="utf-8"))
            self.assertIn("question", req_data)
            self.assertIn("top_k", req_data)

            # Verify Response JSON
            res_data = json.loads(res_f.read_text(encoding="utf-8"))
            self.assertIn("status", res_data)
            self.assertIn("answer", res_data)
            self.assertIn("sources", res_data)
            self.assertIn("citations", res_data)
            self.assertIn("metadata", res_data)

            # Verify Markdown Documentation
            md_text = md_f.read_text(encoding="utf-8")
            self.assertIn("RegulSense Backend API Specification", md_text)
            self.assertIn("/api/v1/query", md_text)
            self.assertIn("/api/v1/health", md_text)
            self.assertIn("422", md_text)


if __name__ == "__main__":
    unittest.main()
