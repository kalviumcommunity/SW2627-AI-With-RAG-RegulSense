"""Unit and Integration Tests for Document Upload and Runtime Indexing Endpoint.

Validates:
1. Task 1 - Create an upload endpoint:
   Multipart file upload accepting documents (.txt, .md, .pdf, .html), sanitizing filenames,
   and storing safely in the upload directory.
2. Task 2 - Ingest, embed, and index:
   End-to-end processing through DocumentLoader, TextCleaner, TokenAwareChunker, dense embeddings,
   and live ChromaDB upserting.
3. Task 3 - Confirm runtime searchability:
   Verifies that newly uploaded documents are immediately retrieved and cited by /api/v1/query
   without restarting the application.
4. Task 4 - Handle upload errors:
   Rejects unsupported extensions (400), 0-byte files (400), oversized files (413),
   empty filenames (400/422), and invalid chunking parameters (400).
5. Task 5 - Commit sample upload run:
   Verifies artifact generation for upload requests, indexing responses, searchability queries,
   and Markdown reports.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api import (
    AppConfig,
    UploadResponse,
    create_app,
    export_sample_upload_artifacts,
    ingest_and_index_document,
)
from src.citation_engine import CitationEngine, CitedAnswerOutput
from src.hallucination_guardrails import GuardrailExecutionResult, HallucinationGuardrail
from src.rag_pipeline import RetrievedContextChunk
from src.retriever import VectorRetriever
from src.vector_db import VectorDatabaseManager


SAMPLE_REGULATORY_DOC_TEXT = """RESERVE BANK OF INDIA
DEPARTMENT OF REGULATION
CENTRAL OFFICE, MUMBAI - 400 001

Circular No: RBI/2024-25/55 - DoR.LRG.REC.33/21.04.098/2024-25
Date: June 15, 2024

Subject: Master Direction on Liquidity Coverage Ratio (LCR) Disclosures and Stress Assumptions

1. Scope and Applicability
This Master Direction applies to all Scheduled Commercial Banks in India. Banks are required to maintain an adequate stock of unencumbered High Quality Liquid Assets (HQLA) that can be easily converted into cash to meet liquidity requirements for a 30-calendar day stress horizon.

2. Minimum LCR Standards
All covered institutions must maintain a minimum Liquidity Coverage Ratio of 100% on an ongoing basis. Any breach below 100% must be reported to the Reserve Bank of India within 2 hours.

3. Retail Deposit Run-off Assumptions
Stable retail deposits shall have a 5% outflow rate. Less stable retail deposits including high net-worth individuals shall carry a 10% outflow rate.
"""


class TestDocumentUploadEndpoint(unittest.TestCase):
    """Test suite for Document Upload & Runtime Ingestion Endpoint."""

    def setUp(self):
        """Set up test application with ephemeral in-memory ChromaDB and mocked LLM/Embedding client."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name) / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Unique collection name per test method to isolate vector counts
        test_method_name = self.id().split(".")[-1]
        unique_col = f"col_{test_method_name}_{int(time.time() * 1000)}"

        self.config = AppConfig(
            upload_dir=self.upload_dir,
            max_upload_size_bytes=1024 * 1024,  # 1 MB
            chroma_collection=unique_col,
            embedding_model="all-minilm",
            app_env="test",
        )

        # In-memory vector database
        self.vdb = VectorDatabaseManager(
            in_memory=True,
            collection_name=self.config.chroma_collection,
            embedding_model=self.config.embedding_model,
        )

        # Mock OpenAI client for deterministic embedding and chat completions
        self.mock_openai = MagicMock()

        def mock_embed_create(*args, **kwargs):
            inputs = kwargs.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            mock_data = []
            for idx, text in enumerate(inputs):
                mock_item = MagicMock()
                val = 0.05 * (idx + 1)
                mock_item.embedding = [val] * 384
                mock_data.append(mock_item)
            res = MagicMock()
            res.data = mock_data
            return res

        self.mock_openai.embeddings.create.side_effect = mock_embed_create

        # Mock Guardrail for query endpoint
        self.mock_guardrail = MagicMock()
        self.mock_retriever = VectorRetriever(
            openai_client=self.mock_openai,
            vector_db=self.vdb,
            embedding_model=self.config.embedding_model,
        )
        self.mock_guardrail.retriever = self.mock_retriever

        # Create FastAPI app with injected test components
        self.app = create_app(
            config=self.config,
            guardrail=self.mock_guardrail,
            vector_db_manager=self.vdb,
        )
        self.app.state.openai_client = self.mock_openai
        self.client = TestClient(self.app)

    def tearDown(self):
        """Clean up temporary directories."""
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Task 1 & 2: Successful Upload, Storage, Ingestion, and Indexing
    # -------------------------------------------------------------------------

    def test_successful_upload_and_indexing(self):
        """Verify valid document upload stores safely, chunks, embeds, and indexes into ChromaDB."""
        filename = "rbi_lcr_direction_2024.txt"
        file_bytes = SAMPLE_REGULATORY_DOC_TEXT.encode("utf-8")

        response = self.client.post(
            "/api/v1/upload",
            files={"file": (filename, file_bytes, "text/plain")},
            data={"chunk_size": 300, "chunk_overlap": 50},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], filename)
        self.assertEqual(data["file_type"], ".txt")
        self.assertEqual(data["file_size_bytes"], len(file_bytes))
        self.assertGreater(data["raw_character_count"], 0)
        self.assertGreater(data["cleaned_character_count"], 0)
        self.assertGreater(data["chunks_created"], 0)
        self.assertEqual(data["records_indexed"], data["chunks_created"])
        self.assertEqual(data["collection_name"], self.config.chroma_collection)
        self.assertEqual(data["total_collection_records"], data["chunks_created"])
        self.assertTrue(data["searchable_immediately"])
        self.assertEqual(len(data["chunk_ids"]), data["chunks_created"])

        # Confirm file exists on disk in upload directory
        saved_file = self.upload_dir / filename
        self.assertTrue(saved_file.exists())
        self.assertEqual(saved_file.read_bytes(), file_bytes)

        # Confirm records exist in ChromaDB collection
        col = self.vdb.get_or_create_collection(self.config.chroma_collection)
        self.assertEqual(col.count(), data["chunks_created"])

    def test_upload_alias_route(self):
        """Verify convenience alias /upload functions identically to /api/v1/upload."""
        filename = "alias_test_circular.txt"
        file_bytes = b"Sample regulatory content for alias upload route testing."

        response = self.client.post(
            "/upload",
            files={"file": (filename, file_bytes, "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["filename"], filename)

    # -------------------------------------------------------------------------
    # Task 3: Runtime Searchability Without Restart
    # -------------------------------------------------------------------------

    def test_runtime_searchability_without_restart(self):
        """Verify newly uploaded document is immediately searchable and citable in the same app instance."""
        filename = "rbi_lcr_direction_2024.txt"
        file_bytes = SAMPLE_REGULATORY_DOC_TEXT.encode("utf-8")

        # Step 1: Upload document
        up_res = self.client.post(
            "/api/v1/upload",
            files={"file": (filename, file_bytes, "text/plain")},
        )
        self.assertEqual(up_res.status_code, 200)
        up_data = up_res.json()
        indexed_chunk_id = up_data["chunk_ids"][0]

        # Step 2: Configure mock guardrail execution to simulate grounded response citing uploaded doc
        mock_output = CitedAnswerOutput(
            query="What are the LCR requirements?",
            answer="Covered institutions must maintain a minimum Liquidity Coverage Ratio of 100% [1].",
            citation_registry={
                "[1]": {
                    "source_document": filename,
                    "chunk_id": indexed_chunk_id,
                    "section": "2. Minimum LCR Standards",
                    "page_number": 1,
                    "similarity_score": 0.892,
                    "verbatim_text": "All covered institutions must maintain a minimum Liquidity Coverage Ratio of 100%.",
                }
            },
            audit_report=MagicMock(unique_markers_cited=["[1]"]),
            is_fallback=False,
            has_fabricated_citations=False,
            prompt_tokens=150,
            completion_tokens=40,
            latency_seconds=0.45,
            model="llama3:latest",
        )

        self.mock_guardrail.execute.return_value = GuardrailExecutionResult(
            query="What are the LCR requirements?",
            action="ANSWER",
            answer=mock_output.answer,
            quality_assessment=None,
            is_refusal=False,
            citations=["[1]"],
            cited_output=mock_output,
            latency_seconds=0.45,
            model="llama3:latest",
        )

        # Step 3: Query the same running application WITHOUT restarting
        query_res = self.client.post(
            "/api/v1/query",
            json={"question": "What are the LCR requirements?", "top_k": 3},
        )

        self.assertEqual(query_res.status_code, 200)
        q_data = query_res.json()

        self.assertEqual(q_data["status"], "success")
        self.assertIn("[1]", q_data["answer"])
        self.assertEqual(len(q_data["sources"]), 1)
        self.assertEqual(q_data["sources"][0]["source_document"], filename)
        self.assertEqual(q_data["sources"][0]["chunk_id"], indexed_chunk_id)
        self.assertAlmostEqual(q_data["sources"][0]["similarity_score"], 0.892, places=3)
        self.assertEqual(q_data["citations"], ["[1]"])

    # -------------------------------------------------------------------------
    # Task 4: Error Handling Matrix
    # -------------------------------------------------------------------------

    def test_reject_unsupported_file_extension(self):
        """Verify unsupported formats (.exe, .zip, .py, .bin) are rejected with HTTP 400 and UNSUPPORTED_FORMAT."""
        unsupported_files = [
            ("malicious_executable.exe", b"MZ\x90\x00BinaryExeContent"),
            ("archive_data.zip", b"PK\x03\x04ZipFileContent"),
            ("exploit_script.py", b"import os; os.system('echo hacked')"),
            ("raw_firmware.bin", b"\x00\x01\x02\x03\x04\x05\x06"),
        ]

        for fname, fbytes in unsupported_files:
            response = self.client.post(
                "/api/v1/upload",
                files={"file": (fname, fbytes, "application/octet-stream")},
            )
            self.assertEqual(response.status_code, 400, f"Expected 400 for {fname}")
            data = response.json()
            self.assertEqual(data["status"], "error")
            self.assertEqual(data["error_code"], "UNSUPPORTED_FORMAT")
            self.assertIn("unsupported file format", data["message"].lower())

    def test_reject_empty_zero_byte_file(self):
        """Verify 0-byte empty files are rejected with HTTP 400 and EMPTY_FILE."""
        for empty_ext in [".txt", ".md", ".pdf", ".html"]:
            fname = f"empty_document{empty_ext}"
            response = self.client.post(
                "/api/v1/upload",
                files={"file": (fname, b"", "text/plain")},
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertEqual(data["status"], "error")
            self.assertEqual(data["error_code"], "EMPTY_FILE")
            self.assertIn("empty", data["message"].lower())

    def test_reject_oversized_file(self):
        """Verify files exceeding max_upload_size_bytes are rejected with HTTP 413 and FILE_TOO_LARGE."""
        giant_payload = b"A" * (1024 * 1024 + 500)
        response = self.client.post(
            "/api/v1/upload",
            files={"file": ("giant_document.txt", giant_payload, "text/plain")},
        )
        self.assertEqual(response.status_code, 413)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], "FILE_TOO_LARGE")
        self.assertIn("exceeds maximum limit", data["message"].lower())

    def test_filename_sanitization_prevents_directory_traversal(self):
        """Verify directory traversal filenames (../../evil.txt) are sanitized safely without escaping upload_dir."""
        malicious_filename = "../../evil_path_traversal.txt"
        file_bytes = b"Harmless content inside directory traversal filename attempt."

        response = self.client.post(
            "/api/v1/upload",
            files={"file": (malicious_filename, file_bytes, "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["filename"], "evil_path_traversal.txt")

        expected_path = self.upload_dir / "evil_path_traversal.txt"
        self.assertTrue(expected_path.exists())

        escaped_path = self.upload_dir.parent / "evil_path_traversal.txt"
        self.assertFalse(escaped_path.exists())

    def test_reject_empty_or_whitespace_filename(self):
        """Verify empty or whitespace filename is rejected with HTTP 400 or 422."""
        for bad_name in ["", "   ", "\t"]:
            response = self.client.post(
                "/api/v1/upload",
                files={"file": (bad_name, b"some content", "text/plain")},
            )
            self.assertIn(response.status_code, [400, 422])
            data = response.json()
            self.assertIn(data["error_code"], ["INVALID_FILENAME", "VALIDATION_ERROR"])

    def test_custom_chunking_parameters_respected(self):
        """Verify chunk_size and chunk_overlap query/form parameters are respected during chunking."""
        filename = "custom_sizing_circular.txt"
        content = (SAMPLE_REGULATORY_DOC_TEXT + "\n") * 5
        response = self.client.post(
            "/api/v1/upload",
            files={"file": (filename, content.encode("utf-8"), "text/plain")},
            data={"chunk_size": 100, "chunk_overlap": 20},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["metadata"]["chunk_size"], 100)
        self.assertEqual(data["metadata"]["chunk_overlap"], 20)

    def test_invalid_chunking_parameters_rejected(self):
        """Verify invalid chunk sizes (<50 or overlap >= size) are rejected with HTTP 400."""
        # chunk_size too small
        res1 = self.client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", b"Valid regulatory content.", "text/plain")},
            data={"chunk_size": 20},
        )
        self.assertEqual(res1.status_code, 400)

        # chunk_overlap >= chunk_size
        res2 = self.client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", b"Valid regulatory content.", "text/plain")},
            data={"chunk_size": 200, "chunk_overlap": 250},
        )
        self.assertEqual(res2.status_code, 400)


class TestUploadArtifactExport(unittest.TestCase):
    """Test suite for Task 5: Sample Upload Run Artifacts."""

    def test_export_sample_upload_artifacts(self):
        """Verify export_sample_upload_artifacts creates valid JSON and Markdown files."""
        with tempfile.TemporaryDirectory() as tmp_out:
            out_path = Path(tmp_out)

            mock_guard = MagicMock()
            mock_guard.execute.return_value = GuardrailExecutionResult(
                query="What is the minimum Liquidity Coverage Ratio?",
                action="ANSWER",
                answer="Covered commercial banks must maintain a minimum LCR of 100% [1].",
                quality_assessment=MagicMock(status="SUFFICIENT_CONTEXT", top_score=0.88, qualifying_chunks=[]),
                is_refusal=False,
                citations=["[1]"],
                cited_output=None,
                latency_seconds=0.1,
                model="llama3:latest",
            )
            mock_vdb = MagicMock()
            mock_vdb.is_reachable.return_value = True

            test_cfg = AppConfig(
                upload_dir=out_path / "uploads",
                chroma_collection="test_artifacts_col",
                embedding_model="all-minilm",
                chat_model="llama3:latest",
                app_env="test",
            )
            test_app = create_app(
                config=test_cfg,
                guardrail=mock_guard,
                vector_db_manager=mock_vdb,
            )

            up_req, up_res, qr_res, report = export_sample_upload_artifacts(
                app_instance=test_app,
                output_dir=out_path,
            )

            self.assertTrue(up_req.exists())
            self.assertTrue(up_res.exists())
            self.assertTrue(qr_res.exists())
            self.assertTrue(report.exists())

            # Validate upload request JSON
            req_data = json.loads(up_req.read_text(encoding="utf-8"))
            self.assertEqual(req_data["endpoint"], "POST /api/v1/upload")
            self.assertIn("filename", req_data)

            # Validate upload response JSON
            res_data = json.loads(up_res.read_text(encoding="utf-8"))
            self.assertEqual(res_data["status"], "success")
            self.assertIn("chunks_created", res_data)
            self.assertIn("records_indexed", res_data)

            # Validate searchability query response JSON
            query_data = json.loads(qr_res.read_text(encoding="utf-8"))
            self.assertEqual(query_data["status"], "success")
            self.assertIn("sources", query_data)

            # Validate Markdown report
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("# RegulSense Runtime Document Upload & Indexing Report", report_text)
            self.assertIn("Runtime Searchability Confirmation", report_text)


if __name__ == "__main__":
    unittest.main()
