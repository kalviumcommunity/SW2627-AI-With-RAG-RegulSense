"""Unit tests for RegulSense End-to-End Modular RAG Pipeline.

Validates:
1. Task 1 & Task 2: Each stage (embed, retrieve, assemble, generate, attribute) executes
   correctly with valid inputs and handles error cases.
2. Task 3: End-to-end pipeline execution returns grounded answer and attributed sources.
3. Task 4: Separation of responsibilities allows independent testing and mocking.
4. Task 5: Report generation and serialization integrity.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.rag_pipeline import (
    AssembledContext,
    GenerationResult,
    PipelineStageMetrics,
    QueryEmbedding,
    RAGPipeline,
    RAGResponse,
    RetrievedContextChunk,
    SourceCitation,
    assemble_context_stage,
    attribute_sources_stage,
    embed_query_stage,
    export_pipeline_artifacts,
    generate_answer_stage,
    generate_markdown_report,
    retrieve_chunks_stage,
)
from src.retriever import RetrievalRunResult, VectorRetriever
from src.vector_db import RetrievedRecord


class TestRAGPipelineStages(unittest.TestCase):
    """Tests each modular pipeline stage in isolation."""

    def setUp(self):
        """Prepares sample chunks and mock clients."""
        self.sample_chunks = [
            RetrievedContextChunk(
                chunk_id="cyber_resilience_chunk_001",
                source_document="cyber_resilience_framework.pdf",
                section="Section 4: Incident Reporting Timelines",
                page_number=3,
                similarity_score=0.8850,
                text="Banks must report Severity 1 cyber security incidents within 2 to 6 hours to CERT-In and RBI.",
                token_count=22,
            ),
            RetrievedContextChunk(
                chunk_id="circular_dor_chunk_004",
                source_document="circular_dor_2024_108.txt",
                section="Section 8: Record Retention",
                page_number=6,
                similarity_score=0.6500,
                text="Customer transaction records and account opening dossiers shall be preserved for 5 years.",
                token_count=18,
            ),
        ]

    # -------------------------------------------------------------------------
    # Stage 1: Embed Stage Tests
    # -------------------------------------------------------------------------

    def test_embed_query_stage_success(self):
        """Verifies embed_query_stage returns valid QueryEmbedding with 384 dimensions."""
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.05] * 384
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response

        res = embed_query_stage(
            query="What are cyber reporting timelines?",
            client=mock_client,
            model="all-minilm",
            dimension=384,
        )

        self.assertIsInstance(res, QueryEmbedding)
        self.assertEqual(res.dimension, 384)
        self.assertEqual(len(res.vector), 384)
        self.assertEqual(res.query_text, "What are cyber reporting timelines?")
        self.assertGreaterEqual(res.latency_seconds, 0.0)

    def test_embed_query_stage_empty_raises(self):
        """Verifies embed_query_stage raises ValueError on empty or whitespace queries."""
        with self.assertRaises(ValueError):
            embed_query_stage(query="")

        with self.assertRaises(ValueError):
            embed_query_stage(query="   \t\n  ")

    def test_embed_query_stage_dimension_mismatch(self):
        """Verifies embed_query_stage raises ValueError if embedding length does not match expected dimension."""
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 128  # 128 instead of 384
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response

        with self.assertRaises(ValueError):
            embed_query_stage(
                query="Test mismatch",
                client=mock_client,
                dimension=384,
            )

    # -------------------------------------------------------------------------
    # Stage 2: Retrieve Stage Tests
    # -------------------------------------------------------------------------

    def test_retrieve_chunks_stage_success(self):
        """Verifies retrieve_chunks_stage converts RetrievedRecord objects and respects top_k."""
        mock_retriever = MagicMock(spec=VectorRetriever)
        mock_records = [
            RetrievedRecord(
                rank=1,
                id="chunk_001",
                similarity_score=0.85,
                distance=0.15,
                document="Test document 1 content",
                metadata={"source_document": "doc1.txt", "section": "Sec 1", "page_number": 1, "token_count": 5},
            ),
            RetrievedRecord(
                rank=2,
                id="chunk_002",
                similarity_score=0.45,
                distance=0.55,
                document="Test document 2 content",
                metadata={"source_document": "doc2.txt", "section": "Sec 2", "page_number": 2, "token_count": 5},
            ),
        ]
        mock_retriever.retrieve.return_value = RetrievalRunResult(
            query_text="Sample query",
            k=2,
            retrieved_count=2,
            chunks=mock_records,
            top_score=0.85,
            lowest_score=0.45,
            mean_score=0.65,
            total_tokens_retrieved=10,
            source_documents=["doc1.txt", "doc2.txt"],
        )

        chunks, latency = retrieve_chunks_stage(
            query_text="Sample query",
            retriever=mock_retriever,
            top_k=2,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_id, "chunk_001")
        self.assertEqual(chunks[0].source_document, "doc1.txt")
        self.assertAlmostEqual(chunks[0].similarity_score, 0.85)
        self.assertGreaterEqual(latency, 0.0)

    def test_retrieve_chunks_stage_score_filtering(self):
        """Verifies retrieve_chunks_stage filters out chunks with similarity below min_score."""
        mock_retriever = MagicMock(spec=VectorRetriever)
        mock_records = [
            RetrievedRecord(
                rank=1,
                id="high_score_chunk",
                similarity_score=0.75,
                distance=0.25,
                document="High score content",
                metadata={"source_document": "doc1.txt"},
            ),
            RetrievedRecord(
                rank=2,
                id="low_score_chunk",
                similarity_score=0.35,
                distance=0.65,
                document="Low score content",
                metadata={"source_document": "doc2.txt"},
            ),
        ]
        mock_retriever.retrieve.return_value = RetrievalRunResult(
            query_text="Sample query",
            k=2,
            retrieved_count=2,
            chunks=mock_records,
            top_score=0.75,
            lowest_score=0.35,
            mean_score=0.55,
            total_tokens_retrieved=10,
            source_documents=["doc1.txt", "doc2.txt"],
        )

        chunks, _ = retrieve_chunks_stage(
            query_text="Sample query",
            retriever=mock_retriever,
            min_score=0.50,  # Should drop chunk 2
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "high_score_chunk")

    # -------------------------------------------------------------------------
    # Stage 3: Assemble Stage Tests
    # -------------------------------------------------------------------------

    def test_assemble_context_stage_formatting(self):
        """Verifies assemble_context_stage formats context with provenance markers and chat messages."""
        assembled = assemble_context_stage(
            query="When must cyber incidents be reported?",
            chunks=self.sample_chunks,
            max_context_tokens=1000,
        )

        self.assertIsInstance(assembled, AssembledContext)
        self.assertEqual(len(assembled.messages), 2)
        self.assertEqual(assembled.messages[0]["role"], "system")
        self.assertEqual(assembled.messages[1]["role"], "user")
        self.assertIn("cyber_resilience_framework.pdf", assembled.raw_context_text)
        self.assertIn("Incident Reporting Timelines", assembled.raw_context_text)
        self.assertIn("When must cyber incidents be reported?", assembled.user_prompt)
        self.assertFalse(assembled.was_truncated)
        self.assertEqual(len(assembled.included_chunks), 2)

    def test_assemble_context_stage_empty_chunks_fallback(self):
        """Verifies assemble_context_stage injects anti-hallucination fallback when chunks is empty."""
        assembled = assemble_context_stage(
            query="Basel III CET1 capital adequacy query",
            chunks=[],
        )

        self.assertEqual(len(assembled.included_chunks), 0)
        self.assertIn("NO DIRECT REGULATORY EVIDENCE FOUND", assembled.raw_context_text)
        self.assertIn("out-of-scope", assembled.raw_context_text.lower())

    def test_assemble_context_stage_token_budget_truncation(self):
        """Verifies assemble_context_stage enforces max_context_tokens truncation."""
        long_chunks = [
            RetrievedContextChunk(
                chunk_id=f"chunk_{i}",
                source_document=f"doc_{i}.pdf",
                section="Section 1",
                page_number=1,
                similarity_score=0.8,
                text="Word " * 200,  # ~200 tokens each
                token_count=200,
            )
            for i in range(5)
        ]

        assembled = assemble_context_stage(
            query="Test query",
            chunks=long_chunks,
            max_context_tokens=150,  # Only enough for 1 block
        )

        self.assertTrue(assembled.was_truncated)
        self.assertEqual(len(assembled.included_chunks), 1)

    # -------------------------------------------------------------------------
    # Stage 4: Generate Stage Tests
    # -------------------------------------------------------------------------

    def test_generate_answer_stage(self):
        """Verifies generate_answer_stage invokes LLM chat completions with parameters."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Severity 1 incidents must be reported within 6 hours."
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 150
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 175
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are RegulSense."},
            {"role": "user", "content": "Report timeline question"},
        ]

        result = generate_answer_stage(
            messages=messages,
            client=mock_client,
            model="llama3:latest",
            temperature=0.2,
            max_tokens=500,
        )

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.answer_text, "Severity 1 incidents must be reported within 6 hours.")
        self.assertEqual(result.prompt_tokens, 150)
        self.assertEqual(result.completion_tokens, 25)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.model, "llama3:latest")
        mock_client.chat.completions.create.assert_called_once_with(
            model="llama3:latest",
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )

    # -------------------------------------------------------------------------
    # Stage 5: Attribute Sources Stage Tests
    # -------------------------------------------------------------------------

    def test_attribute_sources_stage(self):
        """Verifies attribute_sources_stage extracts structured citations with trimmed excerpts."""
        citations = attribute_sources_stage(self.sample_chunks, excerpt_length=40)

        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].chunk_id, "cyber_resilience_chunk_001")
        self.assertEqual(citations[0].source_document, "cyber_resilience_framework.pdf")
        self.assertAlmostEqual(citations[0].similarity_score, 0.8850)
        self.assertTrue(citations[0].excerpt.endswith("..."))
        self.assertLessEqual(len(citations[0].excerpt), 45)


class TestRAGPipelineEndToEnd(unittest.TestCase):
    """Tests the RAGPipeline orchestrator end-to-end."""

    def test_pipeline_run_mocked(self):
        """Runs the entire pipeline end-to-end with mocked clients and verifies all stage contracts."""
        mock_client = MagicMock()

        # Mock embedding response
        mock_embed_data = MagicMock()
        mock_embed_data.embedding = [0.01] * 384
        mock_embed_resp = MagicMock()
        mock_embed_resp.data = [mock_embed_data]
        mock_client.embeddings.create.return_value = mock_embed_resp

        # Mock LLM generation response
        mock_choice = MagicMock()
        mock_choice.message.content = "Under RBI Cyber Resilience guidelines, Severity 1 incidents must be reported within 2 to 6 hours."
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 220
        mock_usage.completion_tokens = 35
        mock_usage.total_tokens = 255
        mock_gen_resp = MagicMock()
        mock_gen_resp.choices = [mock_choice]
        mock_gen_resp.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_gen_resp

        # Mock VectorRetriever
        mock_retriever = MagicMock(spec=VectorRetriever)
        mock_records = [
            RetrievedRecord(
                rank=1,
                id="cyber_001",
                similarity_score=0.89,
                distance=0.11,
                document="Mandatory 6-hour reporting window for Severity 1 incidents.",
                metadata={"source_document": "cyber_resilience.pdf", "section": "Reporting", "page_number": 2},
            )
        ]
        mock_retriever.retrieve.return_value = RetrievalRunResult(
            query_text="What are reporting timelines?",
            k=1,
            retrieved_count=1,
            chunks=mock_records,
            top_score=0.89,
            lowest_score=0.89,
            mean_score=0.89,
            total_tokens_retrieved=12,
            source_documents=["cyber_resilience.pdf"],
        )

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            openai_client=mock_client,
            chat_model="llama3:latest",
            embedding_model="all-minilm",
            vector_dimension=384,
        )

        response = pipeline.run(
            query="What are reporting timelines?",
            top_k=1,
        )

        self.assertIsInstance(response, RAGResponse)
        self.assertIn("Severity 1", response.answer)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].source_document, "cyber_resilience.pdf")
        self.assertEqual(response.metrics.prompt_tokens, 220)
        self.assertEqual(response.metrics.completion_tokens, 35)
        self.assertGreater(response.metrics.total_latency_seconds, 0.0)

    def test_pipeline_serialization_and_markdown(self):
        """Verifies that RAGResponse serializes to JSON and generates markdown documentation."""
        metrics = PipelineStageMetrics(
            embed_latency_seconds=0.05,
            retrieve_latency_seconds=0.08,
            assemble_latency_seconds=0.01,
            generate_latency_seconds=0.75,
            total_latency_seconds=0.89,
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
        )
        chunk = RetrievedContextChunk(
            chunk_id="c1",
            source_document="doc.pdf",
            section="Sec A",
            page_number=1,
            similarity_score=0.92,
            text="Document excerpt",
            token_count=10,
        )
        assembled = AssembledContext(
            system_prompt="System prompt",
            user_prompt="User prompt",
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
            raw_context_text="Raw context",
            included_chunks=[chunk],
            total_context_tokens=30,
            was_truncated=False,
            latency_seconds=0.01,
        )
        gen = GenerationResult(
            answer_text="Test answer",
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
            finish_reason="stop",
            latency_seconds=0.75,
            model="llama3:latest",
        )
        response = RAGResponse(
            query="Sample query",
            answer="Test answer",
            citations=[SourceCitation("c1", "doc.pdf", "Sec A", 0.92, "Document excerpt")],
            metrics=metrics,
            assembled_context=assembled,
            raw_generation=gen,
        )

        # Test dictionary serialization
        data = response.to_dict()
        self.assertEqual(data["query"], "Sample query")
        self.assertEqual(data["answer"], "Test answer")
        self.assertEqual(len(data["citations"]), 1)

        # Test JSON round-trip
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        self.assertEqual(loaded["metrics"]["prompt_tokens"], 120)

        # Test Markdown generation
        md = generate_markdown_report(response)
        self.assertIn("# RegulSense: End-to-End RAG Pipeline Execution", md)
        self.assertIn("doc.pdf", md)
        self.assertIn("Stage 1: Embed", md)

        # Test artifact export
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_file, json_file = export_pipeline_artifacts(response, output_dir=tmp_dir)
            self.assertTrue(md_file.exists())
            self.assertTrue(json_file.exists())


if __name__ == "__main__":
    unittest.main()
