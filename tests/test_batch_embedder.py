"""Unit tests for RegulSense Scalable Batch Embedding Pipeline.

Validates:
1. Task 1: Batching text chunks with configurable batch sizes (env, param, default).
2. Task 2: Retry mechanism with exponential backoff on transient errors and rate limits,
   and recording failed batches in the run summary.
3. Task 3: Reporting totals (chunks, generated, skipped, failures) and approximate USD cost.
4. Task 4: Detecting and skipping already-embedded chunks on re-runs to avoid duplicate calls.
5. Task 5: Persistence and structural validity of batch embedding run summaries (Markdown & JSON).
"""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from openai import APIConnectionError, InternalServerError, RateLimitError

from src.batch_embedder import (
    BatchEmbedder,
    BatchFailure,
    BatchRunSummary,
    run_batch_embedding_pipeline,
)
from src.corpus_embedder import EmbeddedChunkRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_mock_embedding_response(texts, dim=384):
    """Helper to create mock OpenAI embeddings response."""
    data = []
    for i, t in enumerate(texts):
        mock_item = MagicMock()
        # Deterministic dummy vector based on index
        mock_item.embedding = [float((i + 1) * 0.001 * (d + 1)) for d in range(dim)]
        data.append(mock_item)
    response = MagicMock()
    response.data = data
    return response


class TestBatchEmbedder(unittest.TestCase):
    """Test suite for batch embedding, backoff retries, cost tracking, and caching."""

    def setUp(self):
        """Set up sample corpus chunks for batch tests."""
        self.sample_chunks = [
            {
                "chunk_id": f"chunk_aml_{i:03d}",
                "content": f"Banking regulatory circular section {i} regarding customer due diligence and AML compliance rules.",
                "metadata": {
                    "filename": "circular_dor_2024_108.txt",
                    "document_id": "circular_dor_2024_108",
                    "section": f"Section {i}",
                    "page_number": 1,
                    "chunk_index": i,
                },
            }
            for i in range(15)
        ]

    # -------------------------------------------------------------------------
    # Task 1: Embed chunks in batches
    # -------------------------------------------------------------------------

    def test_task1_chunks_processed_in_configurable_batches(self):
        """Verify chunks are embedded in batches according to batch_size."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda model, input: create_mock_embedding_response(input, dim=384)

        embedder = BatchEmbedder(
            model="all-minilm",
            batch_size=5,
            client=mock_client,
            initial_backoff=0.01,
        )

        records, summary = embedder.embed_chunks_batched(
            chunks=self.sample_chunks,
            batch_size=5,
            force_refresh=True,
        )

        self.assertEqual(len(records), 15)
        self.assertEqual(summary.total_chunks, 15)
        self.assertEqual(summary.embeddings_generated, 15)
        self.assertEqual(summary.total_batches, 3)
        self.assertEqual(summary.successful_batches, 3)
        self.assertEqual(summary.failed_batches, 0)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)

        # Verify batch size passed to client
        first_call_input = mock_client.embeddings.create.call_args_list[0][1]["input"]
        self.assertEqual(len(first_call_input), 5)

    def test_task1_batch_size_configuration_precedence(self):
        """Verify batch size configuration precedence: param > env > default."""
        mock_client = MagicMock()

        # 1. Custom param
        embedder1 = BatchEmbedder(model="all-minilm", batch_size=8, client=mock_client)
        self.assertEqual(embedder1.batch_size, 8)

        # 2. Environment variable
        orig_env = os.environ.get("EMBEDDING_BATCH_SIZE")
        try:
            os.environ["EMBEDDING_BATCH_SIZE"] = "12"
            embedder2 = BatchEmbedder(model="all-minilm", client=mock_client)
            self.assertEqual(embedder2.batch_size, 12)
        finally:
            if orig_env is not None:
                os.environ["EMBEDDING_BATCH_SIZE"] = orig_env
            else:
                os.environ.pop("EMBEDDING_BATCH_SIZE", None)

        # 3. Default fallback
        embedder3 = BatchEmbedder(model="all-minilm", client=mock_client)
        self.assertEqual(embedder3.batch_size, 10)

    # -------------------------------------------------------------------------
    # Task 2: Retry with backoff & failure tracking
    # -------------------------------------------------------------------------

    def test_task2_transient_error_retries_and_succeeds(self):
        """Verify transient error triggers retry with backoff and successfully recovers."""
        mock_client = MagicMock()

        # Simulate 2 RateLimitErrors followed by a successful response
        dummy_req = MagicMock()
        rate_err = RateLimitError("Rate limit exceeded. Please retry.", response=dummy_req, body=None)
        success_resp = create_mock_embedding_response(["text1", "text2"])

        mock_client.embeddings.create.side_effect = [
            rate_err,
            rate_err,
            success_resp,
        ]

        retry_delays = []

        def on_retry_callback(attempt, delay, exc):
            retry_delays.append(delay)

        embedder = BatchEmbedder(
            model="all-minilm",
            batch_size=2,
            max_retries=3,
            initial_backoff=0.01,
            backoff_factor=2.0,
            client=mock_client,
        )

        test_chunks = self.sample_chunks[:2]
        records, summary = embedder.embed_chunks_batched(
            chunks=test_chunks,
            on_retry=on_retry_callback,
            force_refresh=True,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(summary.total_retry_attempts, 2)
        self.assertEqual(summary.successful_batches, 1)
        self.assertEqual(summary.failed_batches, 0)
        self.assertEqual(len(retry_delays), 2)
        # Verify exponential growth: delay 1 = 0.01 * 2^0 = 0.01; delay 2 = 0.01 * 2^1 = 0.02
        self.assertAlmostEqual(retry_delays[0], 0.01, places=3)
        self.assertAlmostEqual(retry_delays[1], 0.02, places=3)

    def test_task2_failed_batch_recorded_in_summary(self):
        """Verify unrecoverable error after max retries is recorded in failures ledger."""
        mock_client = MagicMock()
        dummy_req = MagicMock()
        conn_err = APIConnectionError(request=dummy_req, message="Connection timed out")
        mock_client.embeddings.create.side_effect = conn_err

        embedder = BatchEmbedder(
            model="all-minilm",
            batch_size=5,
            max_retries=2,
            initial_backoff=0.001,
            client=mock_client,
        )

        test_chunks = self.sample_chunks[:5]
        records, summary = embedder.embed_chunks_batched(
            chunks=test_chunks,
            continue_on_failure=True,
            force_refresh=True,
        )

        self.assertEqual(len(records), 0)
        self.assertEqual(summary.failed_batches, 1)
        self.assertEqual(summary.failed_chunks, 5)
        self.assertEqual(len(summary.failures), 1)

        failure = summary.failures[0]
        self.assertEqual(failure.batch_index, 1)
        self.assertEqual(len(failure.chunk_ids), 5)
        self.assertEqual(failure.error_type, "APIConnectionError")
        self.assertIn("Connection timed out", failure.error_message)
        self.assertEqual(failure.attempts, 3)  # initial + 2 retries

    def test_task2_partial_batch_failure_visibility(self):
        """Verify mixed batch outcomes: batch 1 succeeds, batch 2 fails and is visible in summary."""
        mock_client = MagicMock()
        dummy_req = MagicMock()

        resp_batch1 = create_mock_embedding_response(["text"] * 5)
        err_batch2 = InternalServerError("Server error", response=dummy_req, body=None)

        # Batch 1 succeeds on 1st try; Batch 2 fails all retries
        mock_client.embeddings.create.side_effect = [
            resp_batch1,
            err_batch2,
            err_batch2,
        ]

        embedder = BatchEmbedder(
            model="all-minilm",
            batch_size=5,
            max_retries=1,
            initial_backoff=0.001,
            client=mock_client,
        )

        test_chunks = self.sample_chunks[:10]
        records, summary = embedder.embed_chunks_batched(
            chunks=test_chunks,
            continue_on_failure=True,
            force_refresh=True,
        )

        self.assertEqual(len(records), 5)
        self.assertEqual(summary.successful_batches, 1)
        self.assertEqual(summary.failed_batches, 1)
        self.assertEqual(summary.failed_chunks, 5)
        self.assertEqual(len(summary.failures), 1)
        self.assertIn("PARTIAL FAILURE", summary.status)

    # -------------------------------------------------------------------------
    # Task 3: Report totals and approximate cost
    # -------------------------------------------------------------------------

    def test_task3_totals_and_cost_estimation_accuracy(self):
        """Verify accurate calculation of tokens and approximate USD cost."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda model, input: create_mock_embedding_response(input)

        rate_per_million = 0.02  # $0.02 per 1M tokens
        embedder = BatchEmbedder(
            model="text-embedding-3-small",
            batch_size=10,
            cost_per_million=rate_per_million,
            client=mock_client,
        )

        records, summary = embedder.embed_chunks_batched(
            chunks=self.sample_chunks[:6],
            force_refresh=True,
        )

        self.assertEqual(summary.total_chunks, 6)
        self.assertEqual(summary.embeddings_generated, 6)
        self.assertEqual(summary.skipped_chunks, 0)
        self.assertEqual(summary.failed_chunks, 0)
        self.assertGreater(summary.total_tokens_processed, 50)

        expected_cost = round((summary.total_tokens_processed / 1_000_000) * rate_per_million, 6)
        self.assertEqual(summary.embedding_cost_usd, expected_cost)
        self.assertEqual(summary.cost_per_million_tokens, rate_per_million)
        self.assertIn("text-embedding-3-small", summary.pricing_model_name)

    # -------------------------------------------------------------------------
    # Task 4: Skip already-embedded chunks on re-runs
    # -------------------------------------------------------------------------

    def test_task4_skips_already_embedded_chunks_on_rerun(self):
        """Verify on re-runs, chunks with existing embeddings are skipped without calling API."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda model, input: create_mock_embedding_response(input)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "embedded_corpus.json"
            summary_md = Path(temp_dir) / "summary.md"
            summary_json = Path(temp_dir) / "summary.json"

            embedder = BatchEmbedder(
                model="all-minilm",
                batch_size=5,
                client=mock_client,
            )

            # First Run: fresh embedding
            records1, summary1 = embedder.embed_chunks_batched(
                chunks=self.sample_chunks[:10],
                cache_path=cache_file,
                force_refresh=False,
            )
            embedder.export_run_artifacts(
                records=records1,
                summary=summary1,
                output_corpus_json=cache_file,
                output_summary_json=summary_json,
                output_summary_markdown=summary_md,
            )

            self.assertEqual(summary1.embeddings_generated, 10)
            self.assertEqual(summary1.skipped_chunks, 0)
            self.assertEqual(mock_client.embeddings.create.call_count, 2)

            # Reset call count
            mock_client.embeddings.create.reset_mock()

            # Second Run: identical corpus chunks
            records2, summary2 = embedder.embed_chunks_batched(
                chunks=self.sample_chunks[:10],
                cache_path=cache_file,
                force_refresh=False,
            )

            # ZERO API calls should have been made!
            self.assertEqual(mock_client.embeddings.create.call_count, 0)
            self.assertEqual(summary2.embeddings_generated, 0)
            self.assertEqual(summary2.skipped_chunks, 10)
            self.assertEqual(summary2.embedding_cost_usd, 0.0)
            self.assertGreater(summary2.cost_saved_usd, 0.0)
            self.assertEqual(len(records2), 10)

    def test_task4_partial_cache_skips_only_existing(self):
        """Verify when only some chunks are cached, only new chunks are embedded."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda model, input: create_mock_embedding_response(input)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "embedded_corpus.json"
            sum_json = Path(temp_dir) / "sum.json"
            sum_md = Path(temp_dir) / "sum.md"

            embedder = BatchEmbedder(
                model="all-minilm",
                batch_size=5,
                client=mock_client,
            )

            # Embed first 5 chunks
            rec_initial, sum_initial = embedder.embed_chunks_batched(
                chunks=self.sample_chunks[:5],
                cache_path=cache_file,
                force_refresh=True,
            )
            embedder.export_run_artifacts(
                records=rec_initial,
                summary=sum_initial,
                output_corpus_json=cache_file,
                output_summary_json=sum_json,
                output_summary_markdown=sum_md,
            )

            mock_client.embeddings.create.reset_mock()

            # Run with all 10 chunks (first 5 cached, next 5 new)
            records_mixed, summary_mixed = embedder.embed_chunks_batched(
                chunks=self.sample_chunks[:10],
                batch_size=5,
                cache_path=cache_file,
                force_refresh=False,
            )

            self.assertEqual(summary_mixed.skipped_chunks, 5)
            self.assertEqual(summary_mixed.embeddings_generated, 5)
            self.assertEqual(len(records_mixed), 10)
            # Only 1 API call for the 5 new chunks
            self.assertEqual(mock_client.embeddings.create.call_count, 1)

    def test_task4_force_refresh_bypasses_cache(self):
        """Verify force_refresh=True re-embeds all chunks even if cached."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda model, input: create_mock_embedding_response(input)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "embedded_corpus.json"

            embedder = BatchEmbedder(model="all-minilm", batch_size=5, client=mock_client)
            rec, sm = embedder.embed_chunks_batched(
                chunks=self.sample_chunks[:5],
                cache_path=cache_file,
                force_refresh=True,
            )
            sum_json = Path(temp_dir) / "sum.json"
            sum_md = Path(temp_dir) / "sum.md"
            embedder.export_run_artifacts(
                records=rec,
                summary=sm,
                output_corpus_json=cache_file,
                output_summary_json=sum_json,
                output_summary_markdown=sum_md,
            )

            mock_client.embeddings.create.reset_mock()

            # Re-run with force_refresh=True
            rec2, sm2 = embedder.embed_chunks_batched(
                chunks=self.sample_chunks[:5],
                cache_path=cache_file,
                force_refresh=True,
            )

            self.assertEqual(sm2.skipped_chunks, 0)
            self.assertEqual(sm2.embeddings_generated, 5)
            self.assertEqual(mock_client.embeddings.create.call_count, 1)

    # -------------------------------------------------------------------------
    # Task 5: Export run summary & markdown formatting
    # -------------------------------------------------------------------------

    def test_task5_run_summary_export_and_content(self):
        """Verify Markdown and JSON run summary outputs are generated with all required sections."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda model, input: create_mock_embedding_response(input)

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_json = Path(temp_dir) / "corpus.json"
            summary_json = Path(temp_dir) / "summary.json"
            summary_md = Path(temp_dir) / "summary.md"

            embedder = BatchEmbedder(model="all-minilm", batch_size=4, client=mock_client)
            records, summary = embedder.embed_chunks_batched(
                chunks=self.sample_chunks[:8],
                force_refresh=True,
            )

            embedder.export_run_artifacts(
                records=records,
                summary=summary,
                output_corpus_json=corpus_json,
                output_summary_json=summary_json,
                output_summary_markdown=summary_md,
            )

            self.assertTrue(corpus_json.exists())
            self.assertTrue(summary_json.exists())
            self.assertTrue(summary_md.exists())

            # Validate summary JSON
            summary_dict = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary_dict["total_chunks"], 8)
            self.assertEqual(summary_dict["embeddings_generated"], 8)
            self.assertEqual(summary_dict["skipped_chunks"], 0)
            self.assertEqual(summary_dict["failed_chunks"], 0)
            self.assertIn("embedding_cost_usd", summary_dict)
            self.assertIn("pricing_model_name", summary_dict)

            # Validate Markdown report content
            md_text = summary_md.read_text(encoding="utf-8")
            self.assertIn("Batch Embedding Pipeline Run Summary", md_text)
            self.assertIn("Executive Operations Ledger", md_text)
            self.assertIn("Token Volume & Financial Cost Analysis", md_text)
            self.assertIn("Resilience & Backoff Retry Audit", md_text)
            self.assertIn("Total Corpus Chunks", md_text)
            self.assertIn("Configured Batch Size", md_text)
            self.assertIn("Embeddings Generated", md_text)
            self.assertIn("Skipped Chunks (Cached)", md_text)
            self.assertIn("Approximate Run Cost", md_text)


if __name__ == "__main__":
    unittest.main()
