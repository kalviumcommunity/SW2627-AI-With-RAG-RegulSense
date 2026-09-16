"""Unit and integration tests for Query Caching, Structured Logging, and Usage Analytics.

Tests:
1. QueryCache normalization, key generation, hit/miss tracking, LRU eviction, and TTL expiration.
2. AuditLogger record creation, JSON Lines formatting, token counting, and asymmetric cost modeling.
3. Usage summary aggregation and Markdown report generation over time.
4. FastAPI endpoints: /api/v1/query and /api/v1/query/stream cache hit behavior.
5. FastAPI analytics endpoints: /api/v1/analytics/usage, /api/v1/analytics/cache, and cache clear.
6. RegulSenseAPIClient analytics and cache control methods.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import create_app
from src.api_client import ClientResponse, RegulSenseAPIClient
from src.audit_logger import AuditLogger, AuditLogRecord
from src.hallucination_guardrails import GuardrailExecutionResult, HallucinationGuardrail
from src.query_cache import CachedQueryEntry, QueryCache


class TestQueryCache(unittest.TestCase):
    """Unit tests for QueryCache data structure, key normalization, and eviction."""

    def setUp(self):
        self.cache = QueryCache(max_size=3, default_ttl_seconds=1.5)

    def test_query_normalization(self):
        """Verifies query normalization removes extraneous whitespace and folds case."""
        q1 = "   What are RBI   reporting rules?  "
        q2 = "what are rbi reporting rules?"
        norm1 = QueryCache.normalize_query(q1)
        norm2 = QueryCache.normalize_query(q2)
        self.assertEqual(norm1, norm2)

        # Keys should be identical
        key1 = QueryCache.generate_cache_key(q1, top_k=3)
        key2 = QueryCache.generate_cache_key(q2, top_k=3)
        self.assertEqual(key1, key2)

    def test_key_sensitivity_to_settings(self):
        """Verifies cache key distinguishes between different top_k values or models."""
        k1 = QueryCache.generate_cache_key("Same question", top_k=3, model="llama3:latest")
        k2 = QueryCache.generate_cache_key("Same question", top_k=5, model="llama3:latest")
        k3 = QueryCache.generate_cache_key("Same question", top_k=3, model="gpt-4o")

        self.assertNotEqual(k1, k2)
        self.assertNotEqual(k1, k3)

    def test_cache_put_and_get_hit(self):
        """Verifies storing and retrieving an entry increments hit counters."""
        entry = self.cache.put(
            query="What is 6-hour rule?",
            top_k=3,
            answer="Notify CERT-In in 6 hours [1].",
            citations=["[1]"],
            sources=[{"marker": "[1]", "source_document": "rbi.pdf"}],
            prompt_tokens=100,
            completion_tokens=20,
            estimated_cost_usd=0.000036,
            original_latency_seconds=0.45,
        )

        retrieved = self.cache.get(query="what is 6-hour rule?  ", top_k=3)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.answer, "Notify CERT-In in 6 hours [1].")
        self.assertEqual(retrieved.hit_count, 1)

        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["hit_rate_pct"], 100.0)

    def test_lru_eviction(self):
        """Verifies oldest accessed items are evicted when cache capacity is exceeded."""
        self.cache.put(query="Query 1", top_k=3, answer="A1", citations=[], sources=[])
        self.cache.put(query="Query 2", top_k=3, answer="A2", citations=[], sources=[])
        self.cache.put(query="Query 3", top_k=3, answer="A3", citations=[], sources=[])

        # Access Query 1 to make Query 2 the LRU candidate
        self.cache.get(query="Query 1", top_k=3)

        # Insert 4th item -> should evict Query 2
        self.cache.put(query="Query 4", top_k=3, answer="A4", citations=[], sources=[])

        self.assertIsNotNone(self.cache.get("Query 1", top_k=3))
        self.assertIsNone(self.cache.get("Query 2", top_k=3))  # Evicted!
        self.assertIsNotNone(self.cache.get("Query 3", top_k=3))
        self.assertIsNotNone(self.cache.get("Query 4", top_k=3))

    def test_ttl_expiration(self):
        """Verifies items expire after TTL seconds."""
        self.cache.put(query="Short lived", top_k=3, answer="Temporary", citations=[], sources=[], ttl_seconds=0.1)
        time.sleep(0.15)
        res = self.cache.get("Short lived", top_k=3)
        self.assertIsNone(res)

    def test_invalidate_and_clear(self):
        """Verifies specific invalidation and full cache wipe."""
        self.cache.put("Query A", top_k=3, answer="A", citations=[], sources=[])
        self.cache.put("Query B", top_k=3, answer="B", citations=[], sources=[])

        self.assertTrue(self.cache.invalidate("Query A", top_k=3))
        self.assertIsNone(self.cache.get("Query A", top_k=3))
        self.assertIsNotNone(self.cache.get("Query B", top_k=3))

        self.cache.clear()
        self.assertEqual(self.cache.get_stats()["current_size"], 0)


class TestAuditLogger(unittest.TestCase):
    """Unit tests for AuditLogger record schema, token cost estimation, and summarization."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "test_audit.jsonl"
        self.logger = AuditLogger(
            log_file_path=self.log_file,
            input_cost_per_m=0.20,
            output_cost_per_m=0.80,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_token_counting_and_cost_estimation(self):
        """Verifies accurate token counts and asymmetric cost calculation."""
        text = "Reserve Bank of India Master Direction"
        count = self.logger.count_tokens(text)
        self.assertGreater(count, 0)

        # 10,000 prompt tokens ($0.20/1M) = $0.0020
        # 5,000 completion tokens ($0.80/1M) = $0.0040 -> total = $0.0060
        cost = self.logger.calculate_cost(prompt_tokens=10_000, completion_tokens=5_000)
        self.assertAlmostEqual(cost, 0.0060, places=4)

    def test_log_request_persists_json_lines(self):
        """Verifies log_request generates structured JSON line and appends to file."""
        rec = self.logger.log_request(
            question="What is the UPI transaction limit?",
            answer="The standard limit is ₹1,00,000 per day [1].",
            sources=[{"marker": "[1]", "source_document": "upi_circular.txt", "similarity_score": 0.82}],
            citations=["[1]"],
            cache_hit=False,
            status="success",
            latency_seconds=0.35,
            prompt_tokens=150,
            completion_tokens=25,
            top_k=3,
        )

        self.assertIsInstance(rec, AuditLogRecord)
        self.assertEqual(rec.question, "What is the UPI transaction limit?")
        self.assertIn("[1]", rec.citations)
        self.assertFalse(rec.cache_hit)
        self.assertGreater(rec.estimated_cost_usd, 0.0)

        # Verify file content
        self.assertTrue(self.log_file.exists())
        lines = self.log_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["request_id"], rec.request_id)
        self.assertEqual(parsed["status"], "success")

    def test_cache_hit_cost_savings(self):
        """Verifies cache hits register $0.00 actual cost and positive cost_saved_usd."""
        rec_hit = self.logger.log_request(
            question="What is the UPI transaction limit?",
            answer="The standard limit is ₹1,00,000 per day [1].",
            cache_hit=True,
            status="success",
            latency_seconds=0.001,
            prompt_tokens=150,
            completion_tokens=25,
        )

        self.assertEqual(rec_hit.estimated_cost_usd, 0.0)
        self.assertGreater(rec_hit.cost_saved_usd, 0.0)

    def test_summarize_usage_and_markdown_report(self):
        """Verifies usage aggregation over time and markdown report rendering."""
        # Log 1 miss and 1 hit
        self.logger.log_request(
            question="Q1", answer="A1", cache_hit=False, status="success",
            latency_seconds=0.50, prompt_tokens=100, completion_tokens=50,
        )
        self.logger.log_request(
            question="Q1", answer="A1", cache_hit=True, status="success",
            latency_seconds=0.002, prompt_tokens=100, completion_tokens=50,
        )
        # Log 1 refusal
        self.logger.log_request(
            question="Out of corpus", answer="Unable to answer", cache_hit=False, status="refusal",
            latency_seconds=0.10, prompt_tokens=50, completion_tokens=10,
        )

        summary = self.logger.summarize_usage()
        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["cache_hits"], 1)
        self.assertEqual(summary["cache_misses"], 2)
        self.assertAlmostEqual(summary["cache_hit_rate_pct"], 33.33, places=1)
        self.assertGreater(summary["total_cost_saved_usd"], 0.0)
        self.assertIn("success", summary["status_breakdown"])
        self.assertIn("refusal", summary["status_breakdown"])

        md_report = self.logger.generate_markdown_report()
        self.assertIn("# RegulSense Query Caching, Structured Logging", md_report)
        self.assertIn("Cache Hit Rate", md_report)
        self.assertIn("Total Requests", md_report)


class TestBackendCachingEndpoints(unittest.TestCase):
    """Integration tests for FastAPI /api/v1/query and /api/v1/analytics endpoints with caching."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "test_api_audit.jsonl"
        self.query_cache = QueryCache(max_size=50, default_ttl_seconds=3600.0)
        self.audit_logger = AuditLogger(log_file_path=self.log_file)

        mock_guardrail = MagicMock(spec=HallucinationGuardrail)
        mock_res = GuardrailExecutionResult(
            query="What is the reporting rule?",
            action="ANSWER",
            answer="Report within 6 hours [1].",
            quality_assessment=None,
            is_refusal=False,
            citations=["[1]"],
            latency_seconds=0.30,
            model="llama3:latest",
        )
        mock_guardrail.execute.return_value = mock_res
        mock_guardrail.execute_stream.return_value = [
            ("sources", {"[1]": {"source_document": "rbi.txt", "chunk_id": "c1"}}, []),
            ("token", "Report within "),
            ("token", "6 hours [1]."),
            ("done", mock_res),
        ]

        self.app = create_app(
            guardrail=mock_guardrail,
            query_cache=self.query_cache,
            audit_logger=self.audit_logger,
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_query_cache_miss_then_hit(self):
        """Verifies first query call is a Cache MISS and identical second call is a Cache HIT."""
        payload = {"question": "What is the reporting rule?", "top_k": 3, "include_metadata": True}

        # First request -> Cache MISS
        resp1 = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertFalse(data1["metadata"]["cache_hit"])
        self.assertIn("Report within 6 hours [1].", data1["answer"])

        # Second request -> Cache HIT
        resp2 = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertTrue(data2["metadata"]["cache_hit"])
        self.assertEqual(data2["answer"], data1["answer"])
        self.assertEqual(data2["citations"], data1["citations"])
        self.assertEqual(data2["metadata"]["estimated_cost_usd"], 0.0)
        self.assertGreater(data2["metadata"]["cost_saved_usd"], 0.0)

        # Cache stats should reflect 1 hit, 1 miss
        stats = self.query_cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_query_stream_cache_hit(self):
        """Verifies streaming endpoint serves from cache when entry exists."""
        payload = {"question": "What is the reporting rule?", "top_k": 3}

        # Prime the cache first
        self.client.post("/api/v1/query", json=payload)

        # Streaming query should now be served from cache
        resp_stream = self.client.post("/api/v1/query/stream", json=payload)
        self.assertEqual(resp_stream.status_code, 200)
        self.assertEqual(resp_stream.headers.get("content-type"), "text/event-stream; charset=utf-8")

        lines = resp_stream.text.strip().split("\n\n")
        self.assertGreaterEqual(len(lines), 3)

        # Last event must be done with cache_hit: True
        last_evt = json.loads(lines[-1].replace("data: ", ""))
        self.assertEqual(last_evt["event"], "done")
        self.assertTrue(last_evt["data"]["metadata"]["cache_hit"])
        self.assertEqual(last_evt["data"]["metadata"]["estimated_cost_usd"], 0.0)

    def test_analytics_endpoints(self):
        """Verifies GET /api/v1/analytics/usage and /cache and POST /cache/clear."""
        # Make a request to generate data
        self.client.post("/api/v1/query", json={"question": "Test query for analytics", "top_k": 3})

        # GET usage analytics
        resp_usage = self.client.get("/api/v1/analytics/usage")
        self.assertEqual(resp_usage.status_code, 200)
        usage_data = resp_usage.json()
        self.assertEqual(usage_data["total_requests"], 1)
        self.assertIn("total_estimated_cost_usd", usage_data)

        # GET cache stats
        resp_cache = self.client.get("/api/v1/analytics/cache")
        self.assertEqual(resp_cache.status_code, 200)
        cache_data = resp_cache.json()
        self.assertEqual(cache_data["current_size"], 1)

        # POST cache clear
        resp_clear = self.client.post("/api/v1/analytics/cache/clear")
        self.assertEqual(resp_clear.status_code, 200)
        self.assertEqual(resp_clear.json()["status"], "success")

        # Verify cache is now empty
        resp_cache_after = self.client.get("/api/v1/analytics/cache")
        self.assertEqual(resp_cache_after.json()["current_size"], 0)


class TestAPIClientAnalytics(unittest.TestCase):
    """Unit tests for RegulSenseAPIClient analytics and cache methods."""

    def setUp(self):
        self.client = RegulSenseAPIClient(base_url="http://mock-api:8000", enable_direct_fallback=False)

    @patch("requests.get")
    def test_client_get_usage_summary(self, mock_get):
        """Verifies client parses usage analytics JSON."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total_requests": 10,
            "cache_hits": 4,
            "cache_hit_rate_pct": 40.0,
            "total_tokens": 1500,
        }
        mock_get.return_value = mock_resp

        res = self.client.get_usage_summary()
        self.assertTrue(res.success)
        self.assertEqual(res.data["total_requests"], 10)
        self.assertEqual(res.data["cache_hits"], 4)

    @patch("requests.post")
    def test_client_clear_cache(self, mock_post):
        """Verifies client triggers cache clear."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "success", "message": "Cleared"}
        mock_post.return_value = mock_resp

        res = self.client.clear_cache()
        self.assertTrue(res.success)
        self.assertEqual(res.data["status"], "success")


if __name__ == "__main__":
    unittest.main()
