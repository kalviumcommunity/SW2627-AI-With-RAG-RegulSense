"""Structured Audit Logger and Usage Tracking Engine for RegulSense.

Implements Tasks 2, 3, and 4:
- Logs requests, answers, citations, sources, cache outcomes, and errors in JSON Lines.
- Tracks exact token counts and asymmetric cost modeling per request.
- Aggregates usage metrics over time (total queries, hit rate, latency, cost savings).
- Exports formatted JSON summaries and Markdown demonstration reports.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

import tiktoken

logger = logging.getLogger("AuditLogger")

# Default asymmetric enterprise pricing (USD per 1M tokens)
DEFAULT_INPUT_COST_PER_M = 0.20   # $0.20 / 1M prompt tokens
DEFAULT_OUTPUT_COST_PER_M = 0.80  # $0.80 / 1M completion tokens


@dataclass
class AuditLogRecord:
    """Represents a structured audit record for a single compliance inquiry."""
    request_id: str
    timestamp: str
    question: str
    answer_preview: str
    full_answer: str
    sources: List[Dict[str, Any]]
    citations: List[str]
    cache_hit: bool
    status: str            # "success", "refusal", "error"
    error: Optional[str]
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cost_saved_usd: float
    top_k: int
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes record to JSON-compatible dictionary."""
        return asdict(self)

    def to_json_line(self) -> str:
        """Serializes record to a single-line JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditLogger:
    """Manages structured JSON Lines query logging, token estimation, and usage metrics."""

    def __init__(
        self,
        log_file_path: Optional[Path] = None,
        input_cost_per_m: float = DEFAULT_INPUT_COST_PER_M,
        output_cost_per_m: float = DEFAULT_OUTPUT_COST_PER_M,
        tokenizer_encoding: str = "cl100k_base",
    ):
        self.log_file_path = log_file_path
        self.input_cost_per_m = input_cost_per_m
        self.output_cost_per_m = output_cost_per_m

        # Tokenizer setup
        try:
            self.encoder = tiktoken.get_encoding(tokenizer_encoding)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self._lock = threading.RLock()
        self._records: List[AuditLogRecord] = []

        # If log file exists, optionally load previous records
        if self.log_file_path and self.log_file_path.exists():
            self._load_existing_logs()

        logger.info(
            "Initialized AuditLogger (log_path=%s, pricing=$%.2f/1M in, $%.2f/1M out)",
            self.log_file_path,
            self.input_cost_per_m,
            self.output_cost_per_m,
        )

    def count_tokens(self, text: str) -> int:
        """Counts tokens for text using cl100k_base tokenizer."""
        if not text:
            return 0
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # Fallback estimation: ~4 characters per token
            return max(1, len(text) // 4)

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculates total dollar cost for prompt and completion tokens."""
        prompt_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_m
        comp_cost = (completion_tokens / 1_000_000) * self.output_cost_per_m
        return round(prompt_cost + comp_cost, 6)

    def log_request(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        citations: Optional[List[str]] = None,
        cache_hit: bool = False,
        status: str = "success",
        error: Optional[str] = None,
        latency_seconds: float = 0.0,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        top_k: int = 3,
        model: str = "llama3:latest",
        request_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLogRecord:
        """Constructs, stores, and persists a structured audit log record (Task 2 & 3)."""
        clean_q = question.strip() if question else ""
        clean_ans = answer.strip() if answer else ""
        req_id = request_id or str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()

        # Generate answer preview (first 140 chars)
        ans_preview = clean_ans[:140] + ("..." if len(clean_ans) > 140 else "")

        # Sanitize sources
        sanitized_sources = []
        for idx, s in enumerate(sources or [], start=1):
            if isinstance(s, dict):
                sanitized_sources.append({
                    "marker": s.get("marker", f"[{idx}]"),
                    "source_document": s.get("source_document", "Unknown"),
                    "chunk_id": s.get("chunk_id", "N/A"),
                    "section": s.get("section", "General"),
                    "page_number": s.get("page_number", 1),
                    "similarity_score": round(float(s.get("similarity_score", 0.0)), 4),
                })

        # Calculate tokens if not supplied
        p_tok = prompt_tokens if prompt_tokens is not None else self.count_tokens(clean_q)
        c_tok = completion_tokens if completion_tokens is not None else self.count_tokens(clean_ans)
        tot_tok = p_tok + c_tok

        # Calculate cost & savings
        nominal_cost = self.calculate_cost(p_tok, c_tok)
        if cache_hit:
            actual_cost = 0.0
            saved_cost = nominal_cost
        else:
            actual_cost = nominal_cost
            saved_cost = 0.0

        record = AuditLogRecord(
            request_id=req_id,
            timestamp=ts,
            question=clean_q,
            answer_preview=ans_preview,
            full_answer=clean_ans,
            sources=sanitized_sources,
            citations=citations or [],
            cache_hit=cache_hit,
            status=status,
            error=error,
            latency_seconds=round(latency_seconds, 4),
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=tot_tok,
            estimated_cost_usd=actual_cost,
            cost_saved_usd=saved_cost,
            top_k=top_k,
            model=model,
            metadata=extra_metadata or {},
        )

        with self._lock:
            self._records.append(record)
            self._append_to_file(record)

        logger.info(
            "[AUDIT] [%s] Q='%s...' Cache=%s Status=%s Latency=%.4fs Tok=%d Cost=$%.5f",
            req_id[:8],
            clean_q[:35],
            "HIT" if cache_hit else "MISS",
            status,
            latency_seconds,
            tot_tok,
            actual_cost,
        )

        return record

    def _append_to_file(self, record: AuditLogRecord):
        """Appends record as a JSON line to disk."""
        if not self.log_file_path:
            return
        try:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(record.to_json_line() + "\n")
        except Exception as exc:
            logger.error("Failed to write audit record to %s: %s", self.log_file_path, exc)

    def _load_existing_logs(self):
        """Loads historical log lines from file."""
        if not self.log_file_path or not self.log_file_path.exists():
            return
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = AuditLogRecord(**data)
                        self._records.append(record)
                    except Exception:
                        continue
            logger.info("Loaded %d historical audit records from %s", len(self._records), self.log_file_path)
        except Exception as exc:
            logger.warning("Could not parse existing logs from %s: %s", self.log_file_path, exc)

    def get_records(self, limit: Optional[int] = None) -> List[AuditLogRecord]:
        """Returns recorded audit logs in chronological order."""
        with self._lock:
            if limit:
                return list(self._records[-limit:])
            return list(self._records)

    # -------------------------------------------------------------------------
    # Task 4: Summarize Usage Over Time
    # -------------------------------------------------------------------------

    def summarize_usage(self) -> Dict[str, Any]:
        """Calculates aggregated usage metrics over time (Task 4)."""
        with self._lock:
            records = list(self._records)

        total_reqs = len(records)
        if total_reqs == 0:
            return {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate_pct": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_estimated_cost_usd": 0.0,
                "total_cost_saved_usd": 0.0,
                "average_latency_seconds": 0.0,
                "average_cache_hit_latency_seconds": 0.0,
                "average_cache_miss_latency_seconds": 0.0,
                "status_breakdown": {"success": 0, "refusal": 0, "error": 0},
                "pricing_reference": {
                    "input_per_million_usd": self.input_cost_per_m,
                    "output_per_million_usd": self.output_cost_per_m,
                },
            }

        hits = sum(1 for r in records if r.cache_hit)
        misses = total_reqs - hits
        hit_rate = (hits / total_reqs) * 100.0

        p_tokens = sum(r.prompt_tokens for r in records)
        c_tokens = sum(r.completion_tokens for r in records)
        tot_tokens = p_tokens + c_tokens

        tot_cost = sum(r.estimated_cost_usd for r in records)
        tot_saved = sum(r.cost_saved_usd for r in records)

        avg_lat = sum(r.latency_seconds for r in records) / total_reqs
        hit_latencies = [r.latency_seconds for r in records if r.cache_hit]
        miss_latencies = [r.latency_seconds for r in records if not r.cache_hit]

        avg_hit_lat = (sum(hit_latencies) / len(hit_latencies)) if hit_latencies else 0.0
        avg_miss_lat = (sum(miss_latencies) / len(miss_latencies)) if miss_latencies else 0.0

        status_counts = {"success": 0, "refusal": 0, "error": 0}
        for r in records:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        # Calculate timeline buckets (hourly)
        timeline_buckets: Dict[str, Dict[str, Any]] = {}
        for r in records:
            hour_key = r.timestamp[:13] + ":00:00Z"
            if hour_key not in timeline_buckets:
                timeline_buckets[hour_key] = {
                    "requests": 0,
                    "hits": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "saved_usd": 0.0,
                }
            bucket = timeline_buckets[hour_key]
            bucket["requests"] += 1
            if r.cache_hit:
                bucket["hits"] += 1
            bucket["tokens"] += r.total_tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + r.estimated_cost_usd, 6)
            bucket["saved_usd"] = round(bucket["saved_usd"] + r.cost_saved_usd, 6)

        return {
            "summary_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_requests": total_reqs,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "total_prompt_tokens": p_tokens,
            "total_completion_tokens": c_tokens,
            "total_tokens": tot_tokens,
            "total_estimated_cost_usd": round(tot_cost, 6),
            "total_cost_saved_usd": round(tot_saved, 6),
            "average_latency_seconds": round(avg_lat, 4),
            "average_cache_hit_latency_seconds": round(avg_hit_lat, 4),
            "average_cache_miss_latency_seconds": round(avg_miss_lat, 4),
            "latency_reduction_pct": round(((avg_miss_lat - avg_hit_lat) / avg_miss_lat * 100.0) if avg_miss_lat > 0 else 0.0, 2),
            "status_breakdown": status_counts,
            "hourly_timeline": timeline_buckets,
            "pricing_reference": {
                "input_per_million_usd": self.input_cost_per_m,
                "output_per_million_usd": self.output_cost_per_m,
            },
        }

    def generate_markdown_report(self) -> str:
        """Generates a comprehensive Markdown usage summary report (Task 4)."""
        summary = self.summarize_usage()
        records = self.get_records()

        lines = [
            "# RegulSense Query Caching, Structured Logging & Usage Analytics Report",
            "",
            "## Executive Overview",
            "This report summarizes query traffic, cache efficiency, token consumption, and cost economics for the **RegulSense RAG Intelligence Assistant**.",
            "",
            "---",
            "",
            "## 1. Key Performance Indicators",
            "",
            "| Metric | Value | Description |",
            "| :--- | :---: | :--- |",
            f"| **Total Requests** | `{summary['total_requests']:,}` | Total compliance inquiries processed |",
            f"| **Cache Hits** | `{summary['cache_hits']:,}` | Inquiries served instantly from in-memory cache |",
            f"| **Cache Misses** | `{summary['cache_misses']:,}` | Inquiries requiring full vector retrieval & LLM generation |",
            f"| **Cache Hit Rate** | **`{summary['cache_hit_rate_pct']:.1f}%`** | Proportion of inquiries accelerated by cache |",
            f"| **Avg Cache Miss Latency** | `{summary['average_cache_miss_latency_seconds']:.4f}s` | Full RAG pipeline response time |",
            f"| **Avg Cache Hit Latency** | **`{summary['average_cache_hit_latency_seconds']:.4f}s`** | Instant in-memory cache lookup time |",
            f"| **Latency Reduction** | **`{summary['latency_reduction_pct']:.1f}%`** | Performance acceleration from query caching |",
            f"| **Total Tokens Consumed** | `{summary['total_tokens']:,}` | `{summary['total_prompt_tokens']:,}` prompt + `{summary['total_completion_tokens']:,}` completion |",
            f"| **Total Estimated Cost** | **`${summary['total_estimated_cost_usd']:.5f}`** | Actual LLM compute expenditure incurred |",
            f"| **Total Cost Saved** | **`${summary['total_cost_saved_usd']:.5f}`** | Financial savings achieved via cache hits |",
            "",
            "---",
            "",
            "## 2. Request Status Breakdown",
            "",
            "| Status | Count | Percentage |",
            "| :--- | :---: | :---: |",
        ]

        total = max(1, summary["total_requests"])
        for st_name, count in summary["status_breakdown"].items():
            pct = (count / total) * 100.0
            icon = "🟢" if st_name == "success" else ("🟠" if st_name == "refusal" else "🔴")
            lines.append(f"| {icon} **{st_name.title()}** | `{count}` | `{pct:.1f}%` |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Detailed Request Audit Log (Sample Records)",
            "",
            "| Req ID | Timestamp (UTC) | Query Preview | Cache | Status | Latency | Tokens | Cost | Saved |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for r in records[-15:]:
            q_short = r.question[:30] + ("..." if len(r.question) > 30 else "")
            cache_badge = "⚡ HIT" if r.cache_hit else "MISS"
            cost_str = f"${r.estimated_cost_usd:.5f}" if not r.cache_hit else "$0.00000"
            saved_str = f"${r.cost_saved_usd:.5f}" if r.cache_hit else "$0.00000"
            lines.append(
                f"| `{r.request_id[:8]}` | `{r.timestamp[11:19]}` | *{q_short}* | `{cache_badge}` | `{r.status}` | `{r.latency_seconds:.3f}s` | `{r.total_tokens}` | `{cost_str}` | `{saved_str}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Economic Pricing Model Reference",
            f"- **Input Prompt Rate**: `${summary['pricing_reference']['input_per_million_usd']:.2f} per 1M tokens`",
            f"- **Output Completion Rate**: `${summary['pricing_reference']['output_per_million_usd']:.2f} per 1M tokens`",
            "- **Cost Formula**: `Cost = (PromptTokens * InputRate + CompletionTokens * OutputRate) / 1,000,000`",
            "- **Cache Savings**: For every cache hit, full RAG LLM invocation is avoided, directly translating to **100% cost avoidance** for that inquiry.",
        ])

        return "\n".join(lines)
