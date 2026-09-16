"""Script to execute realistic query workload, generate sample audit logs, and export usage summaries.

Implements Tasks 4 & 5:
- Runs query scenarios (Cache MISS, Cache HIT, Guardrail Refusal, Validation Error, Streaming HIT).
- Generates outputs/sample_query_audit.log (JSON Lines structured logs).
- Generates outputs/usage_summary_report.json (Structured metrics over time).
- Generates outputs/usage_summary_report.md (Executive and engineering Markdown report).
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import RegulSenseAPIClient
from src.audit_logger import AuditLogger
from src.citation_engine import CitationAuditReport, CitedAnswerOutput
from src.query_cache import QueryCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ExportUsageSummary")


def make_mock_cited_output(query: str) -> CitedAnswerOutput:
    """Creates realistic cited output for regulatory compliance questions."""
    if "cyber" in query.lower():
        answer = (
            "Under the RBI Master Direction on Cyber Resilience and Digital Payment Security Controls [1], "
            "regulated entities must report all Severity 1 cybersecurity incidents to CERT-In and the Reserve Bank "
            "within a mandatory timeframe of 6 hours of detection [2]."
        )
        citations = {
            "[1]": {
                "source_document": "cyber_resilience_framework.pdf",
                "chunk_id": "cyber_resilience_001",
                "section": "2. Incident Reporting Timelines",
                "page_number": 1,
                "similarity_score": 0.7420,
                "verbatim_text": "Regulated entities must notify CERT-In within 6 hours of cybersecurity incident detection.",
            },
            "[2]": {
                "source_document": "circular_dor_2024_108.txt",
                "chunk_id": "circular_dor_108_003",
                "section": "Reporting Obligations",
                "page_number": 2,
                "similarity_score": 0.6890,
                "verbatim_text": "Severity 1 events mandate immediate RBI notification.",
            },
        }
        prompt_tokens = 940
        completion_tokens = 68
    else:
        answer = (
            "In accordance with Rule 5 of the Prevention of Money-Laundering (Maintenance of Records) Rules [1], "
            "regulated banking entities must maintain records of all cash transactions and suspicious activities "
            "for a minimum retention period of five (5) years from the date of cessation of business relations [2]."
        )
        citations = {
            "[1]": {
                "source_document": "aml_kyc_directions_2023.pdf",
                "chunk_id": "aml_kyc_012",
                "section": "5. Record Retention Obligations",
                "page_number": 4,
                "similarity_score": 0.7610,
                "verbatim_text": "Maintain customer identification data and transaction records for at least five years.",
            },
            "[2]": {
                "source_document": "pmla_statutory_framework.txt",
                "chunk_id": "pmla_rule_5_001",
                "section": "Rule 5 Retention Period",
                "page_number": 1,
                "similarity_score": 0.7180,
                "verbatim_text": "Preserve transaction registers for five years following account closure.",
            },
        }
        prompt_tokens = 812
        completion_tokens = 62

    return CitedAnswerOutput(
        query=query,
        answer=answer,
        citation_registry=citations,
        audit_report=CitationAuditReport(
            total_citations_found=2,
            unique_markers_cited=["[1]", "[2]"],
            verified_citations_count=2,
            fabricated_citations_count=0,
            unsupported_claims_count=0,
            citation_precision=1.0,
            claim_verifications=[],
            fabricated_markers=[],
            overall_verdict="PASS",
            audit_notes="All claims fully grounded in verified regulatory text.",
        ),
        is_fallback=False,
        has_fabricated_citations=False,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_seconds=0.285,
        model="llama3:latest",
    )


def run_workload_and_export(output_dir: Path):
    """Executes representative queries across miss/hit states, records audit logs, and exports summaries."""
    logger.info("Starting Caching & Logging Workload Execution...")

    audit_log_path = output_dir / "sample_query_audit.log"
    if audit_log_path.exists():
        audit_log_path.unlink()

    from fastapi.testclient import TestClient
    from src.api import app, QueryCache, AuditLogger

    # Configure application state with dedicated audit logger and fresh query cache
    sample_audit_logger = AuditLogger(log_file_path=audit_log_path)
    sample_query_cache = QueryCache(max_size=100, default_ttl_seconds=3600.0)
    app.state.audit_logger = sample_audit_logger
    app.state.query_cache = sample_query_cache

    # Ensure deterministic and fast generation without blocking on external CPU model
    original_generate = getattr(app.state.guardrail.citation_engine, "generate_cited_answer", None)
    app.state.guardrail.citation_engine.generate_cited_answer = lambda query, chunks, **kwargs: make_mock_cited_output(query)

    client = TestClient(app)

    # Predefined representative inquiries
    q_cyber = (
        "What are the mandatory timeframe and reporting procedures for banks to notify "
        "CERT-In and RBI regarding Severity 1 cyber security incidents?"
    )
    q_retention = "What are the record retention obligations under AML and PML Rules?"
    q_refusal = "What are the capital reserve requirements for commercial banks operating on Mars?"

    # 1. Cold Grounded Query (Cache MISS)
    logger.info("Executing Query 1 (Cold - Expect Cache MISS)...")
    resp1 = client.post("/api/v1/query", json={"question": q_cyber, "top_k": 3, "include_metadata": True})
    data1 = resp1.json()
    logger.info("Query 1 result: Cache=%s, Latency=%.4fs", data1.get("metadata", {}).get("cache_hit"), data1.get("metadata", {}).get("latency_seconds", 0))

    # 2. Warm Grounded Query (Cache HIT)
    logger.info("Executing Query 1 Repeat (Warm - Expect Cache HIT)...")
    resp2 = client.post("/api/v1/query", json={"question": q_cyber, "top_k": 3, "include_metadata": True})
    data2 = resp2.json()
    logger.info("Query 1 Repeat result: Cache=%s, Latency=%.4fs", data2.get("metadata", {}).get("cache_hit"), data2.get("metadata", {}).get("latency_seconds", 0))

    # 3. Distinct Cold Grounded Query (Cache MISS)
    logger.info("Executing Query 2 (Cold - Expect Cache MISS)...")
    resp3 = client.post("/api/v1/query", json={"question": q_retention, "top_k": 3, "include_metadata": True})
    data3 = resp3.json()
    logger.info("Query 2 result: Cache=%s, Latency=%.4fs", data3.get("metadata", {}).get("cache_hit"), data3.get("metadata", {}).get("latency_seconds", 0))

    # 4. Repeat Distinct Grounded Query (Cache HIT)
    logger.info("Executing Query 2 Repeat (Warm - Expect Cache HIT)...")
    resp4 = client.post("/api/v1/query", json={"question": q_retention, "top_k": 3, "include_metadata": True})
    data4 = resp4.json()
    logger.info("Query 2 Repeat result: Cache=%s, Latency=%.4fs", data4.get("metadata", {}).get("cache_hit"), data4.get("metadata", {}).get("latency_seconds", 0))

    # 5. Out-of-Corpus Refusal Query (Cache MISS)
    logger.info("Executing Query 3 (Refusal - Expect Cache MISS)...")
    resp5 = client.post("/api/v1/query", json={"question": q_refusal, "top_k": 3, "include_metadata": True})
    data5 = resp5.json()
    logger.info("Query 3 result: Status=%s, Cache=%s", data5.get("status"), data5.get("metadata", {}).get("cache_hit"))

    # 6. Repeat Refusal Query (Cache HIT)
    logger.info("Executing Query 3 Repeat (Refusal - Expect Cache HIT)...")
    resp6 = client.post("/api/v1/query", json={"question": q_refusal, "top_k": 3, "include_metadata": True})
    data6 = resp6.json()
    logger.info("Query 3 Repeat result: Status=%s, Cache=%s", data6.get("status"), data6.get("metadata", {}).get("cache_hit"))

    # 7. Streaming Query (Cache HIT)
    logger.info("Executing Query 1 Streaming (Expect Streaming Cache HIT)...")
    resp_stream = client.post("/api/v1/query/stream", json={"question": q_cyber, "top_k": 3})
    logger.info("Streaming query result status: %s", resp_stream.status_code)

    # 8. Validation Error Query (Task 4 error tracking)
    logger.info("Executing Query Validation Error Case ('hi')...")
    resp_err = client.post("/api/v1/query", json={"question": "hi"})
    logger.info("Validation query result status: %s", resp_err.status_code)

    # Restore generator if modified
    if original_generate:
        app.state.guardrail.citation_engine.generate_cited_answer = original_generate

    # Fetch live usage summary and cache stats via API
    summary_resp = client.get("/api/v1/analytics/usage")
    cache_resp = client.get("/api/v1/analytics/cache")

    summary_data = summary_resp.json() if summary_resp.status_code == 200 else {}
    cache_data = cache_resp.json() if cache_resp.status_code == 200 else {}

    logger.info("Workload Execution Complete! Summary: Total=%s, Hits=%s, Hit Rate=%s%%",
                summary_data.get("total_requests"),
                summary_data.get("cache_hits"),
                summary_data.get("cache_hit_rate_pct"))

    # Export usage_summary_report.json (Task 4 & 5)
    summary_json_path = output_dir / "usage_summary_report.json"
    full_export_data = {
        "export_metadata": {
            "title": "RegulSense Query Caching, Structured Logging & Usage Analytics Summary",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "production-simulation",
            "tokenizer": "tiktoken (cl100k_base)",
        },
        "usage_metrics": summary_data,
        "cache_metrics": cache_data,
    }
    summary_json_path.write_text(json.dumps(full_export_data, indent=2), encoding="utf-8")
    logger.info("Saved usage summary JSON to: %s", summary_json_path)

    # Export usage_summary_report.md (Task 4 & 5)
    md_report = sample_audit_logger.generate_markdown_report()
    summary_md_path = output_dir / "usage_summary_report.md"
    summary_md_path.write_text(md_report, encoding="utf-8")
    logger.info("Saved usage summary Markdown report to: %s", summary_md_path)


def main():
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_workload_and_export(outputs_dir)


if __name__ == "__main__":
    main()

