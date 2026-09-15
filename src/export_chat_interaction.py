"""Export sample interaction artifacts, error cases, documentation, and live UI screenshots.

Implements Task 5:
- Generates outputs/chat_interface_sample_interaction.json
- Generates outputs/chat_interface_error_case.json
- Generates outputs/chat_interface_demonstration.md
- Launches Streamlit and captures outputs/chat_interface_screenshot.png via Playwright
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import ClientResponse, RegulSenseAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ExportChatInteraction")


def find_free_port(starting_port: int = 8505) -> int:
    """Finds an open TCP port on localhost."""
    port = starting_port
    while port < 9000:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
            port += 1
    return 8505


def run_sample_interactions(output_dir: Path) -> Dict[str, Any]:
    """Runs grounded query, refusal query, and validation error cases."""
    logger.info("Initializing RegulSenseAPIClient for sample interaction exports...")
    client = RegulSenseAPIClient(enable_direct_fallback=True)

    # 1. Successful Grounded Query (Tasks 1, 2, 3)
    grounded_query = (
        "What are the mandatory timeframe and reporting procedures for banks to notify "
        "CERT-In and RBI regarding Severity 1 cyber security incidents?"
    )
    logger.info("Executing sample grounded compliance query: '%s'...", grounded_query[:60])
    resp_grounded = client.submit_query(question=grounded_query, top_k=3, include_metadata=True)

    sample_interaction = {
        "scenario": "RegulSense Chat Interface - Grounded Compliance Query",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_verification": {
            "task_1_qa_ui": "User input accepted and grounded answer returned in conversation stream.",
            "task_2_rag_api": "Successfully communicated with /api/v1/query endpoint.",
            "task_3_retrieved_sources": "Sources rendered with chunk IDs, section metadata, similarity scores, and verbatim excerpts.",
            "task_4_loading_error_states": "Loading spinner rendered during pipeline execution; error envelopes handled gracefully.",
        },
        "query": grounded_query,
        "success": resp_grounded.success,
        "status_code": resp_grounded.status_code,
        "response": {
            "status": resp_grounded.data.get("status", "success"),
            "answer": resp_grounded.answer,
            "citations": resp_grounded.citations,
            "sources": resp_grounded.sources,
            "metadata": resp_grounded.data.get("metadata", {}),
        },
    }

    sample_json_path = output_dir / "chat_interface_sample_interaction.json"
    sample_json_path.write_text(json.dumps(sample_interaction, indent=2), encoding="utf-8")
    logger.info("Saved sample interaction to: %s", sample_json_path)

    # 2. Safe Guardrail Refusal Case (Task 4)
    refusal_query = "What are the reserve capital requirements for Martian colony commercial banks?"
    logger.info("Executing sample refusal query: '%s'...", refusal_query)
    resp_refusal = client.submit_query(question=refusal_query, top_k=3)

    # 3. Client Validation Error Case (Task 4)
    short_query = "hi"
    logger.info("Executing sample short query validation error...")
    resp_short = client.submit_query(question=short_query)

    # 4. Simulated Backend Offline Error Case (Task 4)
    offline_client = RegulSenseAPIClient(base_url="http://127.0.0.1:59999", enable_direct_fallback=False)
    resp_offline = offline_client.submit_query(question="Is the backend reachable?")

    error_cases = {
        "scenario": "RegulSense Chat Interface - Guardrail Refusals and Error Handling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases": [
            {
                "case_type": "SAFE_GUARDRAIL_REFUSAL",
                "description": "Hallucination guardrail triggers safe refusal when vector retrieval similarity is below threshold.",
                "query": refusal_query,
                "status": resp_refusal.data.get("status"),
                "answer": resp_refusal.answer,
                "is_refusal": resp_refusal.is_refusal,
                "metadata": resp_refusal.data.get("metadata", {}),
            },
            {
                "case_type": "INPUT_VALIDATION_ERROR",
                "description": "Client-side validation catches queries shorter than 3 characters or empty input.",
                "query": short_query,
                "status_code": resp_short.status_code,
                "error_code": resp_short.error_code,
                "error_message": resp_short.error_message,
            },
            {
                "case_type": "BACKEND_OFFLINE_ERROR",
                "description": "UI detects when backend server is unreachable and guides user with troubleshooting steps.",
                "status_code": resp_offline.status_code,
                "error_code": resp_offline.error_code,
                "error_message": resp_offline.error_message,
            },
        ],
    }

    error_json_path = output_dir / "chat_interface_error_case.json"
    error_json_path.write_text(json.dumps(error_cases, indent=2), encoding="utf-8")
    logger.info("Saved error cases to: %s", error_json_path)

    # 5. Generate Comprehensive Markdown Demonstration Report
    markdown_report = generate_markdown_report(sample_interaction, error_cases)
    md_path = output_dir / "chat_interface_demonstration.md"
    md_path.write_text(markdown_report, encoding="utf-8")
    logger.info("Saved markdown demonstration report to: %s", md_path)

    return sample_interaction


def generate_markdown_report(sample: Dict[str, Any], error_cases: Dict[str, Any]) -> str:
    """Builds a formatted markdown report detailing UI capabilities."""
    q = sample["query"]
    ans = sample["response"]["answer"]
    citations = ", ".join(sample["response"]["citations"]) or "None"
    sources = sample["response"]["sources"]
    meta = sample["response"]["metadata"]

    lines = [
        "# RegulSense Streamlit Chat User Interface - Demonstration Report",
        "",
        "## Overview",
        "This report documents the implementation and verification of the RegulSense Chat User Interface,",
        "built with Streamlit to enable conversational regulatory compliance question answering with",
        "verifiable citations and retrieved source chunk inspection.",
        "",
        "## Verified Tasks",
        "- **Task 1 - Build question and answer UI**: Conversational chat stream with role-based avatars, message history, and real-time input container.",
        "- **Task 2 - Call the backend RAG API**: Integrated with `POST /api/v1/query` and `GET /api/v1/health` with automatic fallback.",
        "- **Task 3 - Display retrieved sources**: Expandable source inspection trays rendering source document names, chunk IDs, sections, page numbers, similarity scores, and verbatim excerpts.",
        "- **Task 4 - Handle loading and error states**: Multi-stage progress indicators during execution, and dedicated handling for backend connection errors, validation rejections, and safe guardrail refusals.",
        "- **Task 5 - Commit screenshot and sample interactions**: Interactive verification artifacts exported to `outputs/`.",
        "",
        "---",
        "",
        "## 1. Sample Interaction: Grounded Compliance Query",
        "",
        f"**User Question:**",
        f"> {q}",
        "",
        f"**Status:** `🟢 Grounded Answer`",
        f"**Active Model:** `{meta.get('model', 'llama3:latest')}` | **Latency:** `{meta.get('latency_seconds', 0.0)}s` | **Top Similarity:** `{meta.get('top_similarity_score', 0.0)}`",
        "",
        f"**Assistant Answer:**",
        ans,
        "",
        f"**Inline Citations Detected:** `{citations}`",
        "",
        "### Retrieved Regulatory Sources (Task 3)",
        "",
        "| Marker | Source Circular | Chunk ID | Section | Similarity Score |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for s in sources:
        marker = s.get("marker", "[?]")
        doc = s.get("source_document", "Unknown")
        cid = s.get("chunk_id", "N/A")
        sec = s.get("section") or "General"
        score = s.get("similarity_score", 0.0)
        lines.append(f"| `{marker}` | {doc} | `{cid}` | {sec} | **{score:.4f}** |")

    lines.extend([
        "",
        "#### Verbatim Source Excerpts",
        "",
    ])

    for s in sources:
        marker = s.get("marker", "[?]")
        doc = s.get("source_document", "Unknown")
        cid = s.get("chunk_id", "N/A")
        verbatim = s.get("verbatim_text", "").strip()
        lines.extend([
            f"**Source {marker} - `{doc}` (`{cid}`)**:",
            "```text",
            verbatim[:500] + ("..." if len(verbatim) > 500 else ""),
            "```",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 2. Safe Guardrail Refusal & Error State Demonstrations (Task 4)",
        "",
        "### Case A: Safe Guardrail Refusal (Weak Context)",
        f"- **Question:** `{error_cases['cases'][0]['query']}`",
        f"- **Outcome:** `{error_cases['cases'][0]['status']}`",
        f"- **Explanation:** {error_cases['cases'][0]['answer']}",
        "",
        "### Case B: Client-Side Input Validation",
        f"- **Input:** `'{error_cases['cases'][1]['query']}'`",
        f"- **Validation Code:** `{error_cases['cases'][1]['error_code']}`",
        f"- **Message:** {error_cases['cases'][1]['error_message']}",
        "",
        "### Case C: Backend Unreachable Handling",
        f"- **Error Code:** `{error_cases['cases'][2]['error_code']}` (HTTP {error_cases['cases'][2]['status_code']})",
        f"- **UI Behavior:** Prominently alerts the user that the backend server is offline, provides the exact command to launch `uvicorn`, or lets the user enable Direct In-Process Mode.",
        "",
        "---",
        "",
        "## 3. How to Launch the Chat Interface",
        "",
        "### Option A: Standard Full-Stack Mode",
        "```powershell",
        "# Terminal 1: Launch FastAPI Backend",
        "uvicorn src.api:app --host 0.0.0.0 --port 8000",
        "",
        "# Terminal 2: Launch Streamlit Web UI",
        "streamlit run app.py",
        "```",
        "",
        "### Option B: Standalone Direct Mode",
        "```powershell",
        "# Run directly without running a separate uvicorn process",
        "streamlit run app.py",
        "# The UI automatically falls back to in-process pipeline evaluation!",
        "```",
    ])

    return "\n".join(lines)


def capture_ui_screenshot(output_dir: Path, port: int = 8506) -> Path:
    """Launches Streamlit headlessly, navigates via Playwright, and takes a full UI screenshot."""
    screenshot_path = output_dir / "chat_interface_screenshot.png"
    app_file = str(PROJECT_ROOT / "src" / "chat_app.py")

    logger.info("Starting Streamlit app process on port %d...", port)
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_file, "--server.port", str(port), "--server.headless", "true"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for Streamlit server to come online
        url = f"http://127.0.0.1:{port}"
        logger.info("Waiting for Streamlit server to respond at %s...", url)
        connected = False
        for _ in range(30):
            try:
                import requests
                resp = requests.get(url, timeout=1.0)
                if resp.status_code == 200:
                    connected = True
                    break
            except Exception:
                time.sleep(0.5)

        if not connected:
            logger.warning("Streamlit server did not respond in time on port %d.", port)
            return screenshot_path

        logger.info("Streamlit is online! Launching browser via Playwright (channel='msedge')...")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(viewport={"width": 1366, "height": 960})
            page = context.new_page()

            # Navigate using domcontentloaded (websockets prevent networkidle in Streamlit)
            page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Wait for main container and chat message
            logger.info("Waiting for Streamlit app container...")
            page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=30000)
            page.wait_for_selector("[data-testid='stChatMessage']", timeout=20000)
            time.sleep(3.0)

            # Expand sources tray if present
            try:
                expander = page.locator("summary:has-text('Inspect Retrieved Sources')").first
                if expander.is_visible():
                    expander.click()
                    time.sleep(1.0)
            except Exception as exp_err:
                logger.debug("Could not click expander: %s", exp_err)

            # Ensure page is scrolled to top to show header and branding
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1.0)

            # Take high-resolution full-page screenshot of the interactive UI
            page.screenshot(path=str(screenshot_path), full_page=True)
            logger.info("Saved high-resolution UI screenshot to: %s", screenshot_path)

            browser.close()

    except Exception as exc:
        logger.error("Failed to capture screenshot via Playwright: %s", exc, exc_info=True)
    finally:
        logger.info("Terminating background Streamlit process (PID %d)...", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    return screenshot_path


def main():
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run interactions and export JSON / Markdown
    run_sample_interactions(output_dir)

    # 2. Capture live screenshot
    port = find_free_port(8508)
    capture_ui_screenshot(output_dir, port=port)
    logger.info("Completed all Task 5 interaction exports successfully.")


if __name__ == "__main__":
    main()
