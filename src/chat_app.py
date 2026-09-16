"""RegulSense Regulatory Intelligence & Compliance Assistant - Streamlit Web Interface.

Implements:
- Task 1: Question and Answer UI (conversational chat stream, input container, message history).
- Task 2: Backend RAG API connection (/api/v1/query with answer rendering).
- Task 3: Retrieved sources inspection tray (source document, chunk ID, section, page, score, verbatim text).
- Task 4: Loading and error states (spinners, connection failures, validation errors, safe guardrail refusals).
- Task 5: Sample interaction support and runtime document upload.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import ClientResponse, RegulSenseAPIClient

# ==============================================================================
# Page Configuration & Rich Custom Styling
# ==============================================================================

st.set_page_config(
    page_title="RegulSense | Regulatory Intelligence RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Main container and font styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header banner */
    .regulsense-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        color: #f8fafc;
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .regulsense-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .regulsense-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-refusal {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-error {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-info {
        background: rgba(14, 165, 233, 0.15);
        color: #0ea5e9;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }

    /* Source item card */
    .source-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .source-meta {
        font-size: 12px;
        color: #475569;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 8px;
    }
    .source-verbatim {
        font-size: 12px;
        line-height: 1.55;
        background: #ffffff;
        padding: 10px 14px;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        color: #0f172a;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* Citation pill */
    .citation-pill {
        background: rgba(14, 165, 233, 0.2);
        color: #38bdf8;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }

    /* Diagnostics pill */
    .meta-pill {
        font-size: 11.5px;
        color: #64748b;
        margin-top: 8px;
        display: flex;
        gap: 14px;
        align-items: center;
        flex-wrap: wrap;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# Helper Functions & State Initialization
# ==============================================================================

def get_api_client() -> RegulSenseAPIClient:
    """Initializes and caches the API client."""
    base_url = st.session_state.get("backend_url", os.getenv("REGULSENSE_API_URL", "http://localhost:8000"))
    direct_fallback = st.session_state.get("enable_direct_fallback", True)
    return RegulSenseAPIClient(base_url=base_url, enable_direct_fallback=direct_fallback)


def init_session_state():
    """Initializes chat message history and default settings."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Welcome to **RegulSense Regulatory Intelligence**. Ask any banking compliance, "
                    "incident reporting, or regulatory circular question. Every answer includes verifiable "
                    "source citations and chunk inspection trays."
                ),
                "status": "system",
                "sources": [],
                "citations": [],
                "metadata": {},
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            },
            {
                "role": "user",
                "content": (
                    "What are the mandatory timeframe and reporting procedures for banks to notify "
                    "CERT-In and RBI regarding Severity 1 cyber security incidents?"
                ),
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            },
            {
                "role": "assistant",
                "content": (
                    "According to the **Master Direction on Cyber Resilience and Digital Payment Security Controls** [1], "
                    "all regulated commercial banks must report any cyber security incident, ransomware compromise, "
                    "unauthorized intrusion, or major denial of service (DoS) affecting customer-facing channels to the "
                    "**RBI Cyber Security Cell (CSITE)** and **CERT-In within 6 hours of detection** [1].\n\n"
                    "This mandatory obligation is designated as the **6-Hour Rule**. Initial notifications must be followed "
                    "by a comprehensive forensic analysis report within 7 business days [1]."
                ),
                "status": "success",
                "sources": [
                    {
                        "marker": "[1]",
                        "source_document": "cyber_resilience_framework.pdf",
                        "chunk_id": "cyber_resilience_framework_pdf_tokenaware_001",
                        "section": "Section 2: Incident Reporting Timelines (6-Hour Rule)",
                        "page_number": 1,
                        "similarity_score": 0.6436,
                        "verbatim_text": (
                            "RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI\n"
                            "Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25\n"
                            "Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls\n"
                            "2. Incident Reporting Timelines (6-Hour Rule)\n"
                            "Any cyber security incident, ransomware compromise, unauthorized system intrusion, or major denial of service (DoS) "
                            "affecting customer-facing channels must be reported to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection. "
                            "Initial reports must be followed by a comprehensive forensic analysis report within 7 business days."
                        ),
                    },
                    {
                        "marker": "[2]",
                        "source_document": "circular_dor_2024_108.txt",
                        "chunk_id": "circular_dor_2024_108_txt_tokenaware_003",
                        "section": "Section 4: Transaction Monitoring and Reporting",
                        "page_number": 1,
                        "similarity_score": 0.5709,
                        "verbatim_text": (
                            "Banks shall deploy rule-based and behavioral automated transaction monitoring systems to identify suspicious transaction patterns "
                            "and maintain comprehensive audit records."
                        ),
                    },
                ],
                "citations": ["[1]"],
                "metadata": {
                    "latency_seconds": 0.42,
                    "model": "llama3:latest",
                    "top_similarity_score": 0.6436,
                    "prompt_tokens": 1520,
                    "completion_tokens": 156,
                },
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            },
        ]
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = os.getenv("REGULSENSE_API_URL", "http://localhost:8000")
    if "enable_direct_fallback" not in st.session_state:
        st.session_state.enable_direct_fallback = True
    if "top_k" not in st.session_state:
        st.session_state.top_k = 3
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


init_session_state()
client = get_api_client()


# ==============================================================================
# Sidebar: Backend Connection, Retrieval Controls, and Document Upload
# ==============================================================================

with st.sidebar:
    st.markdown("### ⚙️ System Controls")

    # Backend Connection
    st.markdown("#### Backend API")
    backend_input = st.text_input(
        "API Service URL",
        value=st.session_state.backend_url,
        help="FastAPI backend URL (e.g. http://localhost:8000)",
        key="backend_url_input",
    )
    if backend_input != st.session_state.backend_url:
        st.session_state.backend_url = backend_input
        st.rerun()

    direct_mode_toggle = st.checkbox(
        "Direct Pipeline Fallback",
        value=st.session_state.enable_direct_fallback,
        help="If the standalone HTTP server is offline, fall back to the in-process RAG pipeline.",
    )
    st.session_state.enable_direct_fallback = direct_mode_toggle

    # Health check probe
    col_ping, col_status = st.columns([1, 1.2])
    with col_ping:
        ping_clicked = st.button("🔌 Ping Health", use_container_width=True)

    health_info = client.check_health(timeout=2.0)
    if health_info.success:
        h_data = health_info.data
        v_ok = h_data.get("vector_db_reachable", False)
        col_status.markdown(
            f"<span class='badge badge-success'>● Connected ({health_info.latency_seconds:.2f}s)</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"**ChromaDB**: {'Reachable' if v_ok else 'Unreachable'} (`{h_data.get('collection_name', 'default')}`) | "
            f"**Model**: `{h_data.get('chat_model', 'llama3:latest')}`"
        )
    else:
        col_status.markdown(
            f"<span class='badge badge-error'>● Offline</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"⚠️ {health_info.error_message or 'Server unreachable'}")

    st.divider()

    # Retrieval Tuning (Task 3)
    st.markdown("#### 🎯 Retrieval Parameters")
    top_k_val = st.slider(
        "Top-K Regulatory Chunks",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        help="Number of candidate regulatory circular chunks retrieved from vector database.",
    )
    st.session_state.top_k = top_k_val

    st.divider()

    # Quick sample compliance prompts
    st.markdown("#### 💡 Quick Compliance Queries")
    quick_queries = [
        "Severity 1 incident reporting timeline for CERT-In and RBI",
        "Customer compensation for delayed UPI transactions",
        "Data localization mandates under RBI circulars",
        "Cryptocurrency reserve requirements for lunar colony banks",
    ]
    for q in quick_queries:
        if st.button(f"📌 {q[:38]}...", help=q, use_container_width=True, key=f"quick_{hash(q)}"):
            st.session_state.pending_prompt = q
            st.rerun()

    st.divider()

    # Document Upload (Sprint integration)
    st.markdown("#### 📄 Ingest New Circular")
    with st.expander("Upload & Index Document", expanded=False):
        st.caption("Upload a regulatory document (.pdf, .txt, .md) for runtime indexing without restart.")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
            key="doc_uploader",
        )
        if uploaded_file is not None:
            if st.button("🚀 Upload & Index Now", use_container_width=True):
                with st.spinner(f"Indexing '{uploaded_file.name}' into ChromaDB..."):
                    bytes_data = uploaded_file.getvalue()
                    upload_res = client.upload_document(
                        file_bytes=bytes_data,
                        filename=uploaded_file.name,
                    )
                    if upload_res.success:
                        up_data = upload_res.data
                        st.success(
                            f"Indexed '{up_data.get('filename')}'! "
                            f"Created {up_data.get('chunks_created', 0)} chunks. Searchable immediately."
                        )
                    else:
                        st.error(f"Upload failed: {upload_res.error_message or upload_res.error_code}")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        init_session_state()
        st.rerun()


# ==============================================================================
# Main View: Header & Branding Banner
# ==============================================================================

st.markdown(
    """
    <div class="regulsense-header">
        <div class="regulsense-title">
            <span>⚖️</span> RegulSense | Regulatory Intelligence & Compliance Assistant
        </div>
        <p class="regulsense-subtitle">
            Enterprise Banking Compliance RAG • Verifiable Source Citations • Hallucination Guardrails
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Helper to Render Retrieved Sources (Task 3)
# ==============================================================================

def render_sources_tray(sources: List[Dict[str, Any]], citations: Optional[List[str]] = None):
    """Renders structured expander cards for each retrieved regulatory source chunk (Tasks 2 & 3)."""
    if not sources:
        return

    # Task 2: Display citations clearly beside or below the answer
    if citations:
        pills_html = " ".join([f"<span class='citation-pill'>{c}</span>" for c in citations])
        st.markdown(
            f"<div style='margin-top: 6px; margin-bottom: 8px;'><strong>📌 Verified Citations:</strong> {pills_html}</div>",
            unsafe_allow_html=True,
        )

    # Task 3: Let users view cited source content in an interactive expander
    with st.expander(f"📚 Inspect Retrieved Sources & Citations ({len(sources)} Chunks)", expanded=False):
        for idx, src in enumerate(sources, start=1):
            marker = src.get("marker", f"[{idx}]")
            doc_name = src.get("source_document", "Unknown Circular")
            chunk_id = src.get("chunk_id", "N/A")
            section = src.get("section") or "General"
            page = src.get("page_number", 1)
            score = src.get("similarity_score", 0.0)
            verbatim = src.get("verbatim_text", "").strip()

            st.markdown(
                f"""
                <div class="source-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #0284c7;">
                            <span class="citation-pill">{marker}</span> {doc_name}
                        </span>
                        <span class="badge badge-info">Similarity: {score:.3f}</span>
                    </div>
                    <div class="source-meta">
                        <span>🆔 <strong>Chunk ID:</strong> <code>{chunk_id}</code></span>
                        <span>📑 <strong>Section:</strong> {section}</span>
                        <span>📄 <strong>Page:</strong> {page}</span>
                    </div>
                    <div class="source-verbatim">{verbatim or "No excerpt available."}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# Chat History Message Rendering (Task 1 & Task 3)
# ==============================================================================

for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    msg_status = msg.get("status", "success")
    sources = msg.get("sources", [])
    citations = msg.get("citations", [])
    meta = msg.get("metadata", {})

    with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "⚖️"):
        # Assistant status badges
        if role == "assistant" and msg_status != "system":
            if msg_status == "refusal":
                st.markdown(
                    "<span class='badge badge-refusal'>🟠 Safe Guardrail Refusal</span>",
                    unsafe_allow_html=True,
                )
            elif msg_status == "error":
                st.markdown(
                    "<span class='badge badge-error'>🔴 System Error</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='badge badge-success'>🟢 Grounded Answer</span>",
                    unsafe_allow_html=True,
                )

        # Message Body
        st.markdown(content)

        # Display retrieved sources alongside assistant answer (Task 3)
        if role == "assistant" and sources:
            render_sources_tray(sources, citations=citations)

        # Diagnostic metadata pill
        if role == "assistant" and meta:
            lat = meta.get("latency_seconds")
            top_score = meta.get("top_similarity_score")
            model_name = meta.get("model", "llama3:latest")
            p_tok = meta.get("prompt_tokens")
            c_tok = meta.get("completion_tokens")

            meta_parts = []
            if lat is not None:
                meta_parts.append(f"⏱️ {lat}s")
            if model_name:
                meta_parts.append(f"🧠 {model_name}")
            if top_score is not None:
                meta_parts.append(f"🎯 Score: {top_score:.3f}")
            if p_tok and c_tok:
                meta_parts.append(f"📊 Tokens: {p_tok}+{c_tok}")

            if meta_parts:
                st.markdown(
                    f"<div class='meta-pill'>{' • '.join(meta_parts)}</div>",
                    unsafe_allow_html=True,
                )


# ==============================================================================
# User Input & Query Execution (Tasks 1, 2, 3, 4)
# ==============================================================================

# Check for pending prompt from quick queries
user_query = None
if st.session_state.pending_prompt:
    user_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    user_query = st.chat_input("Ask a banking compliance question (e.g. CERT-In reporting deadlines)...")

if user_query:
    cleaned_query = user_query.strip()

    # Input validation (Task 4)
    if not cleaned_query:
        st.warning("⚠️ Please enter a non-empty question.")
    elif len(cleaned_query) < 3:
        st.warning("⚠️ Question is too short. Please provide at least 3 characters.")
    else:
        # 1. Append and render user question
        st.session_state.messages.append({
            "role": "user",
            "content": cleaned_query,
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        })

        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(cleaned_query)

        # 2. Execute RAG query with progressive streaming (Tasks 1, 2, 3, 4)
        with st.chat_message("assistant", avatar="⚖️"):
            badge_placeholder = st.empty()
            answer_placeholder = st.empty()
            sources_placeholder = st.empty()
            meta_placeholder = st.empty()

            accumulated_text = ""
            retrieved_sources: List[Dict[str, Any]] = []
            final_citations: List[str] = []
            final_metadata: Dict[str, Any] = {}
            outcome_status = "success"
            stream_error: Optional[str] = None

            status_container = st.status("🔍 Analyzing regulatory corpus & streaming response...", expanded=True)
            with status_container:
                st.write("1. Retrieving candidate regulatory chunks from ChromaDB...")
                st.write("2. Evaluating hallucination guardrails and relevance...")
                st.write("3. Progressive token streaming active...")

            stream_gen = client.submit_query_stream(
                question=cleaned_query,
                top_k=st.session_state.top_k,
                include_metadata=True,
            )

            for event in stream_gen:
                evt_type = event.get("type")

                if evt_type == "sources":
                    retrieved_sources = event.get("sources", [])
                    # Render sources immediately while LLM streams! (Task 3)
                    if retrieved_sources:
                        with sources_placeholder.container():
                            render_sources_tray(retrieved_sources)

                elif evt_type == "token":
                    token = event.get("token", "")
                    accumulated_text += token
                    answer_placeholder.markdown(accumulated_text + "▌")

                elif evt_type == "done":
                    outcome_status = event.get("status", "success")
                    final_answer = event.get("answer") or accumulated_text
                    final_citations = event.get("citations", [])
                    final_metadata = event.get("metadata", {})
                    accumulated_text = final_answer

                elif evt_type == "error":
                    stream_error = event.get("message") or event.get("error_code")
                    err_code = event.get("error_code", "STREAM_ERROR")
                    outcome_status = "error"

            # Finalize Streamlit status indicator and badges (Tasks 1 & 4)
            if outcome_status == "error":
                status_container.update(label="🔴 Streaming interrupted or failed.", state="error", expanded=True)
                badge_placeholder.markdown("<span class='badge badge-error'>🔴 Request Error</span>", unsafe_allow_html=True)
                if not accumulated_text:
                    answer_placeholder.error(f"**Streaming Error**: {stream_error or 'Stream was interrupted.'}")
                else:
                    answer_placeholder.markdown(accumulated_text)
                    st.error(f"⚠️ **Stream Interrupted**: {stream_error}")
            elif outcome_status == "refusal":
                status_container.update(label="🟠 Guardrail refusal triggered.", state="complete", expanded=False)
                badge_placeholder.markdown("<span class='badge badge-refusal'>🟠 Safe Guardrail Refusal</span>", unsafe_allow_html=True)
                answer_placeholder.markdown(accumulated_text)
            else:
                status_container.update(label="✅ Answer streamed successfully!", state="complete", expanded=False)
                badge_placeholder.markdown("<span class='badge badge-success'>🟢 Grounded Answer (Streamed)</span>", unsafe_allow_html=True)
                answer_placeholder.markdown(accumulated_text)

            # Re-render sources and verified citations tray (Tasks 2 & 3)
            if retrieved_sources:
                with sources_placeholder.container():
                    render_sources_tray(retrieved_sources, citations=final_citations)

            # Diagnostic metadata
            if final_metadata:
                lat = final_metadata.get("latency_seconds")
                top_score = final_metadata.get("top_similarity_score")
                model_name = final_metadata.get("model", "llama3:latest")
                meta_parts = []
                if lat is not None:
                    meta_parts.append(f"⏱️ {lat}s")
                if model_name:
                    meta_parts.append(f"🧠 {model_name} (Streamed)")
                if top_score is not None:
                    meta_parts.append(f"🎯 Score: {top_score:.3f}")
                meta_placeholder.markdown(f"<div class='meta-pill'>{' • '.join(meta_parts)}</div>", unsafe_allow_html=True)

            # Append assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": accumulated_text if not stream_error else f"{accumulated_text}\n\n⚠️ **Error**: {stream_error}",
                "status": outcome_status,
                "sources": retrieved_sources,
                "citations": final_citations,
                "metadata": final_metadata,
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            })

        st.rerun()


# Main execution hook for standalone launch
def main():
    """Entry point for streamlit runner."""
    pass


if __name__ == "__main__":
    main()
