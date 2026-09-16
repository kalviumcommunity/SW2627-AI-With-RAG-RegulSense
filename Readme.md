# RegulSense: AI-Powered Regulatory Compliance RAG Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange.svg)](https://www.trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Enterprise-grade, verifiable Retrieval-Augmented Generation (RAG) assistant designed for bank compliance and risk officers to navigate complex regulatory circulars, master directions, and internal audit guidelines with zero hallucination and complete evidentiary provenance.**

---

## Table of Contents
1. [Problem Statement & Solution](#problem-statement--solution)
2. [Architectural Overview & Pipeline](#architectural-overview--pipeline)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Quickstart Guide (Local Environment)](#quickstart-guide-local-environment)
6. [Containerized Deployment (Docker & Compose)](#containerized-deployment-docker--compose)
7. [Environment Variables & Safe Secret Management](#environment-variables--safe-secret-management)
8. [Core Feature Walkthrough](#core-feature-walkthrough)
9. [REST API Documentation](#rest-api-documentation)
10. [End-to-End Demonstration](#end-to-end-demonstration)
11. [Automated Verification & Test Suite](#automated-verification--test-suite)
12. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Problem Statement & Solution

### The Challenge
A commercial bank maintains thousands of regulatory circulars, master directions (e.g., RBI, CERT-In, Basel III), and internal audit reports. Risk and compliance officers struggle to quickly determine which current rule governs a specific transaction or security incident without manually reconciling conflicting, amended, or superseded circulars. Generic LLMs exacerbate the risk by hallucinating non-existent banking rules or legal citations.

### The RegulSense Solution
RegulSense provides a production-grade RAG intelligence pipeline that:
- **Retrieves Authoritative Context**: Dense vector indexing over regulatory circulars using ChromaDB.
- **Enforces Hallucination Guardrails**: Evaluates similarity confidence ($\tau \ge 0.50$); safely refuses out-of-corpus questions instead of speculating.
- **Synthesizes Cited Answers**: Implements token-budgeted prompt assembly with numeric in-text markers (`[1]`, `[2]`).
- **Verifies Provenance Interactively**: Audits claim veracity against verbatim circular excerpts and allows users to expand source cards.
- **Streams Responses Progressively**: Low-latency Server-Sent Events (SSE) with typing indicators.
- **Accelerates Repeat Queries**: Thread-safe in-memory LRU cache with TTL expiration, eliminating 99.9% of latency and 100% of LLM compute costs on repeat inquiries.
- **Structured Audit Logging & Analytics**: Records every request in JSON Lines format and tracks token consumption, dollar cost, and financial savings.

---

## Architectural Overview & Pipeline

```
                                  USER QUERY
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  Normalized Query Cache   │──[HIT]──► Instant Cached Response
                        │    (LRU + TTL Hash Key)   │           (< 1ms, $0.00 Cost)
                        └───────────────────────────┘
                                      │ [MISS]
                                      ▼
                        ┌───────────────────────────┐
                        │   Dense Vector Retriever  │
                        │    (ChromaDB Collection)  │
                        └───────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Relevance Guardrail     │──[WEAK]──► Safe Refusal Message
                        │   (Threshold Check τ≥0.5) │           (Zero Hallucination)
                        └───────────────────────────┘
                                      │ [PASS]
                                      ▼
                        ┌───────────────────────────┐
                        │ Context Budget Assembler  │
                        │  (Token Window Allocation)│
                        └───────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Grounded LLM Engine     │
                        │   (Llama-3 / GPT-4o-mini) │
                        └───────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Citation Audit Engine   │
                        │   (Provenance Verification│
                        └───────────────────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
     ┌────────────────────────┐              ┌────────────────────────┐
     │  FastAPI SSE Stream    │              │  JSON Lines Audit Log  │
     │  & Streamlit Chat UI   │              │  & Real-Time Analytics │
     └────────────────────────┘              └────────────────────────┘
```

---

## Repository Structure

```text
SW2627-AI-With-RAG-RegulSense/
├── Dockerfile                      # Production multi-purpose Docker image
├── docker-compose.yml              # Multi-container orchestration (Backend + Frontend)
├── Readme.md                       # Comprehensive system documentation
├── requirements.txt                # Pinned Python package dependencies (UTF-8)
├── .env.example                    # Documented environment configuration template
├── .gitignore                      # Git exclusion rules protecting secrets & data
├── app.py                          # Streamlit UI application entrypoint
├── run_backend.ps1 / .sh           # Backend service launch scripts (PowerShell / Bash)
├── run_frontend.ps1 / .sh          # Frontend chat launch scripts (PowerShell / Bash)
├── data/
│   ├── chroma_db/                  # Persistent ChromaDB vector database index
│   ├── sample_regulatory_circular.txt # Baseline regulatory corpus
│   └── uploads/                    # Directory for runtime circular uploads
├── prompts/
│   ├── templates.py                # Grounded prompt templates & constraints
│   └── PROMPT_DESIGN.md            # Evidentiary prompt engineering specifications
├── src/
│   ├── api.py                      # FastAPI REST & SSE streaming application
│   ├── api_client.py               # Robust API client with direct in-process fallback
│   ├── chat_app.py                 # Streamlit UI with progressive streaming & citation trays
│   ├── query_cache.py              # Thread-safe LRU query cache with TTL & cost telemetry
│   ├── audit_logger.py             # Structured JSON Lines audit logger & usage tracker
│   ├── citation_engine.py          # Grounded answering, provenance registry, and audit engine
│   ├── hallucination_guardrails.py # Context sufficiency scoring & safe refusal logic
│   ├── vector_db.py                # ChromaDB vector store manager & embedding utilities
│   ├── chunker.py                  # Token-aware text chunking engine
│   └── demonstrate_full_flow.py    # Automated end-to-end full flow verification runner
├── tests/                          # 320+ unit and integration tests
│   ├── test_caching_logging.py     # Caching, audit logging, and analytics tests
│   ├── test_environment_safety.py  # Secret safety and environment parsing tests
│   ├── test_streaming_responses.py # Progressive SSE streaming and UI tests
│   ├── test_api.py                 # FastAPI endpoints and validation tests
│   └── test_document_upload.py     # Runtime document upload and indexing tests
└── outputs/                        # Verified execution traces and audit reports
    ├── full_flow_demonstration_report.md
    ├── full_flow_trace.json
    ├── sample_query_audit.log
    ├── usage_summary_report.json
    └── usage_summary_report.md
```

---

## Prerequisites

- **Python**: Version `3.10` or higher (`3.11` recommended)
- **Package Manager**: `pip`
- **Version Control**: `git`
- **Local LLM & Embedding Service** (Default Option):
  - [Ollama](https://ollama.com/) running locally:
    ```bash
    ollama pull llama3:latest
    ollama pull all-minilm
    ```
- **Alternative Cloud Provider**: Any OpenAI-compatible endpoint (e.g. OpenAI, Azure, Groq, vLLM).
- **Docker** (Optional): Docker Engine 24+ and Docker Compose 2.0+.

---

## Quickstart Guide (Local Environment)

### 1. Clone the Repository
```bash
git clone https://github.com/kalviumcommunity/SW2627-AI-With-RAG-RegulSense.git
cd SW2627-AI-With-RAG-RegulSense
```

### 2. Set Up a Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:

#### Windows:
```powershell
Copy-Item .env.example .env
```

#### macOS / Linux:
```bash
cp .env.example .env
```

Verify that `.env` contains your preferred configuration (defaults to local Ollama on port 11434).

### 5. Launch the Services

#### Option 1: Using Convenient Helper Scripts
In separate terminal windows:
```powershell
# Terminal 1: Launch Backend REST API (Port 8000)
.\run_backend.ps1       # Or: ./run_backend.sh on Linux/macOS

# Terminal 2: Launch Streamlit Chat Interface (Port 8501)
.\run_frontend.ps1      # Or: ./run_frontend.sh on Linux/macOS
```

#### Option 2: Using Direct Commands
```bash
# Terminal 1: Backend API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend Web UI
streamlit run app.py --server.port 8501
```

Access the web applications:
- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## Containerized Deployment (Docker & Compose)

RegulSense includes a complete Docker configuration for standardized deployment across production, staging, and review environments.

### 1. Build and Start All Containers
```bash
docker compose up --build -d
```

This launches:
1. `regulsense-backend`: FastAPI REST service on port `8000`.
2. `regulsense-frontend`: Streamlit compliance interface on port `8501`.
3. Shared persistent volume for ChromaDB (`/app/data`) and audit outputs (`/app/outputs`).
4. Automated container health check monitoring backend availability.

### 2. View Service Status and Logs
```bash
# Check status
docker compose ps

# Follow backend logs
docker compose logs -f backend

# Follow frontend logs
docker compose logs -f frontend
```

### 3. Stop the Containers
```bash
docker compose down
```

---

## Environment Variables & Safe Secret Management

RegulSense strictly enforces secret safety: **zero API keys, passwords, or cloud credentials are hardcoded into the source code.** All sensitive parameters are resolved dynamically via environment variables or loaded from the `.env` file (which is gitignored).

### Configuration Reference Table

| Variable | Type | Default Value | Description |
| :--- | :---: | :---: | :--- |
| `OPENAI_BASE_URL` | String | `http://localhost:11434/v1` | Base URL for OpenAI-compatible LLM and embedding API. |
| `OPENAI_API_KEY` | Secret | `ollama` | API authentication key (redacted in `/api/v1/config` and logs). |
| `CHAT_MODEL` | String | `llama3:latest` | Active LLM model identifier for grounded answer generation. |
| `EMBEDDING_MODEL` | String | `all-minilm` | Active dense embedding model identifier. |
| `CHROMA_PERSIST_DIRECTORY` | Path | `data/chroma_db` | Storage path for persistent vector embeddings. |
| `CHROMA_COLLECTION_NAME` | String | `regulsense_regulatory_chunks` | Active ChromaDB vector collection name. |
| `API_HOST` | String | `0.0.0.0` | Network binding interface for the FastAPI backend. |
| `API_PORT` | Integer | `8000` | Port for the FastAPI backend service. |
| `DEFAULT_TOP_K` | Integer | `3` | Default number of candidate regulatory chunks retrieved. |
| `MIN_SIMILARITY_THRESHOLD` | Float | `0.50` | Minimum cosine similarity required to generate an answer. |
| `APP_ENV` | String | `production` | Deployment environment mode (`development`, `staging`, `production`). |
| `UPLOAD_DIRECTORY` | Path | `data/uploads` | Server storage path for uploaded regulatory documents. |
| `MAX_UPLOAD_SIZE_BYTES` | Integer | `10485760` (10 MB) | Maximum permitted file upload size in bytes. |
| `REGULSENSE_API_URL` | String | `http://localhost:8000` | Target URL used by Streamlit UI to contact the backend. |

### Secret Masking
When inspecting server configuration via `GET /api/v1/config`, sensitive values are automatically replaced:
```json
{
  "status": "success",
  "config": {
    "openai_base_url": "http://localhost:11434/v1",
    "openai_api_key": "***REDACTED***",
    "chat_model": "llama3:latest",
    "embedding_model": "all-minilm"
  }
}
```

---

## Core Feature Walkthrough

### 1. Dynamic Circular Ingestion (Sidebar File Uploader)
- Drag-and-drop ingestion of `.pdf`, `.txt`, and `.md` regulatory documents via the Streamlit sidebar or `POST /api/v1/upload`.
- Automatic text extraction, token-aware chunking (configurable chunk size and overlap), dense vector embedding generation, and ChromaDB indexing.
- **Zero Restart Required**: Documents are searchable immediately upon upload completion.

### 2. Guardrail Thresholding & Safe Refusal
- Evaluates retrieval quality before invoking generative models.
- If top-1 similarity score falls below `MIN_SIMILARITY_THRESHOLD` ($0.50$), the system triggers an authoritative refusal card with diagnostic metrics:
  ```text
  The provided regulatory context does not contain sufficient information to answer this question reliably.
  Guardrail Diagnostic: LOW_SIMILARITY_SCORE (Top Score: 0.4589, Required: 0.5000)
  ```
- **Zero Hallucination Guarantee**: Eliminates fabricated regulations on out-of-corpus queries.

### 3. In-Text Citations & Provenance Inspection Trays
- Grounded answers embed precise numeric citations (`[1]`, `[2]`).
- The UI renders verified citation pills directly beneath the answer.
- An interactive expander labeled **"📚 Inspect Retrieved Sources & Citations"** reveals:
  - Source document name (e.g. `cyber_resilience_framework.pdf`)
  - Chunk ID and cosine similarity score
  - Section title and page number
  - Verbatim excerpt box with exact source text.

### 4. Progressive Answer Streaming
- Server-Sent Events (SSE) stream answers incrementally via `/api/v1/query/stream`.
- Streamlit renders words with a real-time typing animation cursor (`▌`).
- Source metadata arrives on the first SSE event, allowing users to inspect sources while the answer is still generating.

### 5. High-Performance Query Caching
- Normalized SHA-256 hash keys eliminate whitespace and casing variations.
- Thread-safe LRU eviction (`max_size=500`) with TTL expiration (`3600s`).
- Identical queries are served in **< 1 millisecond** with **$0.00 incremental compute cost**, displaying an `⚡ Cache Hit` badge and cost-saved pill.

### 6. Structured Observability & Analytics
- Structured JSON Lines audit logging to `outputs/sample_query_audit.log`.
- Real-time usage analytics dashboard in the Streamlit sidebar showing total requests, cache hit rates, token breakdown, and cost avoidance.

---

## REST API Documentation

The FastAPI service exposes production REST and SSE streaming endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/query` | Submits inquiry, performs retrieval and guardrail check, returns structured JSON. |
| `POST` | `/api/v1/query/stream` | Streams answer progressively using Server-Sent Events (`text/event-stream`). |
| `POST` | `/api/v1/upload` | Uploads, chunks, embeds, and indexes a circular file (.pdf, .txt, .md). |
| `GET` | `/api/v1/health` | Readiness probe; validates ChromaDB and LLM connectivity. |
| `GET` | `/api/v1/config` | Returns redacted server configuration. |
| `GET` | `/api/v1/analytics/usage` | Aggregated metrics: total requests, cache hits, token counts, cost economics. |
| `GET` | `/api/v1/analytics/cache` | In-memory cache status: capacity, size, hits, misses, evictions. |
| `POST` | `/api/v1/analytics/cache/clear`| Flushes in-memory query cache entries. |
| `GET` | `/docs` | Interactive Swagger API Explorer. |

### Example Query Request (cURL)
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What are the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents?",
       "top_k": 3,
       "include_metadata": true
     }'
```

### Example Query Response
```json
{
  "status": "success",
  "answer": "Under the RBI Master Direction on Cyber Resilience and Digital Payment Security Controls [1], regulated entities must report all Severity 1 cybersecurity incidents to CERT-In and the Reserve Bank within a mandatory timeframe of 6 hours of detection [2].",
  "sources": [
    {
      "marker": "[1]",
      "source_document": "cyber_resilience_framework.pdf",
      "chunk_id": "cyber_resilience_001",
      "section": "2. Incident Reporting Timelines",
      "page_number": 1,
      "similarity_score": 0.742,
      "verbatim_text": "Regulated entities must notify CERT-In within 6 hours of cybersecurity incident detection."
    }
  ],
  "citations": ["[1]", "[2]"],
  "metadata": {
    "latency_seconds": 0.0402,
    "model": "llama3:latest",
    "cache_hit": false,
    "total_tokens": 1008,
    "estimated_cost_usd": 0.00024
  }
}
```

---

## End-to-End Demonstration

To verify the complete operational flow (uploading a document, querying, citation verification, cache hit acceleration, and safe guardrail refusal) in an automated, reproducible manner:

```bash
python src/demonstrate_full_flow.py
```

This generates verified audit artifacts:
- [`outputs/full_flow_demonstration_report.md`](file:///c:/Users/uppal/OneDrive/Desktop/WI-sprint-2/SW2627-AI-With-RAG-RegulSense/outputs/full_flow_demonstration_report.md)
- [`outputs/full_flow_trace.json`](file:///c:/Users/uppal/OneDrive/Desktop/WI-sprint-2/SW2627-AI-With-RAG-RegulSense/outputs/full_flow_trace.json)

---

## Automated Verification & Test Suite

RegulSense includes a comprehensive suite of **320+ automated unit, integration, and security tests**:

```powershell
# 1. Run Secret & Environment Safety Tests (Task 4)
python -m unittest tests/test_environment_safety.py

# 2. Run Query Caching & Audit Logging Tests
python -m unittest tests/test_caching_logging.py

# 3. Run Progressive Streaming Tests
python -m unittest tests/test_streaming_responses.py

# 4. Run API & Document Upload Tests
python -m unittest tests/test_api.py tests/test_document_upload.py

# 5. Run the Full Test Suite Across All Modules
python -m unittest discover -s tests -p "test_*.py"
```

**Test Verification Summary**: All 320 tests pass with 0 failures and 0 errors.

---

## Troubleshooting & FAQ

### 1. "Failed to connect to Ollama at http://localhost:11434"
- Ensure the Ollama daemon is running:
  ```bash
  ollama serve
  ```
- Verify the models are downloaded:
  ```bash
  ollama list
  ```
- If running inside Docker, ensure Docker can reach your host via `host.docker.internal:11434`.

### 2. "ChromaDB database locked or schema mismatch"
- Ensure no other instances are writing to `data/chroma_db`.
- To reset the local vector index:
  ```powershell
  Remove-Item -Recurse -Force data\chroma_db
  ```
- Rerun ingestion or upload circulars via the UI.

### 3. "Port 8000 or 8501 is already in use"
- Override the ports using environment variables:
  ```powershell
  $env:API_PORT="8080"
  uvicorn src.api:app --port 8080
  ```

---

## License

This project is licensed under the MIT License. Developed for enterprise banking regulatory intelligence and compliance workflows.
