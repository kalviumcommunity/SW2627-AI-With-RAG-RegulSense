# RegulSense Progressive Streaming Responses & Source Citation Inspection

## Executive Overview
RegulSense has been updated with **Progressive Answer Streaming** (Server-Sent Events) and **Interactive Source Citation Inspection**.
Users no longer have to wait for the complete RAG synthesis before seeing output. Candidate circular source chunks are delivered immediately, followed by progressive token streaming with an animated typing cursor and transparent citation verification.

---

## 1. Task Verification Matrix

| Task | Requirement | Status | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Task 1** | Stream answers progressively | **PASSED** | Backend SSE generator yields incremental tokens (`TTFT=111.992s`, `0.9` tokens/sec) |
| **Task 2** | Display citations clearly | **PASSED** | Verifiable citation markers ([1]) displayed alongside answer |
| **Task 3** | Let users view cited sources | **PASSED** | Interactive inspection tray renders 3 source cards with chunk IDs & verbatim excerpts |
| **Task 4** | Handle streaming errors | **PASSED** | Gracefully handles input validation, backend offline, and stream interruptions |
| **Task 5** | Commit sample interaction | **PASSED** | JSON telemetry, Markdown demonstration, UI screenshot, and unit tests committed |

---

## 2. Sample Grounded Streaming Interaction

**User Query:**
> *"What are the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents?"*

**Streamed Grounded Answer:**
> Based on the provided regulatory context, the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents are as follows:

**Initial Report:** Within 6 hours of detection, banks must report the incident to the RBI Cyber Security Cell (CSITE) and CERT-In. [1]

**Comprehensive Forensic Analysis Report:** Initial reports must be followed by a comprehensive forensic analysis report within 7 business days. [1]

Please note that these reporting procedures are specific to Severity 1 cyber security incidents, as defined in the provided regulatory context.

**Verified Citations:** `[1]`

### Streaming Performance Telemetry
- **Time to First Token (TTFT):** `111.9923s`
- **Total Streaming Latency:** `135.2464s`
- **Tokens Streamed:** `117`
- **Effective Generation Speed:** `0.9 tokens/second`

---

## 3. Interactive Source Citation Inspection (Task 3)

Below is the exact metadata rendered in the UI source inspection panel:

### Source [1]: cyber_resilience_framework.pdf
- **Chunk ID:** `cyber_resilience_framework_pdf_tokenaware_001`
- **Regulatory Section:** Preamble / Document Header
- **Page Number:** 1
- **Cosine Similarity Score:** `0.6436`

**Verbatim Circular Excerpt:**
```text
RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls 1. Mandatory Two-Factor Authentication (2FA) All regulated payment system operators and scheduled commercial banks must enforce...
```

### Source [2]: circular_dor_2024_108.txt
- **Chunk ID:** `circular_dor_2024_108_txt_tokenaware_003`
- **Regulatory Section:** 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs
- **Page Number:** 1
- **Cosine Similarity Score:** `0.5709`

**Verbatim Circular Excerpt:**
```text
officer not below the rank of Deputy General Manager.
(b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
(c) Heightened Transaction Monitoring: High-risk accounts shall be subjected to quarterly reviews, compared against the standard biennial review for low-risk customers.

...
```

### Source [3]: sample_regulatory_circular.txt
- **Chunk ID:** `sample_regulatory_circular_txt_tokenaware_003`
- **Regulatory Section:** 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs
- **Page Number:** 1
- **Cosine Similarity Score:** `0.5709`

**Verbatim Circular Excerpt:**
```text
officer not below the rank of Deputy General Manager.
(b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
(c) Heightened Transaction Monitoring: High-risk accounts shall be subjected to quarterly reviews, compared against the standard biennial review for low-risk customers.

...
```

---

## 4. Fault Tolerance & Streaming Error Handling (Task 4)

### Error Scenario: `STREAMED_GUARDRAIL_REFUSAL`
- **Description:** Hallucination guardrail identifies low relevance or out-of-corpus query and streams safe refusal.
- **Test Query:** *"What are the reserve capital requirements for Martian colony commercial banks?"*
- **Streamed Refusal Output:** *"The provided regulatory context does not contain sufficient information to answer this question reliably.

**Guardrail D..."*

### Error Scenario: `INPUT_VALIDATION_TOO_SHORT`
- **Description:** Client validates query length before dispatching stream request.
- **Test Query:** *"hi"*
- **Client Events Emitted:** `[{"type": "error", "error_code": "QUESTION_TOO_SHORT", "message": "Question must be at least 3 characters long."}]`

### Error Scenario: `INPUT_VALIDATION_EMPTY`
- **Description:** Client validates non-empty query before dispatching stream request.
- **Test Query:** *""*
- **Client Events Emitted:** `[{"type": "error", "error_code": "EMPTY_QUESTION", "message": "Question cannot be empty or whitespace only."}]`

### Error Scenario: `BACKEND_OFFLINE_STREAMING_RECOVERY`
- **Description:** Client catches connection failure or timeout and emits structured BACKEND_OFFLINE error.
- **Test Query:** *"Is the service reachable?"*
- **Client Events Emitted:** `[{"type": "error", "error_code": "BACKEND_OFFLINE", "message": "Backend service unreachable at http://127.0.0.1:59999. Start backend with 'uvicorn src.api:app --port 8000'."}]`

---

## 5. End-to-End Architecture Flow

```
+-----------------------------------------------------------------------------------+
| Streamlit Chat UI (src/chat_app.py)                                               |
|  1. User enters prompt                                                            |
|  2. Displays status container & triggers submit_query_stream()                   |
|  3. [Event: sources] -> Immediately renders 'Inspect Retrieved Sources' tray     |
|  4. [Event: token]   -> Progressively appends tokens with animated cursor '▌'     |
|  5. [Event: done]    -> Finalizes citations, badges, and diagnostic metadata      |
+-----------------------------------------------------------------------------------+
                                         |                                           
                     (HTTP Server-Sent Events / In-Process)                         
                                         v                                           
+-----------------------------------------------------------------------------------+
| Backend API (src/api.py: /api/v1/query/stream)                                    |
|  - Calls HallucinationGuardrail.execute_stream()                                  |
|  - Assesses retrieval similarity threshold (tau = 0.50)                           |
|  - Yields: sources payload -> token deltas -> done completion payload             |
+-----------------------------------------------------------------------------------+
```

## 6. How to Run Locally

### Option 1: Standalone Direct Mode
```powershell
streamlit run src/chat_app.py
```

### Option 2: Full Distributed Client-Server Mode
```powershell
# Terminal 1: Launch FastAPI Backend Server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Launch Streamlit Web UI
streamlit run src/chat_app.py
```