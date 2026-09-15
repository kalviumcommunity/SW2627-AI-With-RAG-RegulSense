# RegulSense Streamlit Chat User Interface - Demonstration Report

## Overview
This report documents the implementation and verification of the RegulSense Chat User Interface,
built with Streamlit to enable conversational regulatory compliance question answering with
verifiable citations and retrieved source chunk inspection.

## Verified Tasks
- **Task 1 - Build question and answer UI**: Conversational chat stream with role-based avatars, message history, and real-time input container.
- **Task 2 - Call the backend RAG API**: Integrated with `POST /api/v1/query` and `GET /api/v1/health` with automatic fallback.
- **Task 3 - Display retrieved sources**: Expandable source inspection trays rendering source document names, chunk IDs, sections, page numbers, similarity scores, and verbatim excerpts.
- **Task 4 - Handle loading and error states**: Multi-stage progress indicators during execution, and dedicated handling for backend connection errors, validation rejections, and safe guardrail refusals.
- **Task 5 - Commit screenshot and sample interactions**: Interactive verification artifacts exported to `outputs/`.

---

## 1. Sample Interaction: Grounded Compliance Query

**User Question:**
> What are the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents?

**Status:** `🟢 Grounded Answer`
**Active Model:** `llama3:latest` | **Latency:** `117.0495s` | **Top Similarity:** `0.6436`

**Assistant Answer:**
Based on the provided regulatory context, the answer to the compliance inquiry is:

According to the Master Direction on Cyber Resilience and Digital Payment Security Controls [1], banks must report any cyber security incident, ransomware compromise, unauthorized system intrusion, or major denial of service (DoS) affecting customer-facing channels to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection [1, Section 2].

This reporting requirement is part of the incident reporting timelines, also known as the 6-Hour Rule, which is a mandatory obligation for banks.

No additional information is required to answer this question, as the provided regulatory context explicitly states the timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents.

**Inline Citations Detected:** `[1]`

### Retrieved Regulatory Sources (Task 3)

| Marker | Source Circular | Chunk ID | Section | Similarity Score |
| :--- | :--- | :--- | :--- | :--- |
| `[1]` | cyber_resilience_framework.pdf | `cyber_resilience_framework_pdf_tokenaware_001` | Preamble / Document Header | **0.6436** |
| `[2]` | circular_dor_2024_108.txt | `circular_dor_2024_108_txt_tokenaware_003` | 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | **0.5709** |
| `[3]` | sample_regulatory_circular.txt | `sample_regulatory_circular_txt_tokenaware_003` | 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | **0.5709** |

#### Verbatim Source Excerpts

**Source [1] - `cyber_resilience_framework.pdf` (`cyber_resilience_framework_pdf_tokenaware_001`)**:
```text
RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls 1. Mandatory Two-Factor Authentication (2FA) All regulated payment system operators and scheduled commercial banks must enforce dynamic two-factor authentication (2FA) for all domestic electronic fund transfers (NEFT, RTGS, IMP...
```

**Source [2] - `circular_dor_2024_108.txt` (`circular_dor_2024_108_txt_tokenaware_003`)**:
```text
officer not below the rank of Deputy General Manager.
(b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
(c) Heightened Transaction Monitoring: High-risk accounts shall be subjected to quarterly reviews, compared against the standard biennial review for low-risk customers.

4. Transaction Monitoring and Reporting Thresholds
Banks shall deploy rule-based and behavioral auto...
```

**Source [3] - `sample_regulatory_circular.txt` (`sample_regulatory_circular_txt_tokenaware_003`)**:
```text
officer not below the rank of Deputy General Manager.
(b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
(c) Heightened Transaction Monitoring: High-risk accounts shall be subjected to quarterly reviews, compared against the standard biennial review for low-risk customers.

4. Transaction Monitoring and Reporting Thresholds
Banks shall deploy rule-based and behavioral auto...
```

---

## 2. Safe Guardrail Refusal & Error State Demonstrations (Task 4)

### Case A: Safe Guardrail Refusal (Weak Context)
- **Question:** `What are the reserve capital requirements for Martian colony commercial banks?`
- **Outcome:** `refusal`
- **Explanation:** The provided regulatory context does not contain sufficient information to answer this question reliably.

**Guardrail Diagnostic**: LOW_SIMILARITY_SCORE
- **Top Similarity Score**: `0.3949` (Threshold Required: `0.5000`)
- **Qualifying Chunks**: `0` (Minimum Required: `1`)
- **Refusal Details**: The highest matching regulatory chunk achieved a similarity score of 0.3949, which is below the minimum required confidence threshold of 0.5000. To prevent hallucination, an answer cannot be generated.

### Case B: Client-Side Input Validation
- **Input:** `'hi'`
- **Validation Code:** `QUESTION_TOO_SHORT`
- **Message:** Question must be at least 3 characters long.

### Case C: Backend Unreachable Handling
- **Error Code:** `BACKEND_OFFLINE` (HTTP 503)
- **UI Behavior:** Prominently alerts the user that the backend server is offline, provides the exact command to launch `uvicorn`, or lets the user enable Direct In-Process Mode.

---

## 3. How to Launch the Chat Interface

### Option A: Standard Full-Stack Mode
```powershell
# Terminal 1: Launch FastAPI Backend
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Launch Streamlit Web UI
streamlit run app.py
```

### Option B: Standalone Direct Mode
```powershell
# Run directly without running a separate uvicorn process
streamlit run app.py
# The UI automatically falls back to in-process pipeline evaluation!
```