# RegulSense: End-to-End RAG Pipeline Execution & Grounding Report

- **Execution Timestamp**: `2026-09-14T15:14:46.073266+00:00`
- **Chat Generation Model**: `llama3:latest`
- **Total End-to-End Latency**: `85.285s`
- **Total Tokens Consumed**: `1365` (`1291` prompt + `74` completion)

---

## 1. User Inquiry & Synthesized Grounded Response (Task 3)

### **Compliance Question**:
> *"What is the mandatory timeframe for banks to report Severity 1 cyber security incidents to CERT-In and RBI?"*

### **RegulSense Model Answer**:
According to the Reserve Bank of India's Master Direction on Cyber Resilience and Digital Payment Security Controls, banks must report any Severity 1 cyber security incident, ransomware compromise, unauthorized system intrusion, or major denial of service (DoS) affecting customer-facing channels to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection.

---

## 2. Attributed Sources & Evidentiary Provenance (Task 3)

The following regulatory circular chunks were retrieved and injected as context:

| Rank | Source Document | Section | Chunk ID | Similarity Score | Excerpt Preview |
| :---: | :--- | :--- | :--- | :---: | :--- |
| #1 | `cyber_resilience_framework.pdf` | Preamble / Document Header | `cyber_resilience_framework_pdf_tokenaware_001` | `0.6520` | "RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cy..." |
| #2 | `cyber_resilience_framework.pdf` | Preamble / Document Header | `cyber_resilience_framework_pdf_tokenaware_002` | `0.5681` | "x7x365 Security Operations Centre (SOC) equipped with continuous automated log monitoring, SIEM analytics, and automated threat hunting capabilities. 4. API Security and Third-Party Risk Management (a) All open banking a..." |
| #3 | `circular_dor_2024_108.txt` | 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | `circular_dor_2024_108_txt_tokenaware_003` | `0.5629` | "officer not below the rank of Deputy General Manager. (b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balanc..." |

---

## 3. Modular Stage Execution Breakdown & Latency Ledger (Task 2 & 4)

| Pipeline Stage | Function / Stage | Latency | Tokens / Output | Technical Responsibility |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 1: Embed** | `embed_query_stage()` | `2.7625s` | 384 coordinates | Project query into dense vector space |
| **Stage 2: Retrieve** | `retrieve_chunks_stage()` | `0.0954s` | 3 chunks | ChromaDB cosine similarity search |
| **Stage 3: Assemble** | `assemble_context_stage()` | `0.1420s` | 1002 context tokens | Format headers, enforce budget, render prompts |
| **Stage 4: Generate** | `generate_answer_stage()` | `82.2833s` | 74 tokens | Grounded LLM inference ($T=0.2$) |
| **Stage 5: Attribute** | `attribute_sources_stage()` | `< 0.001s` | 3 citations | Provenance audit trail extraction |
| **Total Pipeline** | `RAGPipeline.run()` | **`85.2846s`** | **`1365` tokens** | Complete end-to-end query resolution |

---

## 4. Assembled Context Preview (Prompt Injected into Generator)

```text
[CIRCULAR EVIDENCE 1]
Document: cyber_resilience_framework.pdf
Section: Preamble / Document Header (Page 1)
Chunk ID: cyber_resilience_framework_pdf_tokenaware_001 | Relevance Score: 0.6520
Content:
RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls 1. Mandatory Two-Factor Authentication (2FA) All regulated payment system operators and scheduled commercial banks must enforce dynamic two-factor authentication (2FA) for all domestic electronic fund transfers (NEFT, RTGS, IMPS, and UPI). At least one factor must be dynamic, such as a time-based one-time password (TOTP) or biometric verification. 2. Incident Reporting Timelines (6-Hour Rule) Any cyber security incident, ransomware compromise, unauthorized system intrusion, or major denial of service (DoS) affecting customer-facing channels must be reported to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection. Initial reports must be followed by a comprehensive forensic analysis report within 7 business days. 3. Security Operations Centre (SOC) Operations Banks must operate a 24x7x365 Security Operations Centre (SOC) equipped with continuous automated log monitoring, SIEM analytics, and automated threat hunting capabilities.

4. API Security and Third-Party Risk Management
(a) All open banking and fintech integrations must mandate

---
[CIRCULAR EVIDENCE 2]
Document: cyber_resilience_framework.pdf
Section: Preamble / Document Header (Page 1)
Chunk ID: cyber_resilience_framework_pdf_tokenaware_002 | Relevance Score: 0.5681
Content:
x7x365 Security Operations Centre (SOC) equipped with continuous automated log monitoring, SIEM analytics, and automated threat hunting capabilities.

4. API Security and Third-Party Risk Management
(a) All open banking and fintech integrations must mandate TLS 1.3 encryption with certificate pinning.
(b) Third-party service providers and cloud vendors must undergo mandatory annual vulnerability assessment and penetration testing (VAPT) conducted by CERT-In empaneled auditors.
5. Audit Trail and Forensic Preservation
All critical access logs, privileged user operations, database queries, and perimeter firewall event records must be stored in immutable WORM (Write Once, Read Many) storage for a minimum duration of three (3) years to aid statutory forensic investigations.
6. Supervisory Penalties
Non-compliance with the cyber resilience directives will invite punitive supervisory action under Section 47A of the Banking Regulation Act, 1949, including operational restrictions on digital onboarding.
Chief General Manager (Information Technology)
Reserve Bank of India

---
[CIRCULAR EVIDENCE 3]
Document: circular_dor_2024_108.txt
Section: 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs (Page 1)
Chunk ID: circular_dor_2024_108_txt_tokenaware_003 | Relevance Score: 0.5629
Content:
officer not below the rank of Deputy General Manager.
(b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
(c) Heightened Transaction Monitoring: High-risk accounts shall be subjected to quarterly reviews, compared against the standard biennial review for low-risk customers.

4. Transaction Monitoring and Reporting Thresholds
Banks shall deploy rule-based and behavioral automated transaction monitoring systems to identify suspicious transaction patterns:
(a) Cash Transaction Reports (CTRs): All cash transactions of the value of more than rupees ten lakhs or its equivalent in foreign currency must be reported monthly to the Financial Intelligence Unit - India (FIU-IND) by the 15th day of the succeeding month.
(b) Counterfeit Currency Reports (CCRs) and Non-Profit Organization Transaction Reports (NTRs): All cross-border non-profit transactions exceeding rupees ten lakhs must be logged and monitored.
(c) Suspicious Transaction Reports (STRs): If any transaction gives rise to reasonable suspicion of illicit funds, evasion, or terrorism financing, an STR shall be furnished to FIU-IND within seven working days of arriving at such a conclusion.

5. Record Retention Obligations
Under Rule 3 and Rule 10 of the PML Rules, 2005, all regulated entities shall maintain:
(a) Transaction Records: Comprehensive records of all transactions, whether completed or attempted, domestic or international, for a minimum

```

---
*Report automatically generated by `src/rag_pipeline.py` for RegulSense Banking Compliance Assistant.*