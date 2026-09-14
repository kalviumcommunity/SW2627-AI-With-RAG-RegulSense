# RegulSense: Grounded Context Assembly & Augmented Prompt Report

- **Execution Timestamp**: `2026-09-14T15:27:56.936541+00:00`
- **Tokenizer**: `tiktoken` (`cl100k_base`)
- **Configured Total Token Budget**: `2048` tokens
- **Budget Adherence**: `✅ STRICTLY WITHIN BUDGET`
- **Injected Chunks**: `3` included (`0` dropped, truncated: `False`)

---

## 1. Token Budget Allocation & Consumption Ledger (Task 2)

| Component | Allocated Tokens | Budget Limit / Role | % of Budget | Status |
| :--- | :---: | :--- | :---: | :---: |
| **System Grounding Instructions** | `334` | Role, boundary guardrails, citation rules | 16.3% | Guaranteed |
| **User Compliance Inquiry** | `24` | Natural language question | 1.2% | Preserved |
| **Injected Regulatory Context** | `1102` | Candidate circular evidence with source markers | 53.8% | Budget-Enforced |
| **Reserved Output Answer** | `450` | Space guaranteed for model completion | 22.0% | Guaranteed |
| **Safety Margin** | `50` | Delimiter & prompt formatting headroom | 2.4% | Buffer |
| **Total Pipeline Allocated** | **`2005`** | **Total Budget Ceiling (`2048`)** | **97.9%** | **PASS** |
| **Unallocated Remaining Slack** | `43` | Remaining headroom in context window | 2.1% | Available |

---

## 2. Injected Regulatory Chunks with Source Markers (Tasks 1 & 3)

Every candidate chunk is assigned an explicit bracketed marker `[X]` allowing exact citation traceability:

| Marker | Source Document | Section | Page | Chunk ID | Similarity | Tokens | Truncated? |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **`[1]`** | `cyber_resilience_framework.pdf` | Preamble / Document Header | 1 | `cyber_resilience_framework_pdf_tokenaware_001` | `0.6436` | 352 | `False` |
| **`[2]`** | `circular_dor_2024_108.txt` | 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | 1 | `circular_dor_2024_108_txt_tokenaware_003` | `0.5709` | 375 | `False` |
| **`[3]`** | `sample_regulatory_circular.txt` | 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | 1 | `sample_regulatory_circular_txt_tokenaware_003` | `0.5709` | 369 | `False` |

---

## 3. Grounding Instructions & Guardrail Directives (Task 4)

The system prompt injects four non-negotiable compliance guardrails:
1. **Strict Context Boundary**: Answer strictly from provided circular context; no outside speculation.
2. **Mandatory In-Text Citations**: Every factual statement must cite markers (e.g. `[1]`, `[2]`).
3. **Insufficient Context Refusal**: If evidence is absent, output the mandated refusal phrase:
   > *"The provided regulatory context does not contain sufficient information to answer this question."*
4. **Professional & Risk-Aware Tone**: Formatted with a direct topic sentence and actionable compliance obligations.

---

## 4. Full Augmented Prompt Preview (Injected into Model)

### A. System Message (`role: 'system'`):
```text
You are RegulSense, an AI Senior Regulatory Compliance Specialist for a commercial bank.
Your sole mission is to answer regulatory compliance questions with absolute accuracy, zero hallucination, and full evidentiary auditability.

CORE GROUNDING RULES:
1. STRICT CONTEXT BOUNDARY:
   - Answer the user's question relying SOLELY and EXCLUSIVELY on the regulatory evidence provided in the "RETRIEVED REGULATORY CONTEXT" section below.
   - Do NOT extrapolate, speculate, or introduce external legal knowledge, general banking industry practices, or unstated assumptions not explicitly documented in the provided excerpts.

2. MANDATORY SOURCE CITATIONS:
   - For every factual assertion, timeline, metric, or requirement in your answer, you MUST cite the corresponding source marker (e.g. [1], [2], or the exact circular name and section).
   - If multiple sources support a point, cite all relevant markers (e.g. [1], [2]).

3. INSUFFICIENT CONTEXT FALLBACK PROTOCOL:
   - If the provided context does NOT contain sufficient, explicit evidence to answer the question with complete certainty, do NOT attempt to answer or guess.
   - You MUST respond with this exact statement:
     "The provided regulatory context does not contain sufficient information to answer this question."
   - You may briefly specify what regulatory information or circular is missing.

4. RESPONSE CONSTRAINTS:
   - Maintain a strictly professional, objective, risk-aware tone.
   - Begin with a direct 1-sentence answer citing the primary source marker.
   - Follow with concise bullet points detailing operative compliance obligations, quoting rule text where appropriate.
```

### B. User Message with Injected Context (`role: 'user'`):
```text
--- RETRIEVED REGULATORY CONTEXT ---
SOURCE [1]: cyber_resilience_framework.pdf
Section: Preamble / Document Header | Page: 1 | Chunk ID: cyber_resilience_framework_pdf_tokenaware_001 | Similarity: 0.6436
Verbatim Regulatory Content:
RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls 1. Mandatory Two-Factor Authentication (2FA) All regulated payment system operators and scheduled commercial banks must enforce dynamic two-factor authentication (2FA) for all domestic electronic fund transfers (NEFT, RTGS, IMPS, and UPI). At least one factor must be dynamic, such as a time-based one-time password (TOTP) or biometric verification. 2. Incident Reporting Timelines (6-Hour Rule) Any cyber security incident, ransomware compromise, unauthorized system intrusion, or major denial of service (DoS) affecting customer-facing channels must be reported to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection. Initial reports must be followed by a comprehensive forensic analysis report within 7 business days. 3. Security Operations Centre (SOC) Operations Banks must operate a 24x7x365 Security Operations Centre (SOC) equipped with continuous automated log monitoring, SIEM analytics, and automated threat hunting capabilities.

4. API Security and Third-Party Risk Management
(a) All open banking and fintech integrations must mandate

----------------------------------------

SOURCE [2]: circular_dor_2024_108.txt
Section: 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | Page: 1 | Chunk ID: circular_dor_2024_108_txt_tokenaware_003 | Similarity: 0.5709
Verbatim Regulatory Content:
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

----------------------------------------

SOURCE [3]: sample_regulatory_circular.txt
Section: 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | Page: 1 | Chunk ID: sample_regulatory_circular_txt_tokenaware_003 | Similarity: 0.5709
Verbatim Regulatory Content:
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
--- END REGULATORY CONTEXT ---

COMPLIANCE INQUIRY:
What are the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents?

Please provide a grounded compliance answer adhering strictly to the source markers and grounding rules above.
```

---

## 5. Live Grounded Model Completion (Verification of Citations & Grounding)

### **Model Generated Answer**:
Based on the provided regulatory context, the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents are as follows:

**Initial Report:** Within 6 hours of detection, banks must report the incident to the RBI Cyber Security Cell (CSITE) and CERT-In. [1] Section 2. Incident Reporting Timelines (6-Hour Rule)

**Comprehensive Forensic Analysis Report:** Initial reports must be followed by a comprehensive forensic analysis report within 7 business days. [1] Section 2. Incident Reporting Timelines (6-Hour Rule)

Please note that these reporting procedures are specific to Severity 1 cyber security incidents, and banks must adhere to these timelines and procedures to ensure compliance with the regulatory requirements.

> [!NOTE]
> Notice how the model directly references source markers `[1]` and verbatim provisions from the injected evidence.

---

## 6. Edge-Case Verification: Insufficient Context Fallback Protocol

### **Out-of-Corpus Query**: *"What are the Basel III Common Equity Tier 1 (CET1) capital buffer ratios for regional rural banks?"*
### **Model Fallback Response**:
I'm unable to provide an answer as there is no retrieved regulatory context available. The provided context is insufficient.

According to the Grounding Rules, I must respond with the following statement:

"The provided regulatory context does not contain sufficient information to answer this question."

Please provide the relevant regulatory context, and I'll be happy to assist you with a grounded compliance answer.

> [!TIP]
> The model correctly invoked the insufficient-context protocol rather than hallucinating speculative answers.

---
*Report automatically generated by `src/context_assembler.py` for RegulSense Banking Compliance Assistant.*