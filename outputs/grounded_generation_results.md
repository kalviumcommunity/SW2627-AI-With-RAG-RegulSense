# RegulSense: Grounded Answer Generation, Source Accuracy & Retrieval Comparison Report

- **Model**: `llama3:latest` (Sampling Temperature: `0.2`)
- **Timestamp**: `2026-09-14T15:51:25.505122+00:00`
- **Claim Fidelity Score**: `100.0%` (PASS)
- **Audit Status**: `✅ PASSED (100% Grounded)`

---

## 1. Grounded Answer Generated from Injected Context (Task 1)

### **Compliance Inquiry**:
> *"What are the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents?"*

### **Grounded Model Response**:
Based on the provided regulatory context, the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents are as follows:

* Initial report: Within 6 hours of detection, banks must report the incident to the RBI Cyber Security Cell (CSITE) and CERT-In. [1] Section 2, Incident Reporting Timelines (6-Hour Rule)
* Comprehensive forensic analysis report: Within 7 business days, banks must follow up with a comprehensive forensic analysis report to the RBI Cyber Security Cell (CSITE) and CERT-In. [1] Section 2, Incident Reporting Timelines (6-Hour Rule)

Please note that these reporting procedures are specific to Severity 1 cyber security incidents and may not apply to other types of incidents or events.

#### **Supporting Retrieved Chunks**:
| Marker | Source Document | Section | Similarity | Tokens |
| :---: | :--- | :--- | :---: | :---: |
| **`[1]`** | `cyber_resilience_framework.pdf` | Preamble / Document Header | `0.6436` | 352 |
| **`[2]`** | `circular_dor_2024_108.txt` | 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | `0.5709` | 375 |
| **`[3]`** | `sample_regulatory_circular.txt` | 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs | `0.5709` | 369 |

---

## 2. Source Accuracy & Claim Verification Audit (Task 2)

- **Verdict**: `PASS`
- **Fidelity Score**: **`100.0%`**
- **Source Citations Count**: `2` (['[1]'])

| Audit Criterion | Finding | Evaluation |
| :--- | :--- | :---: |
| **Mandatory Citation Presence** | Cited markers: `['[1]']` | ✅ PASS |
| **Verified Factual Claims** | `6` claims verified against chunk text: 7 business days, cert-in, severity 1 (from query), csite, rbi, 6 hours... | ✅ PASS |
| **Unsupported Assertions** | `0` ungrounded claims detected | ✅ NONE |
| **Overall Claim Fidelity** | Fidelity score: 100.0%. Verified 6 factual claims against source text. Found 2 source citations. | ✅ VERIFIED |

---

## 3. Missing-Context Fallback Protocol Demonstration (Task 3)

### **Out-of-Corpus Query**:
> *"What are the Basel III Common Equity Tier 1 (CET1) capital adequacy ratio and countercyclical buffer requirements for regional rural banks?"*

### **Model Fallback Response**:
I'm unable to provide a compliance answer as the retrieved regulatory context is insufficient. The provided context does not contain sufficient information to answer this question.

The lack of regulatory context makes it impossible to determine the specific requirements for regional rural banks regarding Basel III Common Equity Tier 1 (CET1) capital adequacy ratio and countercyclical buffer.

> [!TIP]
> **Refusal Verification**: The model strictly adhered to the fallback protocol, stating that the context was insufficient and refusing to speculate on unindexed Basel III capital ratios.

---

## 4. Side-by-Side Comparison: With Retrieval vs. Without Retrieval (Task 4)

### **Test Question**: *"What are the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents?"*

| Evaluation Aspect | With Retrieval (Grounded RAG) | Without Retrieval (Baseline LLM) |
| :--- | :--- | :--- |
| **Generated Answer** | Based on the provided regulatory context, the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents are as follows:

**Initial Report:** Any cyber security incident, ransomware compromise, unauthorized syste... | As a bank staff, you're concerned about the timely reporting of cyber security incidents. According to the guidelines, here are the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents:

**Mandatory Timefra... |
| **Regulatory Citations** | `['[1]']` (Explicit markers) | None (No evidentiary citations) |
| **Statutory 6-Hour Rule** | **Present** (Mandated 6-hour CSITE window) | Present |
| **Circular Reference** | RBI Master Direction on Cyber Resilience | Vague / General Advice |
| **Word Count** | `142` words | `287` words |
| **Latency** | `92.91s` | `61.13s` |

### **Hallucination & Legal Risk Assessment**:
> LOW RISK: The baseline captured core concepts, but lacks traceable evidentiary markers.

### **Comparison Conclusion**:
> Retrieval grounding transformed the response from generic advice into an audit-ready compliance determination with 1 verifiable source citations, exact circular references, and strict fidelity to RBI Master Directions.

---
*Report automatically generated by `src/grounded_generator.py` for RegulSense Banking Compliance Assistant.*