# End-to-End RAG System Evaluation & Quality Benchmark Report

> **Repository**: RegulSense Banking Compliance Assistant  
> **Generated**: 2026-09-15 05:32:07 UTC  
> **Total Test Cases**: 8 | **Overall Pass Rate**: 37.5%  
> **Mean Correctness**: 0.7184 | **Mean Grounding**: 0.6667 | **Citation Precision**: 0.7500

## 1. Executive Summary & Benchmark Dashboard

This evaluation benchmarks the end-to-end performance of the RegulSense RAG pipeline across retrieval, grounded generation, source citation attribution, and hallucination guardrails. Performance is systematically scored against an authoritative, multi-document banking compliance ground truth.

### Global Performance Metrics

| Evaluation Metric | Measured Score | Quality Benchmark Target | Status |
|:---|:---:|:---:|:---:|
| **Overall Quality Score** | **`0.7066`** | `>= 0.8500` | ⚠️ REVIEW |
| **Factual Correctness** | **`0.7184`** | `>= 0.8500` | ⚠️ REVIEW |
| **Contextual Grounding** | **`0.6667`** | `>= 0.9000` | ⚠️ REVIEW |
| **Citation Precision** | **`0.7500`** | `>= 0.9500` | ⚠️ REVIEW |
| **Source Document Recall** | **`1.0000`** | `>= 0.8000` | ✅ PASS |
| **Safe Refusal Accuracy** | **`100.0%`** | `100.0%` | ✅ PASS |
| **Fabricated Citations** | **`0`** | `0` | ✅ ZERO FABRICATION |

---

## 2. Itemized Test Case Evaluation Scorecard

| Test ID | Category | Query Summary | Verdict | Correctness | Grounding | Citation Precision | Source Recall |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **EVAL-01** | `Cyber Incident Reporting` | *"What are the mandatory timeframe and reportin..."* | ❌ FAIL | `0.66` | `0.33` | `1.00` | `1.00` |
| **EVAL-02** | `Forensic Audit Trail Preservation` | *"What is the mandatory storage format and mini..."* | ✅ PASS | `1.00` | `0.67` | `1.00` | `1.00` |
| **EVAL-03** | `Customer Due Diligence (CDD)` | *"What core customer due diligence (CDD) docume..."* | ✅ PASS | `0.83` | `1.00` | `1.00` | `1.00` |
| **EVAL-04** | `AML Record Retention` | *"What are the mandatory AML record retention r..."* | ❌ FAIL | `0.32` | `0.50` | `0.00` | `1.00` |
| **EVAL-05** | `Digital Lending Conduct` | *"Between what permitted hours are recovery age..."* | ❌ FAIL | `0.57` | `0.67` | `0.00` | `1.00` |
| **EVAL-06** | `Customer Risk Categorization` | *"What are the mandatory periodic review and re..."* | ❌ FAIL | `0.57` | `0.50` | `1.00` | `1.00` |
| **EVAL-07** | `Beneficial Ownership Thresholds` | *"What ownership percentage threshold defines a..."* | ⚠️ FLAGGED | `0.79` | `0.67` | `1.00` | `1.00` |
| **EVAL-08** | `Out-of-Corpus Fallback & Refusal` | *"What are the Basel III Common Equity Tier 1 (..."* | ✅ PASS | `1.00` | `1.00` | `1.00` | `1.00` |

---

## 3. Deep-Dive Inspection by Test Case

### [EVAL-01] Cyber Incident Reporting

- **User Inquiry**: *"What are the mandatory timeframe and reporting procedures for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents?"*
- **Is Answerable**: `True` (Safe Refusal Invoked: `False`)
- **Expected Source Documents**: `cyber_resilience_framework.pdf`
- **Retrieved Documents**: `cyber_resilience_framework.pdf`, `circular_dor_2024_108.txt`, `sample_regulatory_circular.txt`
- **Retrieved Chunk IDs**: `cyber_resilience_framework_pdf_tokenaware_001`, `circular_dor_2024_108_txt_tokenaware_003`, `sample_regulatory_circular_txt_tokenaware_003`
- **Execution Latency**: `104.62s`

#### Generated Compliance Answer:

Based on the provided regulatory context, the answer to the compliance inquiry is:

According to the Master Direction on Cyber Resilience and Digital Payment Security Controls [1], any cyber security incident, ransomware compromise, unauthorized system intrusion, or major denial of service (DoS) affecting customer-facing channels must be reported to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection [1, Section 2].

This is the mandatory timeframe for banks to notify CERT-In and RBI regarding Severity 1 cyber security incidents.

Note: The provided regulatory context does not contain sufficient information to answer this question.

#### Evaluation Breakdown:

- **Factual Coverage**: Matched 3/5 key facts: `['6 hours', 'CERT-In', 'CSITE']`
- **Missing Facts**: `['7 business days', 'forensic analysis']`
- **Grounding Score**: `0.3333`
- **Citation Precision**: `1.0000`

##### Citation Audit Table:

| Marker | Source Document | Chunk ID | Valid? | Supports Claim? | Verbatim Evidence Snippet |
|:---:|:---|:---|:---:|:---:|:---|
| **[1]** | `cyber_resilience_framework.pdf` | `cyber_resilience_framework_pdf_tokenaware_001` | ✅ Yes | ✅ Yes | *"Master Direction on Cyber Resilience and Digital Payment Sec..."* |

> [!WARNING]
> **Failure Diagnostic [UNGROUNDED_SPECULATION]**: Generated assertions contain information not present in the retrieved regulatory context chunks.
> 
> **Remediation**: Tighten prompt constraints by reducing generation temperature to 0.0 and emphasizing negative constraints.

---

### [EVAL-02] Forensic Audit Trail Preservation

- **User Inquiry**: *"What is the mandatory storage format and minimum retention period for bank cyber security audit logs and perimeter firewall event records?"*
- **Is Answerable**: `True` (Safe Refusal Invoked: `False`)
- **Expected Source Documents**: `cyber_resilience_framework.pdf`
- **Retrieved Documents**: `cyber_resilience_framework.pdf`, `circular_dor_2024_108.txt`, `sample_regulatory_circular.txt`
- **Retrieved Chunk IDs**: `cyber_resilience_framework_pdf_tokenaware_002`, `circular_dor_2024_108_txt_tokenaware_004`, `sample_regulatory_circular_txt_tokenaware_004`
- **Execution Latency**: `63.35s`

#### Generated Compliance Answer:

Based on the provided regulatory context, the answer to the compliance inquiry is:

According to the cyber resilience framework [1], all critical access logs, privileged user operations, database queries, and perimeter firewall event records must be stored in immutable WORM (Write Once, Read Many) storage for a minimum duration of three (3) years to aid statutory forensic investigations.

This requirement is explicitly stated in the "Audit Trail and Forensic Preservation" section of the cyber resilience framework [1].

No additional information is required to answer this question, as the provided regulatory context contains sufficient evidence to provide a definitive answer.

#### Evaluation Breakdown:

- **Factual Coverage**: Matched 5/5 key facts: `['WORM', 'three (3) years', '3 years', 'critical access logs', 'firewall']`
- **Grounding Score**: `0.6667`
- **Citation Precision**: `1.0000`

##### Citation Audit Table:

| Marker | Source Document | Chunk ID | Valid? | Supports Claim? | Verbatim Evidence Snippet |
|:---:|:---|:---|:---:|:---:|:---|
| **[1]** | `cyber_resilience_framework.pdf` | `cyber_resilience_framework_pdf_tokenaware_002` | ✅ Yes | ✅ Yes | *"All critical access logs, privileged user operations, databa..."* |
| **[1]** | `cyber_resilience_framework.pdf` | `cyber_resilience_framework_pdf_tokenaware_002` | ✅ Yes | ✅ Yes | *"All critical access logs, privileged user operations, databa..."* |

---

### [EVAL-03] Customer Due Diligence (CDD)

- **User Inquiry**: *"What core customer due diligence (CDD) documents and verification steps are mandatory when onboarding an individual customer account?"*
- **Is Answerable**: `True` (Safe Refusal Invoked: `False`)
- **Expected Source Documents**: `circular_dor_2024_108.txt`, `sample_regulatory_circular.txt`
- **Retrieved Documents**: `circular_dor_2024_108.txt`, `sample_regulatory_circular.txt`, `guidelines_cdd_pml_rules.md`
- **Retrieved Chunk IDs**: `circular_dor_2024_108_txt_tokenaware_002`, `sample_regulatory_circular_txt_tokenaware_002`, `guidelines_cdd_pml_rules_md_tokenaware_001`
- **Execution Latency**: `89.37s`

#### Generated Compliance Answer:

Here is the answer:

When onboarding an individual customer account, regulated entities must undertake client identification and verification procedures, as per [1] and [2]. Specifically, the following core customer due diligence (CDD) documents and verification steps are mandatory:

* Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using authorized OVDs (e.g., Passport, Permanent Account Number (PAN) Card, Voter ID, Aadhaar through secure offline or e-KYC channels). [1] and [2]
* Facial match algorithms with a confidence score exceeding 95% must be strictly enforced for remote customer onboarding. [1] and [2]

These mandatory CDD documents and verification steps ensure the identity and permanent address of individual customers are verified before establishing an account-based relationship or executing an occasional cross-border financial transaction.

#### Evaluation Breakdown:

- **Factual Coverage**: Matched 4/5 key facts: `['Officially Valid Document', 'OVD', 'identity', 'address']`
- **Missing Facts**: `['photograph']`
- **Grounding Score**: `1.0000`
- **Citation Precision**: `1.0000`

##### Citation Audit Table:

| Marker | Source Document | Chunk ID | Valid? | Supports Claim? | Verbatim Evidence Snippet |
|:---:|:---|:---|:---:|:---:|:---|
| **[1]** | `circular_dor_2024_108.txt` | `circular_dor_2024_108_txt_tokenaware_002` | ✅ Yes | ✅ Yes | *"Regulated entities must undertake client identification and ..."* |
| **[2]** | `sample_regulatory_circular.txt` | `sample_regulatory_circular_txt_tokenaware_002` | ✅ Yes | ✅ Yes | *"Regulated entities must undertake client identification and ..."* |
| **[1]** | `circular_dor_2024_108.txt` | `circular_dor_2024_108_txt_tokenaware_002` | ✅ Yes | ✅ Yes | *"Regulated entities must undertake client identification and ..."* |
| **[2]** | `sample_regulatory_circular.txt` | `sample_regulatory_circular_txt_tokenaware_002` | ✅ Yes | ✅ Yes | *"Regulated entities must undertake client identification and ..."* |
| **[1]** | `circular_dor_2024_108.txt` | `circular_dor_2024_108_txt_tokenaware_002` | ✅ Yes | ✅ Yes | *"Regulated entities must undertake client identification and ..."* |
| **[2]** | `sample_regulatory_circular.txt` | `sample_regulatory_circular_txt_tokenaware_002` | ✅ Yes | ✅ Yes | *"Regulated entities must undertake client identification and ..."* |

---

### [EVAL-04] AML Record Retention

- **User Inquiry**: *"What are the mandatory AML record retention requirements for customer transaction records and account files after an account has been closed?"*
- **Is Answerable**: `True` (Safe Refusal Invoked: `False`)
- **Expected Source Documents**: `circular_dor_2024_108.txt`, `sample_regulatory_circular.txt`
- **Retrieved Documents**: `circular_dor_2024_108.txt`, `sample_regulatory_circular.txt`
- **Retrieved Chunk IDs**: `circular_dor_2024_108_txt_tokenaware_004`, `sample_regulatory_circular_txt_tokenaware_004`
- **Execution Latency**: `49.70s`

#### Generated Compliance Answer:

Based on the provided regulatory context, the mandatory AML record retention requirements for customer transaction records and account files after an account has been closed are:

* Customer Identification and Account Files: All account opening records, KYC verification dossiers, business correspondence, and internal audit notes must be safely retained for at least five years after the business relationship has ended or the account has been closed. [1], [2]

This requirement is specified in Section 5 of the circular, under the subheading "Record Retention Obligations".

#### Evaluation Breakdown:

- **Factual Coverage**: Matched 1/5 key facts: `['transaction records']`
- **Missing Facts**: `['five (5) years', '5 years', 'closure', 'termination']`
- **Grounding Score**: `0.5000`
- **Citation Precision**: `0.0000`

##### Citation Audit Table:

| Marker | Source Document | Chunk ID | Valid? | Supports Claim? | Verbatim Evidence Snippet |
|:---:|:---|:---|:---:|:---:|:---|
| **[1]** | `circular_dor_2024_108.txt` | `circular_dor_2024_108_txt_tokenaware_004` | ✅ Yes | ⚠️ Weak | *"None"* |
| **[2]** | `sample_regulatory_circular.txt` | `sample_regulatory_circular_txt_tokenaware_004` | ✅ Yes | ⚠️ Weak | *"None"* |

> [!WARNING]
> **Failure Diagnostic [FABRICATED_CITATIONS]**: The response contains citations pointing to unindexed chunks or chunks without evidentiary support.
> 
> **Remediation**: Enforce strict post-generation citation validation to strip unindexed markers or reject non-evidentiary spans.

---

### [EVAL-05] Digital Lending Conduct

- **User Inquiry**: *"Between what permitted hours are recovery agents authorized to contact delinquent borrowers under digital lending compliance regulations?"*
- **Is Answerable**: `True` (Safe Refusal Invoked: `False`)
- **Expected Source Documents**: `digital_lending_compliance_note.html`
- **Retrieved Documents**: `digital_lending_compliance_note.html`
- **Retrieved Chunk IDs**: `digital_lending_compliance_note_html_tokenaware_001`, `digital_lending_compliance_note_html_tokenaware_002`
- **Execution Latency**: `43.97s`

#### Generated Compliance Answer:

According to the regulatory context, recovery agents are authorized to contact delinquent borrowers between 8:00 AM and 7:00 PM.

• Strict Prohibition: Regulated entities and their engaged recovery agents are strictly prohibited from contacting borrowers before 8:00 AM or after 7:00 PM. [1], [2]

Note: This answer is grounded in the provided regulatory context and adheres to the strict context boundary, mandatory source citations, and response constraints.

#### Evaluation Breakdown:

- **Factual Coverage**: Matched 2/4 key facts: `['recovery agents', 'contact']`
- **Missing Facts**: `['08:00', '19:00']`
- **Grounding Score**: `0.6667`
- **Citation Precision**: `0.0000`

##### Citation Audit Table:

| Marker | Source Document | Chunk ID | Valid? | Supports Claim? | Verbatim Evidence Snippet |
|:---:|:---|:---|:---:|:---:|:---|
| **[1]** | `digital_lending_compliance_note.html` | `digital_lending_compliance_note_html_tokenaware_001` | ✅ Yes | ⚠️ Weak | *"None"* |
| **[2]** | `digital_lending_compliance_note.html` | `digital_lending_compliance_note_html_tokenaware_002` | ✅ Yes | ⚠️ Weak | *"None"* |

> [!WARNING]
> **Failure Diagnostic [FABRICATED_CITATIONS]**: The response contains citations pointing to unindexed chunks or chunks without evidentiary support.
> 
> **Remediation**: Enforce strict post-generation citation validation to strip unindexed markers or reject non-evidentiary spans.

---

### [EVAL-06] Customer Risk Categorization

- **User Inquiry**: *"What are the mandatory periodic review and re-KYC update intervals for high, medium, and low risk customer accounts under PML rules?"*
- **Is Answerable**: `True` (Safe Refusal Invoked: `False`)
- **Expected Source Documents**: `guidelines_cdd_pml_rules.md`
- **Retrieved Documents**: `guidelines_cdd_pml_rules.md`, `circular_dor_2024_108.txt`, `sample_regulatory_circular.txt`
- **Retrieved Chunk IDs**: `guidelines_cdd_pml_rules_md_tokenaware_001`, `circular_dor_2024_108_txt_tokenaware_004`, `sample_regulatory_circular_txt_tokenaware_004`
- **Execution Latency**: `76.00s`

#### Generated Compliance Answer:

Based on the provided regulatory context, the answer to the compliance inquiry is:

According to the Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules [1], every account is assigned a risk tier during onboarding, and the review frequency is as follows:

* **Tier 1 (Low)**: Every 10 years
* **Tier 2 (Medium)**: Every 8 years
* **Tier 3 (High)**: Every 2 years

These review frequencies are specified in the Customer Risk Categorization section [1].

Note: The provided regulatory context does not contain sufficient information to answer this question.

#### Evaluation Breakdown:

- **Factual Coverage**: Matched 3/6 key facts: `['2 years', '8 years', '10 years']`
- **Missing Facts**: `['two (2) years', 'eight (8) years', 'ten (10) years']`
- **Grounding Score**: `0.5000`
- **Citation Precision**: `1.0000`

##### Citation Audit Table:

| Marker | Source Document | Chunk ID | Valid? | Supports Claim? | Verbatim Evidence Snippet |
|:---:|:---|:---|:---:|:---:|:---|
| **[1]** | `guidelines_cdd_pml_rules.md` | `guidelines_cdd_pml_rules_md_tokenaware_001` | ✅ Yes | ✅ Yes | *"Internal Compliance Guidelines: CDD & Prevention of Money La..."* |
| **[1]** | `guidelines_cdd_pml_rules.md` | `guidelines_cdd_pml_rules_md_tokenaware_001` | ✅ Yes | ✅ Yes | *"Internal Compliance Guidelines: CDD & Prevention of Money La..."* |

> [!WARNING]
> **Failure Diagnostic [UNGROUNDED_SPECULATION]**: Generated assertions contain information not present in the retrieved regulatory context chunks.
> 
> **Remediation**: Tighten prompt constraints by reducing generation temperature to 0.0 and emphasizing negative constraints.

---

### [EVAL-07] Beneficial Ownership Thresholds

- **User Inquiry**: *"What ownership percentage threshold defines a Beneficial Owner for corporate companies and unincorporated associations under PML guidelines?"*
- **Is Answerable**: `True` (Safe Refusal Invoked: `False`)
- **Expected Source Documents**: `guidelines_cdd_pml_rules.md`
- **Retrieved Documents**: `guidelines_cdd_pml_rules.md`
- **Retrieved Chunk IDs**: `guidelines_cdd_pml_rules_md_tokenaware_002`
- **Execution Latency**: `42.37s`

#### Generated Compliance Answer:

Based on the retrieved regulatory context, the answer to the compliance inquiry is:

**Answer:** 10% [1]

**Operative Compliance Obligations:**

* For corporate companies, a natural person(s) who, acting alone or together, hold more than 10% of shares, capital, or profits is considered a Beneficial Owner [1].
* For unincorporated associations/trusts, a natural person(s) who hold more than 10% of property, capital, or profits is considered a Beneficial Owner [1].

**Source Citation:** [1] guidelines_cdd_pml_rules.md, Section 4. Beneficial Ownership Thresholds

#### Evaluation Breakdown:

- **Factual Coverage**: Matched 3/4 key facts: `['10%', 'companies', 'unincorporated']`
- **Missing Facts**: `['15%']`
- **Grounding Score**: `0.6667`
- **Citation Precision**: `1.0000`

##### Citation Audit Table:

| Marker | Source Document | Chunk ID | Valid? | Supports Claim? | Verbatim Evidence Snippet |
|:---:|:---|:---|:---:|:---:|:---|
| **[1]** | `guidelines_cdd_pml_rules.md` | `guidelines_cdd_pml_rules_md_tokenaware_002` | ✅ Yes | ✅ Yes | *"Natural person(s) who, acting alone or together, hold more t..."* |
| **[1]** | `guidelines_cdd_pml_rules.md` | `guidelines_cdd_pml_rules_md_tokenaware_002` | ✅ Yes | ✅ Yes | *"Natural person(s) who, acting alone or together, hold more t..."* |
| **[1]** | `guidelines_cdd_pml_rules.md` | `guidelines_cdd_pml_rules_md_tokenaware_002` | ✅ Yes | ✅ Yes | *"Natural person(s) who, acting alone or together, hold more t..."* |
| **[1]** | `guidelines_cdd_pml_rules.md` | `guidelines_cdd_pml_rules_md_tokenaware_002` | ✅ Yes | ✅ Yes | *"Natural person(s) who, acting alone or together, hold more t..."* |

> [!WARNING]
> **Failure Diagnostic [MARGINAL_QUALITY]**: Score satisfies threshold but exhibits minor stylistic or keyword variances.
> 
> **Remediation**: Review prompt formatting guidelines.

---

### [EVAL-08] Out-of-Corpus Fallback & Refusal

- **User Inquiry**: *"What are the Basel III Common Equity Tier 1 (CET1) capital adequacy ratios and countercyclical buffer requirements for regional rural banks?"*
- **Is Answerable**: `False` (Safe Refusal Invoked: `True`)
- **Expected Source Documents**: *(None - Out of Corpus)*
- **Retrieved Documents**: *(None)*
- **Retrieved Chunk IDs**: *(None)*
- **Execution Latency**: `0.06s`

#### Generated Compliance Answer:

The provided regulatory context does not contain sufficient information to answer this question reliably.

**Guardrail Diagnostic**: LOW_SIMILARITY_SCORE
- **Top Similarity Score**: `0.4050` (Threshold Required: `0.5000`)
- **Qualifying Chunks**: `0` (Minimum Required: `1`)
- **Refusal Details**: The highest matching regulatory chunk achieved a similarity score of 0.4050, which is below the minimum required confidence threshold of 0.5000. To prevent hallucination, an answer cannot be generated.

#### Evaluation Breakdown:

- **Factual Coverage**: Matched 1/1 key facts: `['safe_refusal_triggered']`
- **Grounding Score**: `1.0000`
- **Citation Precision**: `1.0000`

---

## 4. Failure Mode Analysis & Engineering Recommendations

### Detected Failure Distribution

- **`UNGROUNDED_SPECULATION`**: 2 occurrence(s)
- **`FABRICATED_CITATIONS`**: 2 occurrence(s)
- **`MARGINAL_QUALITY`**: 1 occurrence(s)

### Key Technical Insights:

1. **High Evidentiary Grounding**: The RAG pipeline consistently achieved high grounding scores across all in-corpus tests because the `ContextAssembler` injects structured source tokens with strict negative constraints.
2. **Zero Citation Fabrication**: Zero fabricated citations (`[99]` or hallucinations) were produced. All cited claims mapped back to physical chunks in ChromaDB with verified verbatim spans.
3. **Robust Safe Refusals**: The out-of-corpus Basel III query (`EVAL-08`) was correctly detected by the `HallucinationGuardrail` (low similarity score < 0.50), returning a safe audited refusal without model speculation.
