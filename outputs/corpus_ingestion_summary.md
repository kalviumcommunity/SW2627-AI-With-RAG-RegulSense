# RegulSense: Full Corpus Ingestion & Completeness Audit Report

- **Execution Timestamp**: 2026-09-08 12:42:00
- **Target Corpus Root**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data`
- **Pipeline Stages**: Discovery -> Guarded Loading -> Text Cleaning -> Token Chunking -> Metadata Tagging -> Reconciliation
- **Audit Status**: **PASSED (100% RECONCILED — ZERO SILENT DROPS)**

---

## 1. Executive Ingestion Summary

The complete ingestion pipeline was executed end-to-end across the entire document corpus. Every discovered file was evaluated, loaded through multi-format extractors, sanitized by the retrieval-ready text cleaner, chunked into token-budgeted slices with controlled overlap, and tagged with uniform metadata.

| Ingestion Metric | Value | Audit Verification |
| :--- | :--- | :--- |
| **Total Discovered Files** | `6` | 100% scanned across root & subdirectories |
| **Successfully Ingested Documents** | `5` | Loaded, cleaned, chunked, and tagged |
| **Recorded Skipped / Non-Document Files** | `1` | Explicitly logged with diagnostic codes |
| **Unaccounted Silently Dropped Files** | `0` | **0 (Zero Silent Drops Verified)** |
| **Completeness Reconciliation** | `PASSED` | $\text{Discovered} = \text{Ingested} + \text{Skipped}$ |
| **Total Chunks Created** | `15` | Sized by token budget with controlled overlap |
| **Total Corpus Plain-Text Tokens** | `3,824` | Tiktoken `cl100k_base` encoding |
| **Raw Extracted Characters** | `15,769` | Prior to normalization |
| **Cleaned Characters** | `15,677` | After Unicode, ligature & boilerplate healing |
| **Boilerplate & Artifacts Removed** | `92` | `0.58%` reduction |
| **Metadata Schema Compliance** | `100.0%` | All 13 required fields validated |

---

## 2. Mathematical Completeness Reconciliation (Task 3)

To guarantee that no document is silently skipped when moving from a sample to the full corpus, the pipeline enforces strict mathematical reconciliation:

$$\text{Total Discovered} (6) = \text{Successfully Ingested} (5) + \text{Recorded Non-Documents/Skipped} (1)$$

$$\text{Unaccounted Documents} = 6 - (5 + 1) = 0$$

### Ingestion Proof Ledger

| File Path | Status | Reason / Diagnostic Code | Stage |
| :--- | :--- | :--- | :--- |
| `data\sample_corpus\circular_dor_2024_108.txt` | **INGESTED** | Valid regulatory document (.txt) -> 4 chunks | TAGGED |
| `data\sample_corpus\cyber_resilience_framework.pdf` | **INGESTED** | Valid regulatory document (.pdf) -> 2 chunks | TAGGED |
| `data\sample_corpus\digital_lending_compliance_note.html` | **INGESTED** | Valid regulatory document (.html) -> 2 chunks | TAGGED |
| `data\sample_corpus\guidelines_cdd_pml_rules.md` | **INGESTED** | Valid regulatory document (.md) -> 3 chunks | TAGGED |
| `data\sample_regulatory_circular.txt` | **INGESTED** | Valid regulatory document (.txt) -> 4 chunks | TAGGED |
| `.gitkeep` | **SKIPPED** | `UNSUPPORTED_FORMAT`: Non-document file or unsupported extension '' | LOADING |

> [!NOTE]
> `.gitkeep` is a repository placeholder without file extension. The pipeline detected it, isolated it from the document loader, and explicitly recorded its skipped status without dropping it silently or halting the pipeline.

---

## 3. Document-by-Document Ingestion Details (Task 2)

| Document Filename | Format | Raw Chars | Clean Chars | Chars Removed | Reduction % | Words | Tokens | Chunks |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `circular_dor_2024_108.txt` | `.txt` | 4,747 | 4,747 | 0 | 0.0% | 647 | 1,131 | **4** |
| `cyber_resilience_framework.pdf` | `.pdf` | 2,167 | 2,167 | 0 | 0.0% | 292 | 506 | **2** |
| `digital_lending_compliance_note.html` | `.html` | 1,708 | 1,624 | 84 | 4.92% | 222 | 387 | **2** |
| `guidelines_cdd_pml_rules.md` | `.md` | 2,400 | 2,392 | 8 | 0.33% | 349 | 669 | **3** |
| `sample_regulatory_circular.txt` | `.txt` | 4,747 | 4,747 | 0 | 0.0% | 647 | 1,131 | **4** |

---

## 4. Chunking Statistics & Token Budgeting

- **Chunking Strategy**: `TokenAwareChunker` (`cl100k_base` tokenizer)
- **Target Token Size**: `300` tokens
- **Controlled Token Overlap**: `50` tokens
- **Minimum Chunk Tokens**: `69` tokens
- **Maximum Chunk Tokens**: `300` tokens (strictly $\le 300$)
- **Average Chunk Tokens**: `254.9` tokens
- **Total Chunks Produced**: `15` chunks

---

## 5. Sample Chunk Inspection & Provenance Verification (Task 4)

Representative chunks from each distinct format across the corpus demonstrate cleaned text, sensible boundaries, source identifiers, and uniform metadata tags.

### Sample 1: `circular_dor_2024_108.txt` (.txt) — Chunk 1 of 4

- **Chunk ID**: `circular_dor_2024_108_txt_tokenaware_001`
- **Source Identifier**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\circular_dor_2024_108.txt`
- **Active Section**: `Preamble / Document Header`
- **Document Page**: `1`
- **Character Span**: `[0, 1293]` (1293 chars)
- **Token Count**: `300` tokens (Tokens [0:300])
- **Controlled Overlap**: `50` tokens
- **Provenance Trace Status**: `VERIFIED` (Similarity: 1.0)
- **Standard Citation**: `[Source: circular_dor_2024_108.txt, Chunk: 1/4, Page: 1]`

**Cleaned Chunk Text Content**:
```text
RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTMENT CENTRAL OFFICE, MUMBAI

Ref: RBI/2023-24/108 Circular No: DOR.AML.REC.66/14.01.001/2023-24 Date: January 04, 2024

To, All Scheduled Commercial Banks (excluding RRBs) All Small Finance Banks and Payment Banks All Non-Banking Financial Companies (NBFCs)

Subject: Master Direction - Know Your Customer (KYC) and Anti-Money Laundering (AML) Standards - Enhanced Due Diligence and Transaction Record Management

1. Preliminary and Statutory Authority
In exercise of the powers conferred by Section 35A of the Banking Regulation Act, 1949, read with Section 51A of the Unlawful Activities (Prevention) Act, 1967, and the Prevention of Money-Laundering (Maintenance of Records) Rules, 2005, the Reserve Bank of India hereby issues the updated Master Directions on Customer Due Diligence (CDD) and compliance obligations for regulated entities (REs).

2. Customer Due Diligence (CDD) Requirements
Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
(a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using
```

### Sample 2: `cyber_resilience_framework.pdf` (.pdf) — Chunk 1 of 2

- **Chunk ID**: `cyber_resilience_framework_pdf_tokenaware_001`
- **Source Identifier**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\cyber_resilience_framework.pdf`
- **Active Section**: `Preamble / Document Header`
- **Document Page**: `1`
- **Character Span**: `[0, 1345]` (1345 chars)
- **Token Count**: `300` tokens (Tokens [0:300])
- **Controlled Overlap**: `50` tokens
- **Provenance Trace Status**: `VERIFIED` (Similarity: 1.0)
- **Standard Citation**: `[Source: cyber_resilience_framework.pdf, Chunk: 1/2, Page: 1]`

**Cleaned Chunk Text Content**:
```text
RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls 1. Mandatory Two-Factor Authentication (2FA) All regulated payment system operators and scheduled commercial banks must enforce dynamic two-factor authentication (2FA) for all domestic electronic fund transfers (NEFT, RTGS, IMPS, and UPI). At least one factor must be dynamic, such as a time-based one-time password (TOTP) or biometric verification. 2. Incident Reporting Timelines (6-Hour Rule) Any cyber security incident, ransomware compromise, unauthorized system intrusion, or major denial of service (DoS) affecting customer-facing channels must be reported to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection. Initial reports must be followed by a comprehensive forensic analysis report within 7 business days. 3. Security Operations Centre (SOC) Operations Banks must operate a 24x7x365 Security Operations Centre (SOC) equipped with continuous automated log monitoring, SIEM analytics, and automated threat hunting capabilities.

4. API Security and Third-Party Risk Management
(a) All open banking and fintech integrations must mandate
```

### Sample 3: `digital_lending_compliance_note.html` (.html) — Chunk 1 of 2

- **Chunk ID**: `digital_lending_compliance_note_html_tokenaware_001`
- **Source Identifier**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\digital_lending_compliance_note.html`
- **Active Section**: `Preamble / Document Header`
- **Document Page**: `1`
- **Character Span**: `[0, 1452]` (1452 chars)
- **Token Count**: `300` tokens (Tokens [0:300])
- **Controlled Overlap**: `50` tokens
- **Provenance Trace Status**: `VERIFIED` (Similarity: 1.0)
- **Standard Citation**: `[Source: digital_lending_compliance_note.html, Chunk: 1/2, Page: 1]`

**Cleaned Chunk Text Content**:
```text
Regulatory Advisory: Digital Lending Guidelines & Consumer Protection

RegulSense Regulatory Advisory: Digital Lending & Fair Practices Code

Reference: RBI/2022-23/DOR.CRE.REC.42/21.04.048/2022-23

Target Entities: Commercial Banks, Digital Lending Platforms, Lending Service Providers (LSPs)

1. Direct Fund Disbursal and Repayment

All loan disbursals and repayments must be executed solely between the bank account of the borrower and the regulated entity (RE) without any pass-through or pool account of any third-party Lending Service Provider (LSP).

2. Key Fact Statement (KFS) Mandate

A standardized Key Fact Statement (KFS) must be provided to the prospective borrower before execution of any loan agreement. The KFS must explicitly disclose:

Annual Percentage Rate (APR) including all fees, processing charges, and penal interests.

Cooling-off / look-up period during which borrowers can exit digital loans without penalty.

Detailed schedule of installments and default handling terms.

3. Code of Conduct for Recovery Agents

Strict Prohibition: Regulated entities and their engaged recovery agents are strictly prohibited from contacting borrowers before 8:00 AM or after 7:00 PM. Harassment, verbal intimidation, or unauthorized contact with borrowers' relatives or social networks will lead to immediate regulatory sanction.

4. Data Privacy and Storage Restrictions

Digital lending apps shall not access mobile phone resources such
```

### Sample 4: `guidelines_cdd_pml_rules.md` (.md) — Chunk 1 of 3

- **Chunk ID**: `guidelines_cdd_pml_rules_md_tokenaware_001`
- **Source Identifier**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\guidelines_cdd_pml_rules.md`
- **Active Section**: `Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules`
- **Document Page**: `1`
- **Character Span**: `[0, 1224]` (1224 chars)
- **Token Count**: `300` tokens (Tokens [0:300])
- **Controlled Overlap**: `50` tokens
- **Provenance Trace Status**: `VERIFIED` (Similarity: 1.0)
- **Standard Citation**: `[Source: guidelines_cdd_pml_rules.md, Chunk: 1/3, Section: 'Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules', Page: 1]`

**Cleaned Chunk Text Content**:
```text
# Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules

**Document Identifier**: REG-COMP-2024-V2
**Effective Date**: January 15, 2024
**Applicability**: Branch Operations, Credit Appraisal, and Compliance Teams
**Supervisory Authority**: Reserve Bank of India & FIU-IND

---

## 1. Executive Mandate

All business units must implement strict Customer Due Diligence (CDD) and AML screening prior to account onboarding or executing any single remittance exceeding INR 50,000. Customer risk categorization must be updated dynamically based on real-time transaction activity.

## 2. Customer Risk Categorization

Every account is assigned a risk tier during onboarding:

| Risk Tier | Review Frequency | Typical Profile | Screening Controls |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Low)** | Every 10 years | Salaried individuals, government entities | Standard OVD verification |
| **Tier 2 (Medium)** | Every 8 years | Small businesses, retail traders | Field verification, GST validation |
| **Tier 3 (High)** | Every 2 years | PEPs, jewelers, bullion dealers, offshore trusts | Enhanced Due Diligence (EDD), DGM sign-off |

## 3. Politically Exposed Persons (PEP) Workflow

1. **Identification
```

### Sample 5: `sample_regulatory_circular.txt` (.txt) — Chunk 1 of 4

- **Chunk ID**: `sample_regulatory_circular_txt_tokenaware_001`
- **Source Identifier**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_regulatory_circular.txt`
- **Active Section**: `Preamble / Document Header`
- **Document Page**: `1`
- **Character Span**: `[0, 1293]` (1293 chars)
- **Token Count**: `300` tokens (Tokens [0:300])
- **Controlled Overlap**: `50` tokens
- **Provenance Trace Status**: `VERIFIED` (Similarity: 1.0)
- **Standard Citation**: `[Source: sample_regulatory_circular.txt, Chunk: 1/4, Page: 1]`

**Cleaned Chunk Text Content**:
```text
RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTMENT CENTRAL OFFICE, MUMBAI

Ref: RBI/2023-24/108 Circular No: DOR.AML.REC.66/14.01.001/2023-24 Date: January 04, 2024

To, All Scheduled Commercial Banks (excluding RRBs) All Small Finance Banks and Payment Banks All Non-Banking Financial Companies (NBFCs)

Subject: Master Direction - Know Your Customer (KYC) and Anti-Money Laundering (AML) Standards - Enhanced Due Diligence and Transaction Record Management

1. Preliminary and Statutory Authority
In exercise of the powers conferred by Section 35A of the Banking Regulation Act, 1949, read with Section 51A of the Unlawful Activities (Prevention) Act, 1967, and the Prevention of Money-Laundering (Maintenance of Records) Rules, 2005, the Reserve Bank of India hereby issues the updated Master Directions on Customer Due Diligence (CDD) and compliance obligations for regulated entities (REs).

2. Customer Due Diligence (CDD) Requirements
Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
(a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using
```

---

## 6. Reviewer Verification Checklist

- [x] **Task 1 — Full Pipeline**: Scanned full corpus directory tree, not just a single sample file.
- [x] **Task 2 — Ingestion Summary**: Summary report shows total sources, ingested documents, chunks created, and skipped files.
- [x] **Task 3 — Completeness Validation**: Mathematical check proves $\text{Discovered} (6) == \text{Ingested} (5) + \text{Skipped} (1)$ with zero silent drops.
- [x] **Task 4 — Sample Chunk Inspection**: Sample chunks across all formats verified for clean text, token bounds, section tags, and provenance.
- [x] **Task 5 — Committed Artifacts**: Pipeline code, validation logic, unit tests, and summary reports tracked and committed.
