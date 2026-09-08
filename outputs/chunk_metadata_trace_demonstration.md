# RegulSense: Chunk Metadata Architecture & Source Tracing Demonstration

- **Execution Timestamp**: 2026-09-08 09:13:27
- **Sample Corpus Scope**: `data\sample_corpus` (4 documents processed)
- **Total Chunks Generated**: 9 chunks with consistent metadata
- **Sample Chunks Export Path**: `outputs\sample_chunks_with_metadata.json`

---

## 1. Metadata Schema Architecture (Task 1, 2, 3)

In compliance RAG systems, untagged chunks make responses untraceable, non-verifiable, and legally unenforceable.
RegulSense guarantees a **strictly uniform 13-field metadata schema** attached to every single chunk across all document formats (`.txt`, `.pdf`, `.html`, `.md`):

| Field Name | Type | Description | Corpus Uniformity Guarantee |
| :--- | :---: | :--- | :---: |
| `source` | `str` | Fully-qualified path or URI to the source document | Guaranteed on 100% of chunks |
| `filename` | `str` | Base filename of origin file | Guaranteed on 100% of chunks |
| `document_id` | `str` | Normalized identifier for filtering and partitioning | Guaranteed on 100% of chunks |
| `file_type` | `str` | Extension format (`.txt`, `.pdf`, `.html`, `.md`) | Guaranteed on 100% of chunks |
| `section` | `str` | Specific heading, clause, or circular section | Guaranteed on 100% of chunks |
| `page_number` | `int` | 1-based page location in source file | Guaranteed on 100% of chunks |
| `chunk_index` | `int` | 0-indexed position within the document | Guaranteed on 100% of chunks |
| `total_chunks`| `int` | Total count of chunks derived from this document | Guaranteed on 100% of chunks |
| `char_start`  | `int` | Exact starting character offset in origin text | Guaranteed on 100% of chunks |
| `char_end`    | `int` | Exact ending character offset in origin text | Guaranteed on 100% of chunks |
| `char_count`  | `int` | Length of chunk content in characters | Guaranteed on 100% of chunks |
| `token_count` | `int` | Token count evaluated by `cl100k_base` tokenizer | Guaranteed on 100% of chunks |
| `strategy`    | `str` | Strategy name (`RecursiveStructuralChunker`, etc.) | Guaranteed on 100% of chunks |

---

## 2. End-to-End Source Tracing Demonstrations (Task 4)

The following simulations demonstrate how retrieved chunks are deterministically traced back to their exact character offsets, sections, pages, and surrounding text in the original document.

### Demonstration Case 1: Anti-Money Laundering (AML) / Customer Due Diligence
- **User Compliance Query**: *"Who must approve business relationships with Politically Exposed Persons (PEPs)?"*
- **Retrieved Chunk ID**: `circular_dor_2024_108_txt_recursive_002`
- **Citation**: `[Source: circular_dor_2024_108.txt, Chunk: 2/4, Section: '2. Customer Due Diligence (CDD) Requirements', Page: 1]`
- **Trace Verification Status**: **VERIFIED (100% Character Match)**
- **Source File**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\circular_dor_2024_108.txt`
- **Origin Section**: `2. Customer Due Diligence (CDD) Requirements`
- **Page Number**: `Page 1`
- **Exact Character Span**: `[914 : 2666]` (1752 chars, 334 tokens)

#### Retrieved Chunk Content:
```text
2. Customer Due Diligence (CDD) Requirements
Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
(a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using authorized OVDs (e.g., Passport, Permanent Account Number (PAN) Card, Voter ID, Aadhaar through secure offline or e-KYC channels).
(b) Beneficial Ownership Identification: For corporate entities, trusts, and unincorporated associations, banks shall determine the natural person who ultimately owns or controls a customer, holding at least 10 percent of shares or voting rights.
(c) Video-based Customer Identification Process (V-CIP): Where remote customer onboarding is conducted, live geo-tagging, facial match algorithms with a confidence score exceeding 95%, and liveliness detection must be strictly enforced.

3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs
Accounts classified as high-risk, including Politically Exposed Persons (PEPs), non-resident customers, and trusts, warrant enhanced scrutiny:
(a) Approval from Senior Management: Establishing relationships with PEPs, their family members, or close associates requires written approval from an officer not below the rank of Deputy General Manager.
(b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
(c) Heightened Transaction Monitoring: High-risk accounts shall be subjected to quarterly reviews, compared against the standard biennial review for low-risk customers.
```

#### Verified Source Context Trace (Showing Ground Truth Neighborhood):
```text
...the updated Master Directions on Customer Due Diligence (CDD) and compliance obligations for regulated entities (REs).

 >>> [CHUNK CONTENT] <<< 

4. Transaction Monitoring and Reporting Thresholds
Banks shall deploy rule-based and behavioral automated transaction ...
```

---

### Demonstration Case 2: Cyber Resilience & Payment Security
- **User Compliance Query**: *"What is the mandatory regulatory timeline for reporting cyber security incidents?"*
- **Retrieved Chunk ID**: `cyber_resilience_framework_pdf_recursive_001`
- **Citation**: `[Source: cyber_resilience_framework.pdf, Chunk: 1/2, Page: 1]`
- **Trace Verification Status**: **VERIFIED (100% Character Match)**
- **Source File**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\cyber_resilience_framework.pdf`
- **Origin Section**: `Preamble / Document Header`
- **Page Number**: `Page 1`
- **Exact Character Span**: `[0 : 1237]` (1237 chars, 282 tokens)

#### Retrieved Chunk Content:
```text
RESERVE BANK OF INDIA
DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY
CENTRAL OFFICE, MUMBAI
Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25
Date: March 12, 2024
Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls
1. Mandatory Two-Factor Authentication (2FA)
All regulated payment system operators and scheduled commercial banks must enforce dynamic two-factor authentication (2FA) for all domestic electronic fund transfers (NEFT, RTGS, IMPS, and UPI). At least one factor must be dynamic, such as a time-based one-time password (TOTP) or biometric verification.
2. Incident Reporting Timelines (6-Hour Rule)
Any cyber security incident, ransomware compromise, unauthorized system intrusion, or major denial of service (DoS) affecting customer-facing channels must be reported to the RBI Cyber Security Cell (CSITE) and CERT-In within 6 hours of detection. Initial reports must be followed by a comprehensive forensic analysis report within 7 business days.
3. Security Operations Centre (SOC) Operations
Banks must operate a 24x7x365 Security Operations Centre (SOC) equipped with continuous automated log monitoring, SIEM analytics, and automated threat hunting capabilities.
```

#### Verified Source Context Trace (Showing Ground Truth Neighborhood):
```text
... >>> [CHUNK CONTENT] <<< 

4. API Security and Third-Party Risk Management
(a) All open banking and fintech integrations must mandate TLS 1.3 enc...
```

---

### Demonstration Case 3: Digital Lending Fair Practices
- **User Compliance Query**: *"What are the permissible hours for digital lending recovery agents to contact borrowers?"*
- **Retrieved Chunk ID**: `digital_lending_compliance_note_html_recursive_001`
- **Citation**: `[Source: digital_lending_compliance_note.html, Chunk: 1/1, Page: 1]`
- **Trace Verification Status**: **VERIFIED (100% Character Match)**
- **Source File**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\digital_lending_compliance_note.html`
- **Origin Section**: `Preamble / Document Header`
- **Page Number**: `Page 1`
- **Exact Character Span**: `[0 : 1708]` (1707 chars, 350 tokens)

#### Retrieved Chunk Content:
```text
Regulatory Advisory: Digital Lending Guidelines & Consumer Protection

RegulSense Regulatory Advisory: Digital Lending & Fair Practices Code

Reference:
 RBI/2022-23/DOR.CRE.REC.42/21.04.048/2022-23

Target Entities:
 Commercial Banks, Digital Lending Platforms, Lending Service Providers (LSPs)

1. Direct Fund Disbursal and Repayment

All loan disbursals and repayments must be executed solely between the bank account of the borrower and the regulated entity (RE) without any pass-through or pool account of any third-party Lending Service Provider (LSP).

2. Key Fact Statement (KFS) Mandate

A standardized Key Fact Statement (KFS) must be provided to the prospective borrower before execution of any loan agreement. The KFS must explicitly disclose:

Annual Percentage Rate (APR) including all fees, processing charges, and penal interests.

Cooling-off / look-up period during which borrowers can exit digital loans without penalty.

Detailed schedule of installments and default handling terms.

3. Code of Conduct for Recovery Agents

Strict Prohibition:
 Regulated entities and their engaged recovery agents are strictly prohibited from contacting borrowers before 8:00 AM or after 7:00 PM. Harassment, verbal intimidation, or unauthorized contact with borrowers' relatives or social networks will lead to immediate regulatory sanction.

4. Data Privacy and Storage Restrictions

Digital lending apps shall not access mobile phone resources such as file storage, contact lists, call logs, or camera (except for one-time KYC onboarding with express consent). Biometric data must never be stored on third-party servers.

Department of Supervision • Reserve Bank of India • Confidential Internal Copy
```

#### Verified Source Context Trace (Showing Ground Truth Neighborhood):
```text
... >>> [CHUNK CONTENT] <<< ...
```

---

## 3. Sample Chunks with Consistent Metadata (Task 5)

The table below shows representative chunks across multiple file types demonstrating consistent metadata schema compliance:

| Chunk ID | Document | Format | Section | Page | Pos | Tokens | Sample Snippet |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `circular_dor_2024_108_txt_recursive_001` | `circular_dor_2024_108.txt` | `.txt` | Preamble / Document Header | 1 | 1/4 | 242 | RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTMENT CENTRAL OFFI... |
| `circular_dor_2024_108_txt_recursive_002` | `circular_dor_2024_108.txt` | `.txt` | 2. Customer Due Diligence (CDD) Requirements | 1 | 2/4 | 334 | 2. Customer Due Diligence (CDD) Requirements Regulated entities must undertake c... |
| `circular_dor_2024_108_txt_recursive_003` | `circular_dor_2024_108.txt` | `.txt` | 4. Transaction Monitoring and Reporting Thresholds | 1 | 3/4 | 341 | 4. Transaction Monitoring and Reporting Thresholds Banks shall deploy rule-based... |
| `circular_dor_2024_108_txt_recursive_004` | `circular_dor_2024_108.txt` | `.txt` | 6. Non-Compliance Sanctions | 1 | 4/4 | 69 | 6. Non-Compliance Sanctions Failure to adhere to these directions shall attract ... |
| `cyber_resilience_framework_pdf_recursive_001` | `cyber_resilience_framework.pdf` | `.pdf` | Preamble / Document Header | 1 | 1/2 | 282 | RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CE... |
| `cyber_resilience_framework_pdf_recursive_002` | `cyber_resilience_framework.pdf` | `.pdf` | 4. API Security and Third-Party Risk Management | 2 | 2/2 | 179 | 4. API Security and Third-Party Risk Management (a) All open banking and fintech... |
| `digital_lending_compliance_note_html_recursive_001` | `digital_lending_compliance_note.html` | `.html` | Preamble / Document Header | 1 | 1/1 | 350 | Regulatory Advisory: Digital Lending Guidelines & Consumer Protection RegulSense... |
| `guidelines_cdd_pml_rules_md_recursive_001` | `guidelines_cdd_pml_rules.md` | `.md` | Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules | 1 | 1/2 | 295 | # Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rul... |

---

## 4. Architectural Summary: Why Metadata Tagging is Mandatory for RegulSense

1. **Statutory Citation Requirements**: Banking regulations mandate that compliance answers cite official Circular numbers, Sections, and Clauses. Tagged metadata ensures citations are generated programmatically without hallucination.
2. **Granular Metadata Filtering**: During vector retrieval, queries can be filtered by `file_type: .pdf`, `section: '4. Transaction Monitoring'`, or `document_id` before computing vector similarities.
3. **Deterministic Auditability**: Regulators can trace any advice generated by RegulSense back to the exact byte and character position of the underlying Reserve Bank of India circular.