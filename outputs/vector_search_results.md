# RegulSense: Vector Database Retrieval & Top-k Similarity Search Report

- **Sample User Query**: *"What are the customer due diligence and KYC verification requirements for onboarding new bank accounts?"*
- **Target Collection**: `regulsense_regulatory_chunks`
- **Embedding Model**: `all-minilm` (Dimension: `384` coordinates)
- **Distance Metric Space**: `Cosine Distance` ($HNSW:space = cosine$)
- **Query Vector Preview (First 8 Coordinates)**: `[-0.003556, -0.114709, 0.029186, -0.037025, -0.027214, 0.002123, 0.067351, -0.007881]`

---

## 1. Executive Summary & Retrieval Objectives (Tasks 1 - 3)

| Retrieval Objective | Execution Status | Audit Finding |
| :--- | :---: | :--- |
| **Query Embedding (Task 1)** | **COMPLETED** | Embedded via `all-minilm` with exact 384 dimensions matching indexed corpus |
| **Vector Store Search (Task 2)** | **COMPLETED** | Executed approximate nearest neighbor search directly against ChromaDB persistent collection |
| **Scores & Metadata Binding (Task 3)** | **COMPLETED** | Retained cosine similarity scores, raw distance, verbatim text, and provenance metadata |
| **Demonstrate Changing k (Task 4)** | **COMPLETED** | Contrasted 2 vs 5 nearest neighbors for grounding analysis |

---

## 2. Top-k Similarity Search Results (Task 2 & 3)

### Configuration: Top-2 Retrieved Chunks ($k=2$)

- **Total Chunks Retrieved**: `2`
- **Score Range**: `0.6501` to `0.6501` (Mean: `0.6501`)
- **Total Token Budget**: `600` tokens
- **Unique Source Documents**: `circular_dor_2024_108.txt, sample_regulatory_circular.txt`

| Rank | Similarity Score | Distance | Chunk ID | Source Document | Section Header | Page / Idx | Tokens |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :---: |
| **#1** | 🟢 **`0.6501`** | `0.3499` | `circular_dor_2024_108_txt_tokenaware_002` | `circular_dor_2024_108.txt` | 2. Customer Due Diligence (... | P.1 / #1 | 300 |
| **#2** | 🟢 **`0.6501`** | `0.3499` | `sample_regulatory_circular_txt_tokenaware_002` | `sample_regulatory_circular.txt` | 2. Customer Due Diligence (... | P.1 / #1 | 300 |

#### Verbatim Text Preview for $k=2$ Chunks

##### Chunk #1: `circular_dor_2024_108_txt_tokenaware_002` (Score: `0.6501`)
- **Source**: `circular_dor_2024_108.txt` | **Section**: `2. Customer Due Diligence (CDD) Requirements` | **Page**: `1`

> Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
> (a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using authorized OVDs (e.g., Passport, Permanent Account Number (PAN) Card, Voter ID, Aadhaar through secure offline or e-KYC channels).
> (b) Beneficial Ownership Identification: For corporate entities, trusts, and unincorporated associations, banks shall determine the natural person who ultimately owns or controls a customer, holding at least 10 percent of shares or voting rights.
> (c) Video-based Customer Identification Process (V-CIP): Where remote customer onboarding is conducted, live geo-tagging, facial match algorithms with a confidence score exceeding 95%, and liveliness detection must be strictly enforced.
> 
> 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs
> Accounts classified as high-risk, including Politically Exposed Persons (PEPs), non-resident customers, and trusts, warrant enhanced scrutiny:
> (a) Approval from Senior Management: Establishing relationships with PEPs, their family members, or close associates requires written approval from an officer not below the rank of Deputy General Manager.
> (b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
> (c) Heightened Transaction Monitoring: High

##### Chunk #2: `sample_regulatory_circular_txt_tokenaware_002` (Score: `0.6501`)
- **Source**: `sample_regulatory_circular.txt` | **Section**: `2. Customer Due Diligence (CDD) Requirements` | **Page**: `1`

> Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
> (a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using authorized OVDs (e.g., Passport, Permanent Account Number (PAN) Card, Voter ID, Aadhaar through secure offline or e-KYC channels).
> (b) Beneficial Ownership Identification: For corporate entities, trusts, and unincorporated associations, banks shall determine the natural person who ultimately owns or controls a customer, holding at least 10 percent of shares or voting rights.
> (c) Video-based Customer Identification Process (V-CIP): Where remote customer onboarding is conducted, live geo-tagging, facial match algorithms with a confidence score exceeding 95%, and liveliness detection must be strictly enforced.
> 
> 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs
> Accounts classified as high-risk, including Politically Exposed Persons (PEPs), non-resident customers, and trusts, warrant enhanced scrutiny:
> (a) Approval from Senior Management: Establishing relationships with PEPs, their family members, or close associates requires written approval from an officer not below the rank of Deputy General Manager.
> (b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
> (c) Heightened Transaction Monitoring: High

### Configuration: Top-5 Retrieved Chunks ($k=5$)

- **Total Chunks Retrieved**: `5`
- **Score Range**: `0.5660` to `0.6501` (Mean: `0.6032`)
- **Total Token Budget**: `1500` tokens
- **Unique Source Documents**: `circular_dor_2024_108.txt, sample_regulatory_circular.txt, guidelines_cdd_pml_rules.md`

| Rank | Similarity Score | Distance | Chunk ID | Source Document | Section Header | Page / Idx | Tokens |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :---: |
| **#1** | 🟢 **`0.6501`** | `0.3499` | `circular_dor_2024_108_txt_tokenaware_002` | `circular_dor_2024_108.txt` | 2. Customer Due Diligence (... | P.1 / #1 | 300 |
| **#2** | 🟢 **`0.6501`** | `0.3499` | `sample_regulatory_circular_txt_tokenaware_002` | `sample_regulatory_circular.txt` | 2. Customer Due Diligence (... | P.1 / #1 | 300 |
| **#3** | 🟢 **`0.5839`** | `0.4161` | `guidelines_cdd_pml_rules_md_tokenaware_001` | `guidelines_cdd_pml_rules.md` | Internal Compliance Guideli... | P.1 / #0 | 300 |
| **#4** | 🟢 **`0.5660`** | `0.4340` | `circular_dor_2024_108_txt_tokenaware_001` | `circular_dor_2024_108.txt` | Preamble / Document Header | P.1 / #0 | 300 |
| **#5** | 🟢 **`0.5660`** | `0.4340` | `sample_regulatory_circular_txt_tokenaware_001` | `sample_regulatory_circular.txt` | Preamble / Document Header | P.1 / #0 | 300 |

#### Verbatim Text Preview for $k=5$ Chunks

##### Chunk #1: `circular_dor_2024_108_txt_tokenaware_002` (Score: `0.6501`)
- **Source**: `circular_dor_2024_108.txt` | **Section**: `2. Customer Due Diligence (CDD) Requirements` | **Page**: `1`

> Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
> (a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using authorized OVDs (e.g., Passport, Permanent Account Number (PAN) Card, Voter ID, Aadhaar through secure offline or e-KYC channels).
> (b) Beneficial Ownership Identification: For corporate entities, trusts, and unincorporated associations, banks shall determine the natural person who ultimately owns or controls a customer, holding at least 10 percent of shares or voting rights.
> (c) Video-based Customer Identification Process (V-CIP): Where remote customer onboarding is conducted, live geo-tagging, facial match algorithms with a confidence score exceeding 95%, and liveliness detection must be strictly enforced.
> 
> 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs
> Accounts classified as high-risk, including Politically Exposed Persons (PEPs), non-resident customers, and trusts, warrant enhanced scrutiny:
> (a) Approval from Senior Management: Establishing relationships with PEPs, their family members, or close associates requires written approval from an officer not below the rank of Deputy General Manager.
> (b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
> (c) Heightened Transaction Monitoring: High

##### Chunk #2: `sample_regulatory_circular_txt_tokenaware_002` (Score: `0.6501`)
- **Source**: `sample_regulatory_circular.txt` | **Section**: `2. Customer Due Diligence (CDD) Requirements` | **Page**: `1`

> Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
> (a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using authorized OVDs (e.g., Passport, Permanent Account Number (PAN) Card, Voter ID, Aadhaar through secure offline or e-KYC channels).
> (b) Beneficial Ownership Identification: For corporate entities, trusts, and unincorporated associations, banks shall determine the natural person who ultimately owns or controls a customer, holding at least 10 percent of shares or voting rights.
> (c) Video-based Customer Identification Process (V-CIP): Where remote customer onboarding is conducted, live geo-tagging, facial match algorithms with a confidence score exceeding 95%, and liveliness detection must be strictly enforced.
> 
> 3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs
> Accounts classified as high-risk, including Politically Exposed Persons (PEPs), non-resident customers, and trusts, warrant enhanced scrutiny:
> (a) Approval from Senior Management: Establishing relationships with PEPs, their family members, or close associates requires written approval from an officer not below the rank of Deputy General Manager.
> (b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets.
> (c) Heightened Transaction Monitoring: High

##### Chunk #3: `guidelines_cdd_pml_rules_md_tokenaware_001` (Score: `0.5839`)
- **Source**: `guidelines_cdd_pml_rules.md` | **Section**: `Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules` | **Page**: `1`

> # Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules
> 
> **Document Identifier**: REG-COMP-2024-V2
> **Effective Date**: January 15, 2024
> **Applicability**: Branch Operations, Credit Appraisal, and Compliance Teams
> **Supervisory Authority**: Reserve Bank of India & FIU-IND
> 
> ---
> 
> ## 1. Executive Mandate
> 
> All business units must implement strict Customer Due Diligence (CDD) and AML screening prior to account onboarding or executing any single remittance exceeding INR 50,000. Customer risk categorization must be updated dynamically based on real-time transaction activity.
> 
> ## 2. Customer Risk Categorization
> 
> Every account is assigned a risk tier during onboarding:
> 
> | Risk Tier | Review Frequency | Typical Profile | Screening Controls |
> | :--- | :--- | :--- | :--- |
> | **Tier 1 (Low)** | Every 10 years | Salaried individuals, government entities | Standard OVD verification |
> | **Tier 2 (Medium)** | Every 8 years | Small businesses, retail traders | Field verification, GST validation |
> | **Tier 3 (High)** | Every 2 years | PEPs, jewelers, bullion dealers, offshore trusts | Enhanced Due Diligence (EDD), DGM sign-off |
> 
> ## 3. Politically Exposed Persons (PEP) Workflow
> 
> 1. **Identification

##### Chunk #4: `circular_dor_2024_108_txt_tokenaware_001` (Score: `0.5660`)
- **Source**: `circular_dor_2024_108.txt` | **Section**: `Preamble / Document Header` | **Page**: `1`

> RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTMENT CENTRAL OFFICE, MUMBAI
> 
> Ref: RBI/2023-24/108 Circular No: DOR.AML.REC.66/14.01.001/2023-24 Date: January 04, 2024
> 
> To, All Scheduled Commercial Banks (excluding RRBs) All Small Finance Banks and Payment Banks All Non-Banking Financial Companies (NBFCs)
> 
> Subject: Master Direction - Know Your Customer (KYC) and Anti-Money Laundering (AML) Standards - Enhanced Due Diligence and Transaction Record Management
> 
> 1. Preliminary and Statutory Authority
> In exercise of the powers conferred by Section 35A of the Banking Regulation Act, 1949, read with Section 51A of the Unlawful Activities (Prevention) Act, 1967, and the Prevention of Money-Laundering (Maintenance of Records) Rules, 2005, the Reserve Bank of India hereby issues the updated Master Directions on Customer Due Diligence (CDD) and compliance obligations for regulated entities (REs).
> 
> 2. Customer Due Diligence (CDD) Requirements
> Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
> (a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using

##### Chunk #5: `sample_regulatory_circular_txt_tokenaware_001` (Score: `0.5660`)
- **Source**: `sample_regulatory_circular.txt` | **Section**: `Preamble / Document Header` | **Page**: `1`

> RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTMENT CENTRAL OFFICE, MUMBAI
> 
> Ref: RBI/2023-24/108 Circular No: DOR.AML.REC.66/14.01.001/2023-24 Date: January 04, 2024
> 
> To, All Scheduled Commercial Banks (excluding RRBs) All Small Finance Banks and Payment Banks All Non-Banking Financial Companies (NBFCs)
> 
> Subject: Master Direction - Know Your Customer (KYC) and Anti-Money Laundering (AML) Standards - Enhanced Due Diligence and Transaction Record Management
> 
> 1. Preliminary and Statutory Authority
> In exercise of the powers conferred by Section 35A of the Banking Regulation Act, 1949, read with Section 51A of the Unlawful Activities (Prevention) Act, 1967, and the Prevention of Money-Laundering (Maintenance of Records) Rules, 2005, the Reserve Bank of India hereby issues the updated Master Directions on Customer Due Diligence (CDD) and compliance obligations for regulated entities (REs).
> 
> 2. Customer Due Diligence (CDD) Requirements
> Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction:
> (a) Verification of Officially Valid Documents (OVDs): Banks must verify the identity and permanent address of individual customers using

---

## 3. Comparative Analysis: Demonstrating Changing $k$ (Task 4)

Changing k from 2 to 5 expanded retrieved context from 2 chunks (600 tokens) to 5 chunks (1500 tokens). The top 2 results were strictly preserved. Expanding k introduced 3 additional chunks spanning 3 documents, with similarity score adjusting from 0.6501 down to 0.5660 (drop: 0.0840).

| Evaluation Dimension | Small $k$ ($k=2$) | Large $k$ ($k=5$) | Impact on Downstream Answer Generation |
| :--- | :---: | :---: | :--- |
| **Retrieved Chunks Count** | `2` | `5` | Wider selection provides richer context for complex multi-rule queries |
| **Cumulative Tokens** | `600` | `1500` | Token consumption increases proportionally with higher recall |
| **Minimum Similarity Score** | `0.6501` | `0.5660` | Score drops as rank increases, filtering lower-confidence content |
| **Preserved Common Chunks** | `2/2` (100%) | `2/5` | Top authoritative results remain stable at rank #1 and #2 |
| **Newly Introduced Chunks** | `0` | `3` | Added cross-document guidelines (e.g. PML Rules & Preambles) |

### Architectural Recommendations for RegulSense RAG

1. **Direct Statutory Lookups ($k=2-3$)**: For specific verification questions (e.g., 'What OVD documents are valid?'), $k=2$ maximizes precision and prevents context dilution.
2. **Cross-Regulation Synthesis ($k=5-8$)**: For multi-faceted compliance audits (e.g., conflicting circular resolution, penalty provisions), $k=5$ provides the necessary cross-statutory context.
3. **Score Thresholding**: A similarity cutoff of `score >= 0.50` effectively rejects tangential regulatory sections while retaining core statutory provisions.

---
*Report automatically generated by `src/retriever.py` for RegulSense Banking Compliance Assistant.*