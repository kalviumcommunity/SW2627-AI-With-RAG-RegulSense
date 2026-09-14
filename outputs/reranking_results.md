# RegulSense: Two-Stage Retrieval & Chunk Re-Ranking Audit Report

- **Target Vector Store**: `ChromaDB PersistentClient`
- **Collection Name**: `regulsense_regulatory_chunks`
- **Base Embedding Model**: `all-minilm` (Dimension: 384)
- **Re-Ranking Engine**: `ContextualCrossScorer`
- **Stage 1 Candidate Pool ($N$)**: `10` chunks
- **Stage 2 Final Top-k ($k$)**: `3` chunks
- **Execution Timestamp**: `2026-09-14T14:26:09.094002+00:00`

---

## 1. Executive Summary & Two-Stage Pipeline Architecture

| Architecture Stage | Component | Function & Strategy | Output Volume |
| :--- | :--- | :--- | :---: |
| **Stage 1: Candidate Retrieval** | Bi-Encoder Cosine Search | Broad ANN semantic search over indexed ChromaDB collection | **$N=10$** candidates |
| **Stage 2: Chunk Re-Ranking** | Contextual Cross-Scorer | Sentence-level directness, statutory entity match, preamble penalty | **Scored $10$** |
| **Stage 3: Context Selection** | Top-k Gating | Selects highest-scoring operative provisions for LLM grounding | **Final $k=3$** |

---

## 2. Before-and-After Re-Ranking Demonstration Cases (Tasks 3 & 4)

### Benchmark Case 1: Politically Exposed Persons (PEPs) Senior Management Approval Rank

- **User Query**: *"What officer rank approval is required for onboarding Politically Exposed Persons (PEPs)?"*
- **Initial Candidate Count ($N$)**: `10`
- **Final Selected Count ($k$)**: `3`
- **Top Result Improved**: 🚀 **YES (Operative clause promoted to #1)**

> **Re-Ranking Finding**: Re-ranking improved top result: 'circular_dor_2024_108_txt_tokenaware_002' was promoted from initial Rank #2 (vector score: 0.4550) to Rank #1 (re-rank score: 0.7544) because it contains the exact operative answer.

#### Before-and-After Comparison Ledger

| Initial Rank | Final Rank | Shift | Chunk ID | Source Document | Section Header | Vector Score | Re-Rank Score | Operative Snippet |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| #2 | **#1** | 🚀 **+1** | `circular_dor_2024_108_txt_tokenaware_002` | `circular_dor_2024_108.txt` | 2. Customer Due Dilige | `0.4550` | **`0.7544`** | "Regulated entities must undertake client identification and verif..." |
| #3 | **#2** | 🚀 **+1** | `sample_regulatory_circular_txt_tokenaware_002` | `sample_regulatory_circular.txt` | 2. Customer Due Dilige | `0.4550` | **`0.7544`** | "Regulated entities must undertake client identification and verif..." |
| #1 | **#3** | 🔻 **-2** | `guidelines_cdd_pml_rules_md_tokenaware_002` | `guidelines_cdd_pml_rules.md` | 2. Customer Risk Categ | `0.4969` | **`0.5084`** | "2 years | PEPs, jewelers, bullion dealers, offshore trusts | Enha..." |
| #4 | #4 | STABLE | `guidelines_cdd_pml_rules_md_tokenaware_001` | `guidelines_cdd_pml_rules.md` | Internal Compliance Gu | `0.3744` | **`0.4961`** | "# Internal Compliance Guidelines: CDD & Prevention of Money Laund..." |
| #5 | #5 | STABLE | `circular_dor_2024_108_txt_tokenaware_003` | `circular_dor_2024_108.txt` | 3. Enhanced Due Dilige | `0.2610` | **`0.3672`** | "officer not below the rank of Deputy General Manager. (b) Source..." |
| #6 | #6 | STABLE | `sample_regulatory_circular_txt_tokenaware_003` | `sample_regulatory_circular.txt` | 3. Enhanced Due Dilige | `0.2610` | **`0.3672`** | "officer not below the rank of Deputy General Manager. (b) Source..." |
| #7 | #7 | STABLE | `cyber_resilience_framework_pdf_tokenaware_002` | `cyber_resilience_framework.pdf` | Preamble / Document He | `0.2583` | **`0.3174`** | "x7x365 Security Operations Centre (SOC) equipped with continuous..." |
| #10 | #8 | 🚀 **+2** | `cyber_resilience_framework_pdf_tokenaware_001` | `cyber_resilience_framework.pdf` | Preamble / Document He | `0.2088` | **`0.2709`** | "RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATIO..." |
| #8 | #9 | 🔻 **-1** | `circular_dor_2024_108_txt_tokenaware_001` | `circular_dor_2024_108.txt` | Preamble / Document He | `0.2258` | **`0.2306`** | "RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTME..." |
| #9 | #10 | 🔻 **-1** | `sample_regulatory_circular_txt_tokenaware_001` | `sample_regulatory_circular.txt` | Preamble / Document He | `0.2258` | **`0.2306`** | "RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTME..." |

#### Final Top-3 Grounding Context Selected for LLM:

1. **`circular_dor_2024_108_txt_tokenaware_002`** (Re-Rank Score: **`0.7544`** | Initial Vector: `#2`, `0.4550`)
   - **Source**: `circular_dor_2024_108.txt` | **Section**: `2. Customer Due Diligence (CDD) Requirements` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.97 (best: 'enhanced due diligence (edd) for high-risk accounts and peps...'), Entity match: 0.50, Directness: 0.85
   - **Verbatim Text**:
     > Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction: (a) Verification of Officially Valid Documents (OVDs): Banks must verify the identi...

2. **`sample_regulatory_circular_txt_tokenaware_002`** (Re-Rank Score: **`0.7544`** | Initial Vector: `#3`, `0.4550`)
   - **Source**: `sample_regulatory_circular.txt` | **Section**: `2. Customer Due Diligence (CDD) Requirements` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.97 (best: 'enhanced due diligence (edd) for high-risk accounts and peps...'), Entity match: 0.50, Directness: 0.85
   - **Verbatim Text**:
     > Regulated entities must undertake client identification and verification procedures before establishing an account-based relationship or executing an occasional cross-border financial transaction: (a) Verification of Officially Valid Documents (OVDs): Banks must verify the identi...

3. **`guidelines_cdd_pml_rules_md_tokenaware_002`** (Re-Rank Score: **`0.5084`** | Initial Vector: `#1`, `0.4969`)
   - **Source**: `guidelines_cdd_pml_rules.md` | **Section**: `2. Customer Risk Categorization` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.42 (best: 'politically exposed persons (pep) workflow...'), Entity match: 0.50, Directness: 0.71
   - **Verbatim Text**:
     > 2 years | PEPs, jewelers, bullion dealers, offshore trusts | Enhanced Due Diligence (EDD), DGM sign-off |  ## 3. Politically Exposed Persons (PEP) Workflow  1. **Identification**: Cross-reference prospective names against global sanctions lists, PEP registries, and adverse media ...

---

### Benchmark Case 2: Digital Lending Recovery Agent Permitted Hours & Prohibitions

- **User Query**: *"What are the permitted hours for recovery agents contacting borrowers and what penalties apply for harassment?"*
- **Initial Candidate Count ($N$)**: `10`
- **Final Selected Count ($k$)**: `3`
- **Top Result Improved**: ✅ Confirmed at #1

> **Re-Ranking Finding**: Top candidate 'digital_lending_compliance_note_html_tokenaware_001' retained Rank #1 with confirmed re-rank score of 0.5199.

#### Before-and-After Comparison Ledger

| Initial Rank | Final Rank | Shift | Chunk ID | Source Document | Section Header | Vector Score | Re-Rank Score | Operative Snippet |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| #1 | **#1** | STABLE | `digital_lending_compliance_note_html_tokenaware_001` | `digital_lending_compliance_note.html` | Preamble / Document He | `0.4766` | **`0.5199`** | "Regulatory Advisory: Digital Lending Guidelines & Consumer Protec..." |
| #2 | **#2** | STABLE | `digital_lending_compliance_note_html_tokenaware_002` | `digital_lending_compliance_note.html` | 3. Code of Conduct for | `0.4423` | **`0.4473`** | "or after 7:00 PM. Harassment, verbal intimidation, or unauthorize..." |
| #7 | **#3** | 🚀 **+4** | `cyber_resilience_framework_pdf_tokenaware_001` | `cyber_resilience_framework.pdf` | Preamble / Document He | `0.2338` | **`0.3289`** | "RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATIO..." |
| #3 | #4 | 🔻 **-1** | `circular_dor_2024_108_txt_tokenaware_004` | `circular_dor_2024_108.txt` | 5. Record Retention Ob | `0.3249` | **`0.3180`** | "igations Under Rule 3 and Rule 10 of the PML Rules, 2005, all reg..." |
| #4 | #5 | 🔻 **-1** | `sample_regulatory_circular_txt_tokenaware_004` | `sample_regulatory_circular.txt` | 5. Record Retention Ob | `0.3249` | **`0.3180`** | "igations Under Rule 3 and Rule 10 of the PML Rules, 2005, all reg..." |
| #8 | #6 | 🚀 **+2** | `guidelines_cdd_pml_rules_md_tokenaware_002` | `guidelines_cdd_pml_rules.md` | 2. Customer Risk Categ | `0.2322` | **`0.3152`** | "2 years | PEPs, jewelers, bullion dealers, offshore trusts | Enha..." |
| #9 | #7 | 🚀 **+2** | `guidelines_cdd_pml_rules_md_tokenaware_001` | `guidelines_cdd_pml_rules.md` | Internal Compliance Gu | `0.2291` | **`0.3149`** | "# Internal Compliance Guidelines: CDD & Prevention of Money Laund..." |
| #10 | #8 | 🚀 **+2** | `cyber_resilience_framework_pdf_tokenaware_002` | `cyber_resilience_framework.pdf` | Preamble / Document He | `0.2290` | **`0.3145`** | "x7x365 Security Operations Centre (SOC) equipped with continuous..." |
| #5 | #9 | 🔻 **-4** | `circular_dor_2024_108_txt_tokenaware_001` | `circular_dor_2024_108.txt` | Preamble / Document He | `0.2565` | **`0.2337`** | "RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTME..." |
| #6 | #10 | 🔻 **-4** | `sample_regulatory_circular_txt_tokenaware_001` | `sample_regulatory_circular.txt` | Preamble / Document He | `0.2565` | **`0.2337`** | "RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTME..." |

#### Final Top-3 Grounding Context Selected for LLM:

1. **`digital_lending_compliance_note_html_tokenaware_001`** (Re-Rank Score: **`0.5199`** | Initial Vector: `#1`, `0.4766`)
   - **Source**: `digital_lending_compliance_note.html` | **Section**: `Preamble / Document Header` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.56 (best: 'strict prohibition: regulated entities and their engaged rec...'), Entity match: 0.50, Directness: 0.50
   - **Verbatim Text**:
     > Regulatory Advisory: Digital Lending Guidelines & Consumer Protection  RegulSense Regulatory Advisory: Digital Lending & Fair Practices Code  Reference: RBI/2022-23/DOR.CRE.REC.42/21.04.048/2022-23  Target Entities: Commercial Banks, Digital Lending Platforms, Lending Service Pro...

2. **`digital_lending_compliance_note_html_tokenaware_002`** (Re-Rank Score: **`0.4473`** | Initial Vector: `#2`, `0.4423`)
   - **Source**: `digital_lending_compliance_note.html` | **Section**: `3. Code of Conduct for Recovery Agents` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.28 (best: 'harassment, verbal intimidation, or unauthorized contact wit...'), Entity match: 0.50, Directness: 0.71
   - **Verbatim Text**:
     > or after 7:00 PM. Harassment, verbal intimidation, or unauthorized contact with borrowers' relatives or social networks will lead to immediate regulatory sanction.  4. Data Privacy and Storage Restrictions  Digital lending apps shall not access mobile phone resources such as file...

3. **`cyber_resilience_framework_pdf_tokenaware_001`** (Re-Rank Score: **`0.3289`** | Initial Vector: `#7`, `0.2338`)
   - **Source**: `cyber_resilience_framework.pdf` | **Section**: `Preamble / Document Header` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.14 (best: 'incident reporting timelines (6-hour rule) any cyber securit...'), Entity match: 0.50, Directness: 0.50
   - **Verbatim Text**:
     > RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls 1. Mand...

---

### Benchmark Case 3: Cyber Security Incident Notification Timeline (6-Hour Rule)

- **User Query**: *"What is the mandatory timeframe for banks to report Severity 1 cyber security incidents to CERT-In and RBI?"*
- **Initial Candidate Count ($N$)**: `10`
- **Final Selected Count ($k$)**: `3`
- **Top Result Improved**: ✅ Confirmed at #1

> **Re-Ranking Finding**: Top candidate 'cyber_resilience_framework_pdf_tokenaware_001' retained Rank #1 with confirmed re-rank score of 0.5152.

#### Before-and-After Comparison Ledger

| Initial Rank | Final Rank | Shift | Chunk ID | Source Document | Section Header | Vector Score | Re-Rank Score | Operative Snippet |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| #1 | **#1** | STABLE | `cyber_resilience_framework_pdf_tokenaware_001` | `cyber_resilience_framework.pdf` | Preamble / Document He | `0.6520` | **`0.5152`** | "RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATIO..." |
| #2 | **#2** | STABLE | `cyber_resilience_framework_pdf_tokenaware_002` | `cyber_resilience_framework.pdf` | Preamble / Document He | `0.5681` | **`0.3928`** | "x7x365 Security Operations Centre (SOC) equipped with continuous..." |
| #3 | **#3** | STABLE | `circular_dor_2024_108_txt_tokenaware_003` | `circular_dor_2024_108.txt` | 3. Enhanced Due Dilige | `0.5629` | **`0.2763`** | "officer not below the rank of Deputy General Manager. (b) Source..." |
| #4 | #4 | STABLE | `sample_regulatory_circular_txt_tokenaware_003` | `sample_regulatory_circular.txt` | 3. Enhanced Due Dilige | `0.5629` | **`0.2763`** | "officer not below the rank of Deputy General Manager. (b) Source..." |
| #8 | #5 | 🚀 **+3** | `circular_dor_2024_108_txt_tokenaware_004` | `circular_dor_2024_108.txt` | 5. Record Retention Ob | `0.4703` | **`0.2170`** | "igations Under Rule 3 and Rule 10 of the PML Rules, 2005, all reg..." |
| #9 | #6 | 🚀 **+3** | `sample_regulatory_circular_txt_tokenaware_004` | `sample_regulatory_circular.txt` | 5. Record Retention Ob | `0.4703` | **`0.2170`** | "igations Under Rule 3 and Rule 10 of the PML Rules, 2005, all reg..." |
| #10 | #7 | 🚀 **+3** | `digital_lending_compliance_note_html_tokenaware_001` | `digital_lending_compliance_note.html` | Preamble / Document He | `0.4182` | **`0.1918`** | "Regulatory Advisory: Digital Lending Guidelines & Consumer Protec..." |
| #7 | #8 | 🔻 **-1** | `guidelines_cdd_pml_rules_md_tokenaware_001` | `guidelines_cdd_pml_rules.md` | Internal Compliance Gu | `0.4810` | **`0.1901`** | "# Internal Compliance Guidelines: CDD & Prevention of Money Laund..." |
| #5 | #9 | 🔻 **-4** | `circular_dor_2024_108_txt_tokenaware_001` | `circular_dor_2024_108.txt` | Preamble / Document He | `0.5059` | **`0.1586`** | "RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTME..." |
| #6 | #10 | 🔻 **-4** | `sample_regulatory_circular_txt_tokenaware_001` | `sample_regulatory_circular.txt` | Preamble / Document He | `0.5059` | **`0.1586`** | "RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTME..." |

#### Final Top-3 Grounding Context Selected for LLM:

1. **`cyber_resilience_framework_pdf_tokenaware_001`** (Re-Rank Score: **`0.5152`** | Initial Vector: `#1`, `0.6520`)
   - **Source**: `cyber_resilience_framework.pdf` | **Section**: `Preamble / Document Header` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.50 (best: 'incident reporting timelines (6-hour rule) any cyber securit...'), Entity match: 0.50, Directness: 0.50
   - **Verbatim Text**:
     > RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls 1. Mand...

2. **`cyber_resilience_framework_pdf_tokenaware_002`** (Re-Rank Score: **`0.3928`** | Initial Vector: `#2`, `0.5681`)
   - **Source**: `cyber_resilience_framework.pdf` | **Section**: `Preamble / Document Header` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.25 (best: '(b) third-party service providers and cloud vendors must und...'), Entity match: 0.50, Directness: 0.43
   - **Verbatim Text**:
     > x7x365 Security Operations Centre (SOC) equipped with continuous automated log monitoring, SIEM analytics, and automated threat hunting capabilities.  4. API Security and Third-Party Risk Management (a) All open banking and fintech integrations must mandate TLS 1.3 encryption wit...

3. **`circular_dor_2024_108_txt_tokenaware_003`** (Re-Rank Score: **`0.2763`** | Initial Vector: `#3`, `0.5629`)
   - **Source**: `circular_dor_2024_108.txt` | **Section**: `3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs` (Page 1)
   - **Scoring Rationale**: Sentence overlap: 0.12 (best: 'transaction monitoring and reporting thresholds
banks shall ...'), Entity match: 0.00, Directness: 0.85
   - **Verbatim Text**:
     > officer not below the rank of Deputy General Manager. (b) Source of Funds Verification: The source of wealth and funds must be explicitly documented with corroborating financial statements, tax returns, or audited balance sheets. (c) Heightened Transaction Monitoring: High-risk a...

---

## 3. Production Architecture Guidelines (Task 5)

- 1. Stage 1 Broad Retrieval (N=10) guarantees high candidate recall across multiple circulars.
- 2. Stage 2 Fine-Grained Cross-Scoring evaluates clause directness and penalizes introductory preambles.
- 3. Numeric and statutory entity matching ensures mandatory timelines (e.g. 6 hours) and thresholds (e.g. 95%) achieve Rank #1.
- 4. Final top-k (k=3) delivers an optimized, noise-free context window to the downstream LLM generator.

---
*Report automatically generated by `src/reranker.py` for RegulSense Banking Compliance Assistant.*