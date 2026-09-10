# RegulSense: Embedding Quality & Retrieval Sanity Report

- **Target Embedding Model**: `all-minilm`
- **Corpus Size**: `15 prepared regulatory chunks`
- **Test Suite Size**: `6 known relevance test cases`
- **Overall Pass Rate**: `5/6 passed (83.3%)`
- **Average Separation Margin**: `+0.2339` (Cosine Similarity delta)
- **Execution Latency**: `2.889 seconds`

---

## 1. Executive Sanity Verification Matrix (Tasks 1, 2 & 4)

The table below evaluates whether known relevant regulatory chunks rank above unrelated chunks, reporting top retrieved sources, scores, and margins:

| Test ID | Domain Category | Search Query | Best Rel Rank | Score | Neg Ctrl Score | Margin | Top-1 Source Document | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `SANITY-01` | Customer Due Diligence / KYC | "What officially valid documents (OVDs) are..." | `#1` | `0.6657` | `0.2903` | `+0.3754` | `circular_dor_2024_108.txt` | **PASSED** |
| `SANITY-02` | Cyber Resilience & Incident Response | "What are the incident reporting requiremen..." | `#1` | `0.5845` | `0.3217` | `+0.2628` | `cyber_resilience_framework.pdf` | **PASSED** |
| `SANITY-03` | Statutory Record Retention | "How many years must banks retain customer ..." | `#1` | `0.6386` | `0.3400` | `+0.2986` | `circular_dor_2024_108.txt` | **PASSED** |
| `SANITY-04` | Digital Lending Disclosures | "What disclosures must digital lending plat..." | `#1` | `0.5272` | `0.3155` | `+0.2117` | `digital_lending_compliance_note.html` | **PASSED** |
| `SANITY-05` | Governance & PEP Approval Authority | "Who must approve establishing a banking re..." | `#6` | `0.4071` | `0.3381` | `+0.0690` | `circular_dor_2024_108.txt` | **FAILED (EXPECTED_SURPRISE)** |
| `SANITY-06` | Debt Recovery Conduct Hours | "What hours are recovery agents permitted t..." | `#2` | `0.5080` | `0.3220` | `+0.1860` | `digital_lending_compliance_note.html` | **PASSED** |

---

## 2. Test-by-Test Retrieval Ledger (Task 2 & 4)

### SANITY-01: Customer Due Diligence / KYC

- **Query**: *"What officially valid documents (OVDs) are required for customer due diligence and identity verification?"*
- **Evaluation Status**: **PASSED** (Expected: `PASS`)
- **Separation Margin**: `+0.3754` (Target score `0.6657` vs Unrelated `0.2903`)
- **Diagnostic Verdict**: PASSED: Target chunk 'circular_dor_2024_108_txt_tokenaware_002' achieved Rank 1 (Score: 0.6657), outranking negative control by margin of +0.3754.

**Top-3 Retrieved Chunks**:

| Rank | Chunk ID | Cosine Score | Governing Section | Document Source |
| :---: | :--- | :---: | :--- | :--- |
| `#1` | `circular_dor_2024_108_txt_tokenaware_002` | `0.6657` | 2. Customer Due Diligence (CD... | `circular_dor_2024_108.txt` |
| `#2` | `sample_regulatory_circular_txt_tokenaware_002` | `0.6657` | 2. Customer Due Diligence (CD... | `sample_regulatory_circular.txt` |
| `#3` | `guidelines_cdd_pml_rules_md_tokenaware_001` | `0.4629` | Internal Compliance Guideline... | `guidelines_cdd_pml_rules.md` |

---

### SANITY-02: Cyber Resilience & Incident Response

- **Query**: *"What are the incident reporting requirements and mandatory notification timelines for bank cyber attacks?"*
- **Evaluation Status**: **PASSED** (Expected: `PASS`)
- **Separation Margin**: `+0.2628` (Target score `0.5845` vs Unrelated `0.3217`)
- **Diagnostic Verdict**: PASSED: Target chunk 'cyber_resilience_framework_pdf_tokenaware_001' achieved Rank 1 (Score: 0.5845), outranking negative control by margin of +0.2628.

**Top-3 Retrieved Chunks**:

| Rank | Chunk ID | Cosine Score | Governing Section | Document Source |
| :---: | :--- | :---: | :--- | :--- |
| `#1` | `cyber_resilience_framework_pdf_tokenaware_001` | `0.5845` | Preamble / Document Header | `cyber_resilience_framework.pdf` |
| `#2` | `cyber_resilience_framework_pdf_tokenaware_002` | `0.5523` | Preamble / Document Header | `cyber_resilience_framework.pdf` |
| `#3` | `circular_dor_2024_108_txt_tokenaware_003` | `0.5319` | 3. Enhanced Due Diligence (ED... | `circular_dor_2024_108.txt` |

---

### SANITY-03: Statutory Record Retention

- **Query**: *"How many years must banks retain customer transaction records and account opening files?"*
- **Evaluation Status**: **PASSED** (Expected: `PASS`)
- **Separation Margin**: `+0.2986` (Target score `0.6386` vs Unrelated `0.3400`)
- **Diagnostic Verdict**: PASSED: Target chunk 'circular_dor_2024_108_txt_tokenaware_004' achieved Rank 1 (Score: 0.6386), outranking negative control by margin of +0.2986.

**Top-3 Retrieved Chunks**:

| Rank | Chunk ID | Cosine Score | Governing Section | Document Source |
| :---: | :--- | :---: | :--- | :--- |
| `#1` | `circular_dor_2024_108_txt_tokenaware_004` | `0.6386` | 5. Record Retention Obligations | `circular_dor_2024_108.txt` |
| `#2` | `sample_regulatory_circular_txt_tokenaware_004` | `0.6386` | 5. Record Retention Obligations | `sample_regulatory_circular.txt` |
| `#3` | `guidelines_cdd_pml_rules_md_tokenaware_003` | `0.5560` | 4. Beneficial Ownership Thres... | `guidelines_cdd_pml_rules.md` |

---

### SANITY-04: Digital Lending Disclosures

- **Query**: *"What disclosures must digital lending platforms make regarding all-inclusive interest rates and annual percentage rates?"*
- **Evaluation Status**: **PASSED** (Expected: `PASS`)
- **Separation Margin**: `+0.2117` (Target score `0.5272` vs Unrelated `0.3155`)
- **Diagnostic Verdict**: PASSED: Target chunk 'digital_lending_compliance_note_html_tokenaware_001' achieved Rank 1 (Score: 0.5272), outranking negative control by margin of +0.2117.

**Top-3 Retrieved Chunks**:

| Rank | Chunk ID | Cosine Score | Governing Section | Document Source |
| :---: | :--- | :---: | :--- | :--- |
| `#1` | `digital_lending_compliance_note_html_tokenaware_001` | `0.5272` | Preamble / Document Header | `digital_lending_compliance_note.html` |
| `#2` | `digital_lending_compliance_note_html_tokenaware_002` | `0.4697` | 3. Code of Conduct for Recove... | `digital_lending_compliance_note.html` |
| `#3` | `cyber_resilience_framework_pdf_tokenaware_002` | `0.3605` | Preamble / Document Header | `cyber_resilience_framework.pdf` |

---

### SANITY-05: Governance & PEP Approval Authority

- **Query**: *"Who must approve establishing a banking relationship with a Politically Exposed Person?"*
- **Evaluation Status**: **FAILED (EXPECTED_SURPRISE)** (Expected: `SURPRISING_FAIL`)
- **Separation Margin**: `+0.0690` (Target score `0.4071` vs Unrelated `0.3381`)
- **Diagnostic Verdict**: CONFIRMED SURPRISING CASE: Related chunk outranked negative control (+0.0690), but failed top-3 threshold (Rank 6, Score 0.4071) due to bi-encoder semantic dilution.

**Top-3 Retrieved Chunks**:

| Rank | Chunk ID | Cosine Score | Governing Section | Document Source |
| :---: | :--- | :---: | :--- | :--- |
| `#1` | `circular_dor_2024_108_txt_tokenaware_002` | `0.4681` | 2. Customer Due Diligence (CD... | `circular_dor_2024_108.txt` |
| `#2` | `sample_regulatory_circular_txt_tokenaware_002` | `0.4681` | 2. Customer Due Diligence (CD... | `sample_regulatory_circular.txt` |
| `#3` | `guidelines_cdd_pml_rules_md_tokenaware_002` | `0.4390` | 2. Customer Risk Categorization | `guidelines_cdd_pml_rules.md` |

---

### SANITY-06: Debt Recovery Conduct Hours

- **Query**: *"What hours are recovery agents permitted to call borrowers and what conduct is prohibited?"*
- **Evaluation Status**: **PASSED** (Expected: `BORDERLINE`)
- **Separation Margin**: `+0.1860` (Target score `0.5080` vs Unrelated `0.3220`)
- **Diagnostic Verdict**: PASSED: Target chunk 'digital_lending_compliance_note_html_tokenaware_002' achieved Rank 2 (Score: 0.5080), outranking negative control by margin of +0.1860.

**Top-3 Retrieved Chunks**:

| Rank | Chunk ID | Cosine Score | Governing Section | Document Source |
| :---: | :--- | :---: | :--- | :--- |
| `#1` | `digital_lending_compliance_note_html_tokenaware_001` | `0.5385` | Preamble / Document Header | `digital_lending_compliance_note.html` |
| `#2` | `digital_lending_compliance_note_html_tokenaware_002` | `0.5080` | 3. Code of Conduct for Recove... | `digital_lending_compliance_note.html` |
| `#3` | `circular_dor_2024_108_txt_tokenaware_004` | `0.3220` | 5. Record Retention Obligations | `circular_dor_2024_108.txt` |

---

### Deep Dive: Analysis of Surprising & Borderline Retrieval Cases (Task 3)

Empirical testing across the prepared banking regulatory corpus revealed two key architectural phenomena that provide crucial insight into the behavior of dense embedding bi-encoders:

#### 1. The Politically Exposed Persons (PEP) Approval Authority Failure (`SANITY-05`)
- **Query**: *"Who must approve establishing a banking relationship with a Politically Exposed Person?"*
- **Ground Truth Target**: `circular_dor_2024_108_txt_tokenaware_003` (Section 3: Enhanced Due Diligence for High-Risk Accounts and PEPs).
  - *Direct Substantive Answer*: *"Establishing relationships with PEPs, their family members, or close associates requires written approval from an officer not below the rank of Deputy General Manager."*
- **Empirical Retrieval Outcome**:
  - **Rank 1 (Score 0.4681)**: `circular_dor_2024_108_txt_tokenaware_002` (2. Customer Due Diligence Requirements)
  - **Rank 2 (Score 0.4681)**: `sample_regulatory_circular_txt_tokenaware_002` (Duplicate circular CDD section)
  - **Rank 3 (Score 0.4390)**: `guidelines_cdd_pml_rules_md_tokenaware_002` (Customer Risk Categorization)
  - **Rank 4 (Score 0.4263)**: `circular_dor_2024_108_txt_tokenaware_001` (Document Header/Preamble)
  - **Rank 6 (Score 0.4071)**: **Target Ground Truth Chunk (`tokenaware_003`)**
- **Why Did This Happen? (Root Cause Analysis)**:
  1. **Semantic Dominance of Common Phrases**: The query contains the phrase *"establishing a banking relationship"*. In the embedding space of `all-minilm`, this phrase shares dense semantic overlap with general account-opening procedures described extensively in Section 2 (*"establish an account-based relationship"*).
  2. **Bi-Encoder Vector Compression Without Cross-Attention**: Dense bi-encoders independently project the entire query and entire chunk into fixed 384-dimensional vectors. Because Section 3 also discusses trusts, quarterly reviews, and source of funds, the specific DGM approval token vector is diluted across the 300-token chunk.
  3. **Lack of Keyword Hard-Matching**: The query specifically asks *"Who must approve"* (a governance question). Dense embeddings without BM25 lexical boosting or cross-encoder re-ranking cannot prioritize the keyword match for *"Deputy General Manager approval"*.
- **Architectural Remedy for RAG Pipeline**:
  - Implement a **Hybrid Retrieval (BM25 + Dense)** pipeline. BM25 directly indexes the exact entity *"Politically Exposed Person"* and *"approval"*, lifting the chunk into the candidate pool.
  - Apply a **Cross-Encoder Re-Ranker** (e.g. `bge-reranker-base`) on the top-10 candidates. Cross-encoders perform full cross-attention between every query word and document word, instantly promoting the DGM answer chunk to Rank 1.

#### 2. The Preamble Attraction / Inversion Phenomenon (`SANITY-06`)
- **Query**: *"What hours are recovery agents permitted to call borrowers and what conduct is prohibited?"*
- **Target Clause**: `digital_lending_compliance_note_html_tokenaware_002` (Section 3: Code of Conduct for Recovery Agents - contains exact 08:00 to 19:00 hours).
- **Empirical Retrieval Outcome**:
  - **Rank 1 (Score 0.5385)**: `digital_lending_compliance_note_html_tokenaware_001` (Preamble / Document Header)
  - **Rank 2 (Score 0.5080)**: `digital_lending_compliance_note_html_tokenaware_002` (Operative Recovery Agent Section)
- **Why Did This Happen?**:
  - The document preamble acts as a summary abstract, mentioning digital lending platforms, loan recovery, borrower rights, and fair practices all in one concentrated paragraph.
  - This summary nature gives it artificially high cosine similarity against multiple user queries compared to the narrow substantive section below it.
- **Architectural Remedy**:
  - Add section hierarchy metadata filtering or chunk-type penalties for document preambles during semantic search.

---

## 4. Key Architectural Takeaways for RegulSense

1. **Strong Global Discrimination**: For clean cross-domain queries (e.g. CDD vs Cyber Resilience), the dense embedding model produces large positive separation margins (`+0.30` to `+0.38`), reliably pushing unrelated texts to the bottom ranks.
2. **The High-Level Overview Trap**: Preamble chunks frequently collect high cosine similarity across diverse queries because they contain high-density lexical summaries of multiple topics.
3. **Entity and Threshold Blindness in Pure Bi-Encoders**: Fine-grained governance rules (e.g. *"Deputy General Manager written approval"*) and numerical thresholds (e.g. *"15% for partnership firms"*) can be obscured by broader account-opening jargon unless lexical search (BM25) or cross-encoder re-ranking is introduced.

---
*Report automatically generated by `src/embedding_quality_checker.py` for RegulSense Banking Compliance Assistant.*