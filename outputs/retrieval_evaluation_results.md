# RegulSense: Systematic Retrieval Quality Evaluation & Failure Diagnostics Report

- **Target Database**: `ChromaDB PersistentClient`
- **Target Collection**: `regulsense_regulatory_chunks`
- **Embedding Model**: `all-minilm` (Dimension: 384)
- **Execution Timestamp**: `2026-09-14T14:43:09.017038+00:00`
- **Total Labelled Queries Evaluated**: `8`

---

## 1. Executive Summary & Aggregate Retrieval Metrics (Tasks 2 & 3)

The evaluation script measured empirical retrieval performance across the 8-query labelled dataset:

| Metric | Value | Technical Meaning |
| :--- | :---: | :--- |
| **Recall @ 1** | **`52.1%`** | Fraction of ground-truth chunks captured in the Rank #1 position |
| **Recall @ 3** | **`100.0%`** | Fraction of ground-truth chunks captured within top-3 candidates |
| **Recall @ 5** | **`100.0%`** | Upper-bound candidate completeness across the expanded context window |
| **Precision @ 1** | **`75.0%`** | Accuracy of the top-ranked retrieved chunk |
| **Precision @ 3** | **`58.3%`** | Concentration of relevant regulatory chunks delivered to the generator |
| **Precision @ 5** | **`35.0%`** | Noise dilution across the broader candidate set |
| **Mean Reciprocal Rank (MRR)** | **`0.8125`** | Average speed at which the first relevant chunk appears (1.0 = always #1) |
| **Mean Average Precision (MAP)** | **`0.9375`** | Comprehensive multi-threshold ranking quality |

---

## 2. Labelled Benchmark Dataset Specification (Task 1)

| ID | Domain | Query Text | Type | Difficulty | Target Chunk IDs |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **LQ1** | **Digital Lending Conduct** | "What are the permitted contact hours for recovery agents contacting borrowers under digital lending guidelines?" | `standard` | Easy | `digital_lending_compliance_note_html_tokenaware_002`<br>`digital_lending_compliance_note_html_tokenaware_001` |
| **LQ2** | **Cyber Incident Reporting** | "What is the mandatory timeframe for banks to report Severity 1 cyber security incidents to CERT-In and RBI?" | `standard` | Easy | `cyber_resilience_framework_pdf_tokenaware_001` |
| **LQ3** | **KYC / V-CIP Onboarding** | "What automated facial match confidence score and liveliness checks are required for V-CIP customer onboarding?" | `standard` | Easy | `circular_dor_2024_108_txt_tokenaware_002`<br>`sample_regulatory_circular_txt_tokenaware_002` |
| **LQ4** | **AML Record Retention** | "How many years must banks retain customer KYC and transaction records following account closure or cessation of business?" | `standard` | Easy | `circular_dor_2024_108_txt_tokenaware_004`<br>`sample_regulatory_circular_txt_tokenaware_004`<br>`guidelines_cdd_pml_rules_md_tokenaware_003` |
| **LQ5** | **Beneficial Ownership (BO)** | "What is the controlling ownership threshold for identifying beneficial owners of corporate customers under PML Rules?" | `standard` | Easy | `guidelines_cdd_pml_rules_md_tokenaware_001` |
| **LQ6** | **Digital Lending (Colloquial)** | "Can debt collectors or recovery agents call me late at night or on weekends to demand loan payments?" | `colloquial_phrasing` | Medium | `digital_lending_compliance_note_html_tokenaware_002`<br>`digital_lending_compliance_note_html_tokenaware_001` |
| **LQ7** | **Cross-Domain Statutory Scope** | "What are the statutory retention rules and timeline requirements for bank records?" | `cross_domain_ambiguity` | Hard | `circular_dor_2024_108_txt_tokenaware_004`<br>`sample_regulatory_circular_txt_tokenaware_004`<br>`guidelines_cdd_pml_rules_md_tokenaware_003` |
| **LQ8** | **Prudential Capital Norms (Negative Control)** | "What are the minimum Common Equity Tier 1 (CET1) capital adequacy ratio requirements under Basel III norms?" | `out_of_corpus` | Hard | *None (Negative Control)* |

---

## 3. Query-by-Query Retrieval Performance Ledger (Tasks 2 & 3)

| ID | Domain | Recall@1 | Recall@3 | Prec@1 | Prec@3 | MRR | Status | Top Retrieved Chunk (Score) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LQ1** | Digital Lending Conduc | `50%` | `100%` | `100%` | `67%` | `1.00` | ✅ **PASSED** | `digital_lending_compliance_note_html_tokenaware_001` (0.6013) |
| **LQ2** | Cyber Incident Reporti | `100%` | `100%` | `100%` | `33%` | `1.00` | ✅ **PASSED** | `cyber_resilience_framework_pdf_tokenaware_001` (0.6520) |
| **LQ3** | KYC / V-CIP Onboarding | `50%` | `100%` | `100%` | `67%` | `1.00` | ✅ **PASSED** | `circular_dor_2024_108_txt_tokenaware_002` (0.5471) |
| **LQ4** | AML Record Retention | `33%` | `100%` | `100%` | `100%` | `1.00` | ✅ **PASSED** | `circular_dor_2024_108_txt_tokenaware_004` (0.6969) |
| **LQ5** | Beneficial Ownership ( | `0%` | `100%` | `0%` | `33%` | `0.50` | ✅ **PASSED** | `guidelines_cdd_pml_rules_md_tokenaware_002` (0.5964) |
| **LQ6** | Digital Lending (Collo | `50%` | `100%` | `100%` | `67%` | `1.00` | ✅ **PASSED** | `digital_lending_compliance_note_html_tokenaware_001` (0.4431) |
| **LQ7** | Cross-Domain Statutory | `33%` | `100%` | `100%` | `100%` | `1.00` | ✅ **PASSED** | `circular_dor_2024_108_txt_tokenaware_004` (0.7101) |
| **LQ8** | Prudential Capital Nor | `100%` | `100%` | `0%` | `0%` | `0.00` | ⚠️ **FLAGGED** | `guidelines_cdd_pml_rules_md_tokenaware_002` (0.3765) |

---

## 4. Failure Inspection & Root Cause Diagnostics (Task 4)

The evaluation script automatically audited `1` edge-case or failure scenarios:

### Failure Case 1: Query LQ8 (Prudential Capital Norms (Negative Control))
- **User Query**: *"What are the minimum Common Equity Tier 1 (CET1) capital adequacy ratio requirements under Basel III norms?"*
- **Difficulty / Type**: `Hard` / `out_of_corpus`
- **Observed Metrics**: Recall@3: `100.0%` | Precision@3: `0.0%` | MRR: `0.00`
- **Root Cause Category**: **`Out-of-Corpus Query`**

> **Diagnostic Analysis**: Query 'LQ8' seeks Basel III capital adequacy norms which do not exist in the currently indexed regulatory corpus (AML/KYC, Digital Lending, Cyber Resilience). Retriever returned peripheral chunks with similarity scores up to 0.3765.

> 🛠️ **Prescribed Remediation**: Apply a confidence score cutoff threshold (e.g. min_score >= 0.45) to cleanly reject out-of-corpus requests and trigger a graceful fallback/refusal before generation.

#### Retrieved Chunks at Time of Failure:
| Rank | Chunk ID | Similarity | Verdict | Source Document | Section |
| :---: | :--- | :---: | :---: | :--- | :--- |
| #1 | `guidelines_cdd_pml_rules_md_tokenaware_002` | `0.3765` | **`IRRELEVANT`** | `guidelines_cdd_pml_rules.md` | 2. Customer Risk Categ |
| #2 | `guidelines_cdd_pml_rules_md_tokenaware_001` | `0.3640` | **`IRRELEVANT`** | `guidelines_cdd_pml_rules.md` | Internal Compliance Gu |
| #3 | `circular_dor_2024_108_txt_tokenaware_001` | `0.3319` | **`IRRELEVANT`** | `circular_dor_2024_108.txt` | Preamble / Document He |

---

## 5. Remediation Roadmap for Production Deployment (Task 5)

- 1. Implement LLM Query Rewriting & Expansion for colloquial borrower queries (fixes Colloquial Vocabulary Gaps).
- 2. Enforce Pre-Retrieval Domain Intent Routing or Metadata Filtering (fixes Cross-Domain Ambiguity).
- 3. Enforce Two-Stage Cross-Scoring Re-Ranking to suppress administrative preambles in favor of operative clauses.
- 4. Apply Confidence Score Threshold Gating (min_score >= 0.42) to cleanly reject Out-of-Corpus queries.

---
*Report automatically generated by `src/retrieval_evaluator.py` for RegulSense Banking Compliance Assistant.*