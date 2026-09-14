# RegulSense: Regulatory Corpus Vector Indexing Summary Report

- **Execution Timestamp**: `2026-09-14T12:13:35.755621+00:00`
- **Target Collection**: `regulsense_regulatory_chunks`
- **Storage Engine**: `ChromaDB PersistentClient` (`Persistent Storage`)
- **Persistence Directory**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\chroma_db`
- **Embedding Model**: `all-minilm` (Dimension: `384`)
- **Distance Metric**: `cosine` (`HNSW:space = cosine`)
- **Overall Pipeline Status**: **SUCCESS**

---

## 1. Executive Indexing Audit & Count Reconciliation (Task 1, 2, 3)

| Metric | Audit Value | Baseline Target | Status | Architectural Rule |
| :--- | :---: | :---: | :---: | :--- |
| **Chunks from Ingestion** | `15` | `15` | **PASSED** | Intake corpus text chunks produced |
| **Chunks from Embedding** | `15` | `15` | **PASSED** | Pre-computed 384-dim dense vectors |
| **Records Loaded** | `15` | `15` | **PASSED** | Source chunks bound with metadata |
| **Records Inserted** | `15` | `15` | **PASSED** | Upserted into vector database (Task 1) |
| **Final Collection Count** | `15` | `15` | **PASSED** | Total searchable records in index (Task 3) |
| **Count Reconciliation** | `EXACT MATCH` | `True` | **PASSED** | 1:1 parity between corpus and index |
| **Vector Dimension Uniformity** | `384` | `384` | **PASSED** | Calibrated to `all-minilm` embeddings |
| **Failures Detected** | `0` | `0` | **PASSED** | Zero error rate across batch insertions |
| **Unique Regulatory Documents** | `5` | `5` | **PASSED** | Complete corpus coverage |

---

## 2. Regulatory Document Corpus Distribution

| Source Regulatory Document | Stored Chunks | Token Range | Status |
| :--- | :---: | :---: | :---: |
| `circular_dor_2024_108.txt` | `4` chunks | `~300 tokens` | **INDEXED** |
| `cyber_resilience_framework.pdf` | `2` chunks | `~300 tokens` | **INDEXED** |
| `digital_lending_compliance_note.html` | `2` chunks | `~300 tokens` | **INDEXED** |
| `guidelines_cdd_pml_rules.md` | `3` chunks | `~300 tokens` | **INDEXED** |
| `sample_regulatory_circular.txt` | `4` chunks | `~300 tokens` | **INDEXED** |

---

## 3. Stored Record Schema Architecture (Task 2)

Every stored record strictly conforms to the RegulSense tripartite RAG schema binding dense semantics with retrieval provenance:

```text
+------------------------------------------------------------------------------------+
|                           INDEXED REGULATORY VECTOR RECORD                         |
+------------------------------------------------------------------------------------+
| 1. ID:              Unique chunk ID (e.g. 'circular_dor_2024_108_txt_tokenaware_001')|
| 2. EMBEDDING:       384-dimensional dense float coordinate vector                  |
| 3. DOCUMENT:        Verbatim regulatory source text chunk                          |
| 4. METADATA:        source_document, filename, chunk_index, section, page_number,  |
|                     document_id, token_count, file_type, strategy, relative_path   |
+------------------------------------------------------------------------------------+
```

---

## 4. Stored Record Spot-Check Integrity Audit (Task 4)

Representative records sampled across multiple regulatory documents were retrieved from ChromaDB storage and audited against their source embeddings and raw texts:

| Chunk Identifier | Source Document | Section Header | Page / Index | Vector Dim | Cosine Sim | Integrity Result |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `circular_dor_2024_108_txt_tokenaware_001` | `circular_dor_2024_108.txt` | Preamble / Document Header | P.1 / #0 | 384 | `1.000000` | **PASSED** |
| `circular_dor_2024_108_txt_tokenaware_002` | `circular_dor_2024_108.txt` | 2. Customer Due Diligence (CDD) Req... | P.1 / #1 | 384 | `1.000000` | **PASSED** |
| `circular_dor_2024_108_txt_tokenaware_003` | `circular_dor_2024_108.txt` | 3. Enhanced Due Diligence (EDD) for... | P.1 / #2 | 384 | `1.000000` | **PASSED** |
| `circular_dor_2024_108_txt_tokenaware_004` | `circular_dor_2024_108.txt` | 5. Record Retention Obligations | P.1 / #3 | 384 | `1.000000` | **PASSED** |
| `cyber_resilience_framework_pdf_tokenaware_001` | `cyber_resilience_framework.pdf` | Preamble / Document Header | P.1 / #0 | 384 | `1.000000` | **PASSED** |
| `cyber_resilience_framework_pdf_tokenaware_002` | `cyber_resilience_framework.pdf` | Preamble / Document Header | P.1 / #1 | 384 | `1.000000` | **PASSED** |
| `digital_lending_compliance_note_html_tokenaware_001` | `digital_lending_compliance_note.html` | Preamble / Document Header | P.1 / #0 | 384 | `1.000000` | **PASSED** |
| `digital_lending_compliance_note_html_tokenaware_002` | `digital_lending_compliance_note.html` | 3. Code of Conduct for Recovery Age... | P.1 / #1 | 384 | `1.000000` | **PASSED** |
| `guidelines_cdd_pml_rules_md_tokenaware_001` | `guidelines_cdd_pml_rules.md` | Internal Compliance Guidelines: CDD... | P.1 / #0 | 384 | `1.000000` | **PASSED** |
| `guidelines_cdd_pml_rules_md_tokenaware_002` | `guidelines_cdd_pml_rules.md` | 2. Customer Risk Categorization | P.1 / #1 | 384 | `1.000000` | **PASSED** |
| `guidelines_cdd_pml_rules_md_tokenaware_003` | `guidelines_cdd_pml_rules.md` | 4. Beneficial Ownership Thresholds | P.1 / #2 | 384 | `1.000000` | **PASSED** |
| `sample_regulatory_circular_txt_tokenaware_001` | `sample_regulatory_circular.txt` | Preamble / Document Header | P.1 / #0 | 384 | `1.000000` | **PASSED** |
| `sample_regulatory_circular_txt_tokenaware_002` | `sample_regulatory_circular.txt` | 2. Customer Due Diligence (CDD) Req... | P.1 / #1 | 384 | `1.000000` | **PASSED** |
| `sample_regulatory_circular_txt_tokenaware_003` | `sample_regulatory_circular.txt` | 3. Enhanced Due Diligence (EDD) for... | P.1 / #2 | 384 | `1.000000` | **PASSED** |
| `sample_regulatory_circular_txt_tokenaware_004` | `sample_regulatory_circular.txt` | 5. Record Retention Obligations | P.1 / #3 | 384 | `1.000000` | **PASSED** |

### Detailed Spot-Check Verifications

#### Record 1: `circular_dor_2024_108_txt_tokenaware_001`

- **Source Regulatory File**: `circular_dor_2024_108.txt`
- **Governing Section**: `Preamble / Document Header`
- **Page Number**: `1` | **Chunk Index**: `0`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'circular_dor_2024_108_txt_tokenaware_001' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 2: `circular_dor_2024_108_txt_tokenaware_002`

- **Source Regulatory File**: `circular_dor_2024_108.txt`
- **Governing Section**: `2. Customer Due Diligence (CDD) Requirements`
- **Page Number**: `1` | **Chunk Index**: `1`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'circular_dor_2024_108_txt_tokenaware_002' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 3: `circular_dor_2024_108_txt_tokenaware_003`

- **Source Regulatory File**: `circular_dor_2024_108.txt`
- **Governing Section**: `3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs`
- **Page Number**: `1` | **Chunk Index**: `2`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'circular_dor_2024_108_txt_tokenaware_003' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 4: `circular_dor_2024_108_txt_tokenaware_004`

- **Source Regulatory File**: `circular_dor_2024_108.txt`
- **Governing Section**: `5. Record Retention Obligations`
- **Page Number**: `1` | **Chunk Index**: `3`
- **Token Count**: `231` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'circular_dor_2024_108_txt_tokenaware_004' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 5: `cyber_resilience_framework_pdf_tokenaware_001`

- **Source Regulatory File**: `cyber_resilience_framework.pdf`
- **Governing Section**: `Preamble / Document Header`
- **Page Number**: `1` | **Chunk Index**: `0`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'cyber_resilience_framework_pdf_tokenaware_001' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 6: `cyber_resilience_framework_pdf_tokenaware_002`

- **Source Regulatory File**: `cyber_resilience_framework.pdf`
- **Governing Section**: `Preamble / Document Header`
- **Page Number**: `1` | **Chunk Index**: `1`
- **Token Count**: `206` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'cyber_resilience_framework_pdf_tokenaware_002' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 7: `digital_lending_compliance_note_html_tokenaware_001`

- **Source Regulatory File**: `digital_lending_compliance_note.html`
- **Governing Section**: `Preamble / Document Header`
- **Page Number**: `1` | **Chunk Index**: `0`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'digital_lending_compliance_note_html_tokenaware_001' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 8: `digital_lending_compliance_note_html_tokenaware_002`

- **Source Regulatory File**: `digital_lending_compliance_note.html`
- **Governing Section**: `3. Code of Conduct for Recovery Agents`
- **Page Number**: `1` | **Chunk Index**: `1`
- **Token Count**: `87` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'digital_lending_compliance_note_html_tokenaware_002' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 9: `guidelines_cdd_pml_rules_md_tokenaware_001`

- **Source Regulatory File**: `guidelines_cdd_pml_rules.md`
- **Governing Section**: `Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules`
- **Page Number**: `1` | **Chunk Index**: `0`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'guidelines_cdd_pml_rules_md_tokenaware_001' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 10: `guidelines_cdd_pml_rules_md_tokenaware_002`

- **Source Regulatory File**: `guidelines_cdd_pml_rules.md`
- **Governing Section**: `2. Customer Risk Categorization`
- **Page Number**: `1` | **Chunk Index**: `1`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'guidelines_cdd_pml_rules_md_tokenaware_002' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 11: `guidelines_cdd_pml_rules_md_tokenaware_003`

- **Source Regulatory File**: `guidelines_cdd_pml_rules.md`
- **Governing Section**: `4. Beneficial Ownership Thresholds`
- **Page Number**: `1` | **Chunk Index**: `2`
- **Token Count**: `69` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'guidelines_cdd_pml_rules_md_tokenaware_003' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 12: `sample_regulatory_circular_txt_tokenaware_001`

- **Source Regulatory File**: `sample_regulatory_circular.txt`
- **Governing Section**: `Preamble / Document Header`
- **Page Number**: `1` | **Chunk Index**: `0`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'sample_regulatory_circular_txt_tokenaware_001' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 13: `sample_regulatory_circular_txt_tokenaware_002`

- **Source Regulatory File**: `sample_regulatory_circular.txt`
- **Governing Section**: `2. Customer Due Diligence (CDD) Requirements`
- **Page Number**: `1` | **Chunk Index**: `1`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'sample_regulatory_circular_txt_tokenaware_002' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 14: `sample_regulatory_circular_txt_tokenaware_003`

- **Source Regulatory File**: `sample_regulatory_circular.txt`
- **Governing Section**: `3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs`
- **Page Number**: `1` | **Chunk Index**: `2`
- **Token Count**: `300` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'sample_regulatory_circular_txt_tokenaware_003' verified with 100% integrity (cosine sim: 1.000000)**

#### Record 15: `sample_regulatory_circular_txt_tokenaware_004`

- **Source Regulatory File**: `sample_regulatory_circular.txt`
- **Governing Section**: `5. Record Retention Obligations`
- **Page Number**: `1` | **Chunk Index**: `3`
- **Token Count**: `231` tokens
- **Vector Dimensionality**: `384` coordinates (Expected: `384`)
- **Vector Coordinate Fidelity**: `1.000000` cosine similarity to original embedding
- **ID Match**: `True` | **Verbatim Text Match**: `True` | **Metadata Match**: `True`
- **Audit Status**: **PASSED: Record 'sample_regulatory_circular_txt_tokenaware_004' verified with 100% integrity (cosine sim: 1.000000)**

---

## 5. Summary & Operational Readiness (Task 5)

- **Zero Information Loss**: All 15 corpus chunks are loaded with 100% vector coordinate fidelity and exact text preservation.
- **Retrieval Ready**: The collection `regulsense_regulatory_chunks` is fully indexed and operational for cosine similarity search, nearest-neighbor retrieval, and metadata filtering.
- **Audit Complete**: Verified zero insertion failures and exact count parity with upstream pipeline stages.

---
*Report automatically generated by `src/corpus_indexer.py` for RegulSense Banking Compliance Assistant.*