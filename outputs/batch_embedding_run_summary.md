# RegulSense: Scalable Batch Embedding Pipeline Run Summary

- **Target Embedding Model**: `all-minilm`
- **API Base URL**: `http://localhost:11434/v1`
- **Tokenizer Architecture**: `tiktoken` (`cl100k_base`)
- **Commercial Pricing Benchmark**: `$0.0200 / 1M input tokens`
- **Status**: **VERIFIED & OPERATIONAL (Tasks 1 - 5)**

---

## 1. Executive Demonstration Matrix (Tasks 1, 3 & 4)

The table below contrasts the **Initial Batch Run (Phase 1)** against the **Idempotent Re-Run (Phase 2)**, demonstrating batching throughput, financial cost accounting, and 100% duplicate work elimination:

| Operational Metric | Phase 1: Fresh Batch Ingestion | Phase 2: Re-Run Deduplication | Variance / Efficiency Gain |
| :--- | :---: | :---: | :--- |
| **Total Evaluated Chunks** | `15` | `15` | Full corpus evaluated in both passes |
| **Configured Batch Size** | `5` | `5` | Configurable grouping (Task 1) |
| **Total Batches Dispatched** | `3` | `0` | Batches sent to API endpoint |
| **Fresh Embeddings Generated** | `15` (100%) | `0` (0%) | **100% avoided redundant generation** |
| **Skipped Chunks (Cached)** | `0` (0%) | `15` (100%) | **Detected via `chunk_id` index (Task 4)** |
| **Failed Batches / Chunks** | `0` / `0` | `0` / `0` | Clean zero-failure executions |
| **Input Tokens Ingested** | `3,826` | `0` | Tokens counted via `tiktoken` |
| **Tokens Saved (Deduplicated)** | `0` | `3,826` | **3,826 tokens spared from re-ingestion** |
| **Approximate Embedding Cost** | **`$0.000077`** | **`$0.000000`** | Financial cost of run (Task 3) |
| **Cost Saved (Avoided Spend)** | `$0.000000` | **`$0.000077`** | **100% capital preserved on re-runs** |
| **Execution Latency** | `3.599s` | `0.023s` | **~50x faster execution using cache** |

---

## 2. Task 1 - Configurable Batch Embedding Analysis

- **Batch Partitioning**: Chunks are segmented into contiguous windows matching `batch_size` (configured to `5` chunks/batch).
- **Throughput Optimization**: Instead of dispatching 15 individual HTTP requests, the pipeline dispatched only **3 batched requests**, reducing network round-trip overhead by **80.0%**.
- **Configuration Precedence**: Supports dynamic configuration via CLI argument (`--batch-size`), environment variable (`EMBEDDING_BATCH_SIZE`), or programmatic parameter.

---

## 3. Task 2 - Backoff Retries & Failure Ledger Verification

The batch embedder protects against rate limiting (HTTP 429) and transient network disconnects using an exponential backoff retry loop with configurable attempts and backoff factor:

$$\text{Delay}(a) = \text{initial\_backoff} \times (\text{backoff\_factor})^{a-1}$$

- **Transient Exception Handlers**: Automatically catches `openai.RateLimitError`, `openai.APIConnectionError`, `openai.APITimeoutError`, `openai.InternalServerError`, and socket timeouts.
- **Max Retries Allowed**: `3` attempts per batch before declaring a failure.
- **Visible Batch Failure Ledger**: If retries are exhausted, the failure is appended to `failures` in `BatchRunSummary` with batch index, affected chunk IDs, error type, and timestamp.

### Diagnostic Ledger of Simulated Batch Failures

| Batch # | Impacted Chunk IDs | Failure Type | Error Description | Attempts | Status |
| :---: | :--- | :--- | :--- | :---: | :--- |
| `1` | `circular_dor_2024_108_txt_tokenaware_001, circular_dor_2024_108_txt_tokenaware_002 (+3 more)` | `RateLimitError` | Rate limit exceeded (demo)... | `3` | `LOGGED_IN_SUMMARY` |

---

## 4. Task 3 - Token Counting & Financial Cost Accounting

Every regulatory chunk's text is analyzed using the `tiktoken` (`cl100k_base`) BPE tokenizer:

| Accounting Dimension | Live Metric Value | Architectural Reference |
| :--- | :---: | :--- |
| **Total Processed Tokens** | `3,826` tokens | Measured exactly across all 15 chunks |
| **Average Tokens per Chunk** | `255` tokens | Prepared token-aware chunk size |
| **Commercial Pricing Benchmark** | `$0.0200 / 1M` | Benchmark equivalent (`text-embedding-3-small`) |
| **Total Ingestion Cost (USD)** | **`$0.000077`** | `(tokens / 1,000,000) * rate` |
| **Local Ollama Cost** | **`$0.000000`** | Free self-hosted offline inference |

---

## 5. Task 4 - Duplicate Work Detection & Chunk Skipping

- **Cache Identification**: On pipeline execution, the embedder scans `outputs/embedded_corpus_chunks.json` for previously embedded `chunk_id` keys.
- **Idempotency Guarantee**: Running the script multiple times produces zero new API calls when the corpus is unchanged.
- **Observed Savings**: On Phase 2 re-run, **15 of 15 chunks (100%)** were identified as cached, saving **3,826 tokens** and **$0.000077** in redundant inference expenditure.

---

## 6. Sample Processed Chunks Ledger

| Chunk ID | Source Document | Section | Vector Dim | Trimmed Vector (First 5 Values) | Provenance |
| :--- | :--- | :--- | :---: | :--- | :---: |
| `circular_dor_2024_108_txt_tokenaware_001` | `circular_dor_2024_108.txt` | Preamble / Document Header | `384` | `[-0.0144, 0.0004, -0.0392, -0.0525, -0.0022]` | Verified |
| `circular_dor_2024_108_txt_tokenaware_002` | `circular_dor_2024_108.txt` | 2. Customer Due Diligence... | `384` | `[-0.0711, -0.0280, -0.0266, -0.0616, 0.0413]` | Verified |
| `circular_dor_2024_108_txt_tokenaware_003` | `circular_dor_2024_108.txt` | 3. Enhanced Due Diligence... | `384` | `[-0.0219, -0.0297, -0.1296, -0.0012, 0.0285]` | Verified |
| `circular_dor_2024_108_txt_tokenaware_004` | `circular_dor_2024_108.txt` | 5. Record Retention Oblig... | `384` | `[-0.0789, 0.0154, -0.0014, -0.0275, -0.0379]` | Verified |
| `cyber_resilience_framework_pdf_tokenaware_001` | `cyber_resilience_framework.pdf` | Preamble / Document Header | `384` | `[-0.0121, -0.0051, -0.0741, -0.0082, 0.0371]` | Verified |
| `cyber_resilience_framework_pdf_tokenaware_002` | `cyber_resilience_framework.pdf` | Preamble / Document Header | `384` | `[0.0027, -0.0196, -0.0559, -0.0442, 0.0267]` | Verified |
| `digital_lending_compliance_note_html_tokenaware_001` | `digital_lending_compliance_note.html` | Preamble / Document Header | `384` | `[-0.0023, -0.0593, -0.0219, -0.0338, 0.0693]` | Verified |
| `digital_lending_compliance_note_html_tokenaware_002` | `digital_lending_compliance_note.html` | 3. Code of Conduct for Re... | `384` | `[-0.0700, 0.0433, 0.0075, -0.1063, 0.0104]` | Verified |
| `guidelines_cdd_pml_rules_md_tokenaware_001` | `guidelines_cdd_pml_rules.md` | Internal Compliance Guide... | `384` | `[-0.0189, -0.0213, -0.0501, -0.0127, -0.0130]` | Verified |
| `guidelines_cdd_pml_rules_md_tokenaware_002` | `guidelines_cdd_pml_rules.md` | 2. Customer Risk Categori... | `384` | `[-0.0827, -0.0250, 0.0081, -0.0398, -0.0197]` | Verified |

*... and 5 additional regulatory chunks.*

---

## 7. Sprint Summary & Verification Conclusion

- **Task 1 (Batching)**: Chunks embedded in configurable batches (`batch_size=5`), cutting API request volume from 15 calls down to 3.
- **Task 2 (Retries & Backoff)**: Exponential backoff wraps transient errors with customizable delay, retry ceilings, and visible failure tracking.
- **Task 3 (Totals & Cost)**: Full token ledger with `tiktoken` counting and exact dollar pricing ($0.000077 for full corpus).
- **Task 4 (Skipping)**: Deduplication detects pre-existing chunk embeddings and skips them on re-runs (0 API calls on Phase 2).
- **Task 5 (Artifacts)**: Output JSON and Markdown verification ledgers persisted to `outputs/`.

---
*Report automatically generated by `src/batch_embedder.py` for RegulSense Banking Compliance Assistant.*