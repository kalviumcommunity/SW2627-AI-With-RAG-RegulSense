# RegulSense Runtime Document Upload & Indexing Report

> **Pipeline Stage**: Runtime Knowledge Base Expansion  
> **Generated**: 2026-09-15 08:16:26 UTC  
> **Environment**: `production` | **ChromaDB Collection**: `regulsense_regulatory_chunks`

## 1. Overview & Objectives

This report confirms the implementation of runtime document upload and indexing capabilities for the RegulSense RAG system. New regulatory documents can now be submitted via HTTP multipart upload, processed through cleaning, token-aware chunking, dense embedding, and indexed directly into the active vector database—making them searchable immediately without restarting the application.

### Core Capabilities Verified
- **Safe Multipart Ingestion (Task 1)**: Sanitizes filenames to eliminate directory traversal attacks and stores files securely in `data/uploads`.
- **Complete Ingestion Pipeline (Task 2)**: Runs multi-format loading, text normalization, token-aware chunking with overlap, dense vector embedding generation, and ChromaDB upsert.
- **Runtime Searchability Without Restart (Task 3)**: Confirms that newly uploaded documents are immediately retrieved and cited by `/api/v1/query` in the active process.
- **Strict Validation & Error Resilience (Task 4)**: Rejects unsupported extensions, 0-byte empty files, and oversized payloads with clean HTTP status codes.
- **Reproducible Artifacts (Task 5)**: Full audit trail of upload requests, indexing responses, and downstream query verification.

---

## 2. API Endpoints Reference

| Method | Endpoint Path | Description | Content-Type | Supported Codes |
|:---|:---|:---|:---|:---:|
| `POST` | `/api/v1/upload` | Primary runtime document upload and indexing endpoint | `multipart/form-data` | `200`, `400`, `413`, `500` |
| `POST` | `/upload` | Convenience alias for upload endpoint | `multipart/form-data` | `200`, `400`, `413`, `500` |
| `POST` | `/api/v1/query` | RAG query endpoint retrieving both original and newly uploaded corpus content | `application/json` | `200`, `400`, `422`, `500` |

---

## 3. Upload & Indexing Audit Summary

### Sample Upload Metadata (`POST /api/v1/upload`):
```json
{
  "endpoint": "POST /api/v1/upload",
  "filename": "circular_dor_2024_lcr_framework.txt",
  "content_type": "text/plain",
  "file_size_bytes": 2456,
  "parameters": {
    "chunk_size": 300,
    "chunk_overlap": 50
  },
  "description": "Upload of RBI Master Direction on Basel III Liquidity Coverage Ratio (LCR) standards."
}
```

### Indexing Summary Response (`HTTP 200 OK`):
```json
{
  "status": "success",
  "filename": "circular_dor_2024_lcr_framework.txt",
  "stored_path": "C:\\Users\\uppal\\OneDrive\\Desktop\\WI-sprint-2\\SW2627-AI-With-RAG-RegulSense\\data\\uploads\\circular_dor_2024_lcr_framework.txt",
  "file_type": ".txt",
  "file_size_bytes": 2456,
  "raw_character_count": 2451,
  "cleaned_character_count": 2451,
  "chunks_created": 3,
  "records_indexed": 3,
  "collection_name": "regulsense_regulatory_chunks",
  "total_collection_records": 18,
  "chunk_ids": [
    "circular_dor_2024_lcr_framework_txt_tokenaware_001",
    "circular_dor_2024_lcr_framework_txt_tokenaware_002",
    "circular_dor_2024_lcr_framework_txt_tokenaware_003"
  ],
  "searchable_immediately": true,
  "metadata": {
    "latency_seconds": 0.2808,
    "embedding_model": "all-minilm",
    "chunk_size": 300,
    "chunk_overlap": 50,
    "timestamp": "2026-09-15T08:15:03.473663+00:00"
  }
}
```

---

## 4. Runtime Searchability Confirmation (Task 3)

Immediately following upload, the running application was queried for compliance guidelines regarding the newly uploaded document without any restart:

**Query**: *"What is the minimum Liquidity Coverage Ratio (LCR) requirement and run-off assumptions for stable and less stable retail deposits under RBI directives?"*

### Query Result (`HTTP 200 OK`):
```json
{
  "status": "success",
  "answer": "Based on the provided regulatory context, the minimum Liquidity Coverage Ratio (LCR) requirement is:\n\n* 100% on an ongoing daily basis, as stated in Section 2. Minimum Liquidity Coverage Ratio (LCR) Requirement of [1].\n\nRegarding run-off assumptions for retail deposits, the relevant information is found in Section 4. Run-off Rates for Deposits under 30-Day Stress of [3]:\n\n* Stable Retail Deposits fully insured by DICGC shall have a 5% run-off rate.\n* Less Stable Retail Deposits (including internet banking and high-value HNIs) shall be assigned a 10% run-off rate.\n\nIn summary:\n\n* Minimum LCR requirement: 100% [1]\n* Run-off assumptions for:\n\t+ Stable Retail Deposits: 5% [3]\n\t+ Less Stable Retail Deposits: 10% [3]\n\nPlease note that this answer strictly adheres to the provided regulatory context and source markers, without extrapolation or speculation.",
  "sources": [
    {
      "marker": "[1]",
      "source_document": "circular_dor_2024_lcr_framework.txt",
      "chunk_id": "circular_dor_2024_lcr_framework_txt_tokenaware_001",
      "section": "Preamble / Document Header",
      "page_number": 1,
      "similarity_score": 0.709,
      "verbatim_text": "RESERVE BANK OF INDIA DEPARTMENT OF REGULATION CENTRAL OFFICE, SHAHID BHAGAT SINGH ROAD, MUMBAI - 400 001\n\nCircular No: RBI/2024-25/44 - DoR.LRG.REC.22/21.04.098/2024-25 Date: May 18, 2024\n\nSubject: Master Direction - Basel III Framework on Liquidity Standards: Liquidity Coverage Ratio (LCR) and Run-off Assumptions\n\n1. Purpose and Scope\nThis Master Direction establishes enhanced liquidity resilience standards for Scheduled Commercial Banks (excluding Regional Rural Banks). The objective is to ensure that banks maintain an adequate stock of unencumbered High Quality Liquid Assets (HQLA) that can be converted into cash easily and immediately in private markets to meet their liquidity needs for a 30-calendar day liquidity stress scenario.\n\n2. Minimum Liquidity Coverage Ratio (LCR) Requirement\n(a) All covered banks must maintain a minimum Liquidity Coverage Ratio (LCR) of 100% on an ongoing daily basis.\n(b) The LCR is mathematically defined as the ratio of the Stock of High Quality Liquid Assets (HQLA) to Total Net Cash Outflows over the specified 30-calendar day stress horizon.\n(c) Any breach of the 100% minimum LCR threshold must be reported immediately, within 2 hours of occurrence, to the Chief General Manager-in-Charge, Department of"
    },
    {
      "marker": "[2]",
      "source_document": "circular_dor_2024_lcr_framework.txt",
      "chunk_id": "circular_dor_2024_lcr_framework_txt_tokenaware_002",
      "section": "2. Minimum Liquidity Coverage Ratio (LCR) Requirement",
      "page_number": 1,
      "similarity_score": 0.597,
      "verbatim_text": "Net Cash Outflows over the specified 30-calendar day stress horizon.\n(c) Any breach of the 100% minimum LCR threshold must be reported immediately, within 2 hours of occurrence, to the Chief General Manager-in-Charge, Department of Supervision, Reserve Bank of India, Mumbai.\n\n3. High Quality Liquid Assets (HQLA) Composition and Haircuts\n(a) Level 1 Assets: Cash in hand, excess Cash Reserve Ratio (CRR) balances held with RBI, and eligible government securities under the Facility to Avail Liquidity for Liquidity Coverage Ratio (FALLCR). Level 1 assets are included with zero percent (0%) haircut and no cap.\n(b) Level 2A Assets: Marketable securities issued or guaranteed by sovereigns or central banks with risk-weights of 20%, subject to a mandatory 15% haircut.\n(c) Level 2B Assets: High-quality corporate bonds rated BBB- to A and qualifying common equity shares, subject to a 50% haircut and capped at 15% of total HQLA.\n\n4. Run-off Rates for Deposits under 30-Day Stress\n(a) Stable Retail Deposits fully insured by DICGC shall have a 5% run-off rate.\n(b) Less Stable Retail Deposits (including internet banking and high-value HNIs) shall be assigned a 10% run-off rate.\n(c) Operational Deposits generated by clearing, custody, or cash management services shall carry a"
    },
    {
      "marker": "[3]",
      "source_document": "circular_dor_2024_lcr_framework.txt",
      "chunk_id": "circular_dor_2024_lcr_framework_txt_tokenaware_003",
      "section": "4. Run-off Rates for Deposits under 30-Day Stress",
      "page_number": 1,
      "similarity_score": 0.5266,
      "verbatim_text": "rate.\n(b) Less Stable Retail Deposits (including internet banking and high-value HNIs) shall be assigned a 10% run-off rate.\n(c) Operational Deposits generated by clearing, custody, or cash management services shall carry a 25% run-off factor.\n(d) Non-operational corporate deposits and wholesale funding without business relationship shall carry a 100% outflow assumption."
    }
  ],
  "citations": [
    "[1]",
    "[3]"
  ],
  "metadata": {
    "latency_seconds": 83.4851,
    "model": "llama3:latest",
    "top_k": 3,
    "is_refusal": false,
    "action_taken": "ANSWER",
    "guardrail_status": "SUFFICIENT_CONTEXT",
    "top_similarity_score": 0.709,
    "total_sources_returned": 3,
    "timestamp": "2026-09-15T08:16:26.967291+00:00",
    "prompt_tokens": 1302,
    "completion_tokens": 204
  }
}
```

**Verification Verdict**: The RAG pipeline retrieved the newly indexed chunk with high semantic similarity, passed the hallucination guardrail, and produced a grounded response with citations pointing directly to the uploaded file.

---

## 5. Input Validation & Error Handling Matrix (Task 4)

| Scenario | Request | Expected Status | Returned Error Code | Handling Rationale |
|:---|:---|:---:|:---|:---|
| **Unsupported Extension** | `.exe`, `.zip`, `.py`, `.bin` | `400 Bad Request` | `UNSUPPORTED_FORMAT` | Rejects non-regulatory formats with supported extension list. |
| **Empty File** | 0 bytes payload | `400 Bad Request` | `EMPTY_FILE` | Rejects empty documents to prevent indexing blank vectors. |
| **Oversized File** | `> 10 MB` payload | `413 Payload Too Large` | `FILE_TOO_LARGE` | Protects server memory and storage resources. |
| **Directory Traversal** | `../../malicious.txt` | `200 OK (Sanitized)` | `N/A` | Strips directory paths and stores safely as `malicious.txt`. |
| **Corrupted Binary** | Malformed PDF stream | `400 Bad Request` | `BAD_REQUEST` | Gracefully reports parser failure without crashing the service. |
