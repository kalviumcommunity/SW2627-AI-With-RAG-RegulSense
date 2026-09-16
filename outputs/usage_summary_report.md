# RegulSense Query Caching, Structured Logging & Usage Analytics Report

## Executive Overview
This report summarizes query traffic, cache efficiency, token consumption, and cost economics for the **RegulSense RAG Intelligence Assistant**.

---

## 1. Key Performance Indicators

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Total Requests** | `7` | Total compliance inquiries processed |
| **Cache Hits** | `4` | Inquiries served instantly from in-memory cache |
| **Cache Misses** | `3` | Inquiries requiring full vector retrieval & LLM generation |
| **Cache Hit Rate** | **`57.1%`** | Proportion of inquiries accelerated by cache |
| **Avg Cache Miss Latency** | `0.9094s` | Full RAG pipeline response time |
| **Avg Cache Hit Latency** | **`0.0005s`** | Instant in-memory cache lookup time |
| **Latency Reduction** | **`100.0%`** | Performance acceleration from query caching |
| **Total Tokens Consumed** | `5,028` | `4,470` prompt + `558` completion |
| **Total Estimated Cost** | **`$0.00055`** | Actual LLM compute expenditure incurred |
| **Total Cost Saved** | **`$0.00079`** | Financial savings achieved via cache hits |

---

## 2. Request Status Breakdown

| Status | Count | Percentage |
| :--- | :---: | :---: |
| 🟢 **Success** | `5` | `71.4%` |
| 🟠 **Refusal** | `2` | `28.6%` |
| 🔴 **Error** | `0` | `0.0%` |

---

## 3. Detailed Request Audit Log (Sample Records)

| Req ID | Timestamp (UTC) | Query Preview | Cache | Status | Latency | Tokens | Cost | Saved |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `2897d9ee` | `04:07:34` | *What are the mandatory timefra...* | `MISS` | `success` | `2.656s` | `1008` | `$0.00024` | `$0.00000` |
| `30357fba` | `04:07:34` | *What are the mandatory timefra...* | `⚡ HIT` | `success` | `0.000s` | `1008` | `$0.00000` | `$0.00024` |
| `838aa857` | `04:07:34` | *What are the record retention ...* | `MISS` | `success` | `0.040s` | `874` | `$0.00021` | `$0.00000` |
| `e7053708` | `04:07:34` | *What are the record retention ...* | `⚡ HIT` | `success` | `0.000s` | `874` | `$0.00000` | `$0.00021` |
| `e5ea33b0` | `04:07:34` | *What are the capital reserve r...* | `MISS` | `refusal` | `0.032s` | `128` | `$0.00010` | `$0.00000` |
| `d4769868` | `04:07:34` | *What are the capital reserve r...* | `⚡ HIT` | `refusal` | `0.000s` | `128` | `$0.00000` | `$0.00010` |
| `723a105a` | `04:07:34` | *What are the mandatory timefra...* | `⚡ HIT` | `success` | `0.001s` | `1008` | `$0.00000` | `$0.00024` |

---

## 4. Economic Pricing Model Reference
- **Input Prompt Rate**: `$0.20 per 1M tokens`
- **Output Completion Rate**: `$0.80 per 1M tokens`
- **Cost Formula**: `Cost = (PromptTokens * InputRate + CompletionTokens * OutputRate) / 1,000,000`
- **Cache Savings**: For every cache hit, full RAG LLM invocation is avoided, directly translating to **100% cost avoidance** for that inquiry.