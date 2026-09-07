# RegulSense: Document Intake & Corpus Verification Report

- **Execution Timestamp**: 2026-09-07 13:46:29
- **Total Files Examined**: 4
- **Successfully Ingested**: 4
- **Skipped / Failed**: 0
- **Total Volume Ingested**: 11,021 characters (1,522 words)
- **Status**: Document Loading Pipeline Verified Across Multi-Format Corpus

---

## 1. Executive Summary & Verification

The RegulSense Document Loader normalizes heterogeneous compliance assets (PDF, TXT, HTML, Markdown) into a unified plain-text `Document` representation with full provenance preservation and citation readiness. Missing, corrupt, or unsupported files are safely caught and documented without interrupting pipeline execution.

---

## 2. Successfully Ingested Corpus

| # | Document Filename | Format | Characters | Words | Citation Reference | Title / Details |
|---|---|---|---|---|---|---|
| 1 | `circular_dor_2024_108.txt` | `.txt` | 4,747 | 647 | \[Source: circular_dor_2024_108.txt\] | Pages: 1 |
| 2 | `cyber_resilience_framework.pdf` | `.pdf` | 2,167 | 292 | \[Source: cyber_resilience_framework.pdf, Pages: 2\] | Pages: 2 |
| 3 | `digital_lending_compliance_note.html` | `.html` | 1,707 | 234 | \[Source: digital_lending_compliance_note.html\] | Regulatory Advisory: Digital Lending Guidelines & Consumer Protection |
| 4 | `guidelines_cdd_pml_rules.md` | `.md` | 2,400 | 349 | \[Source: guidelines_cdd_pml_rules.md\] | Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules |

---

## 3. Sample Intake Confirmation (Previews)

### Document 1: `circular_dor_2024_108.txt`
- **Source**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\circular_dor_2024_108.txt`
- **Format**: `.txt` | **Length**: 4,747 characters | **Words**: 647
- **Citation**: `[Source: circular_dor_2024_108.txt]`

```text
RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTMENT CENTRAL OFFICE, MUMBAI Ref: RBI/2023-24/108 Circular No: DOR.AML.REC.66/14.01.001/2023-24 Date: January 04, 2024 To, All Scheduled Commercial Banks (exc...
```

### Document 2: `cyber_resilience_framework.pdf`
- **Source**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\cyber_resilience_framework.pdf`
- **Format**: `.pdf` | **Length**: 2,167 characters | **Words**: 292
- **Citation**: `[Source: cyber_resilience_framework.pdf, Pages: 2]`

```text
RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cy...
```

### Document 3: `digital_lending_compliance_note.html`
- **Source**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\digital_lending_compliance_note.html`
- **Format**: `.html` | **Length**: 1,707 characters | **Words**: 234
- **Citation**: `[Source: digital_lending_compliance_note.html]`

```text
Regulatory Advisory: Digital Lending Guidelines & Consumer Protection RegulSense Regulatory Advisory: Digital Lending & Fair Practices Code Reference: RBI/2022-23/DOR.CRE.REC.42/21.04.048/2022-23 Target Entities: Commerc...
```

### Document 4: `guidelines_cdd_pml_rules.md`
- **Source**: `C:\Users\uppal\OneDrive\Desktop\WI-sprint-2\SW2627-AI-With-RAG-RegulSense\data\sample_corpus\guidelines_cdd_pml_rules.md`
- **Format**: `.md` | **Length**: 2,400 characters | **Words**: 349
- **Citation**: `[Source: guidelines_cdd_pml_rules.md]`

```text
# Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules **Document Identifier**: REG-COMP-2024-V2 **Effective Date**: January 15, 2024 **Applicability**: Branch Operations, Credit Appraisal, an...
```
