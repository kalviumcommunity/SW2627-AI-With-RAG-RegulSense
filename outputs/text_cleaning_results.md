# RegulSense: Text Cleaning & Retrieval Normalization Report

- **Execution Timestamp**: 2026-09-07 14:03:15
- **Total Corpus Documents Cleaned**: 4
- **Total Raw Characters**: 11,022
- **Total Cleaned Characters**: 10,930
- **Noise Removed**: 92 characters (0.83%)
- **Status**: Uniform Cleaning Pipeline Verified Across Corpus

---

## 1. Executive Summary

The RegulSense text cleaning pipeline prepares ingested regulatory documents for high-accuracy dense vector retrieval. It eliminates four primary noise vectors:
1. **Boilerplate & Running Headers/Footers**: Page counters (`Page 1 of 5`), classification stamps (`Confidential Internal Copy`), web navigation breadcrumbs (`Home > ...`), and legal disclaimers.
2. **Broken Line Wraps & Hyphenation**: Words split across line boundaries (`trans-\naction` -> `transaction`) are reunited, restoring natural phrase embeddings.
3. **Unicode NFKC & Encoding Artifacts**: Harmonizes smart curly quotes (`“` / `”`), em/en dashes (`—` / `–`), ligatures (`ﬁ` -> `fi`), non-breaking spaces, and zero-width BOMs.
4. **Whitespace Normalization**: Collapses runaway multiple spaces and tabs into a single space and vertical blank lines (3+ newlines collapsed to 2).

---

## 2. Corpus-Wide Uniform Application Metrics (Task 3)

| # | Document Filename | Format | Raw Chars | Cleaned Chars | Chars Removed | Reduction % | Cleaned Words |
|---|---|---|---|---|---|---|---|
| 1 | `circular_dor_2024_108.txt` | `.txt` | 4,747 | 4,747 | 0 | 0.0% | 647 |
| 2 | `cyber_resilience_framework.pdf` | `.pdf` | 2,167 | 2,167 | 0 | 0.0% | 292 |
| 3 | `digital_lending_compliance_note.html` | `.html` | 1,708 | 1,624 | 84 | 4.92% | 222 |
| 4 | `guidelines_cdd_pml_rules.md` | `.md` | 2,400 | 2,392 | 8 | 0.33% | 349 |

---

## 3. Targeted Before / After Case Evidence (Task 4)

### Case 1: Broken Line Wraps & Hyphenation Splits (PDF Extracted)
- **Category**: Hyphenation & Soft Line Wraps
- **Raw**: 171 chars (26 words, 4 lines)
- **Cleaned**: 165 chars (23 words, 1 lines)
- **Reduction**: 6 chars removed (3.51%)

#### [BEFORE: RAW EXTRACTED TEXT]
```text
Under Section 35A of the Banking Regulation Act, 1949, all regu-
latory entities must enforce auto-
mated trans-
action monitoring systems to detect suspicious fund flows.
```

#### [AFTER: RETRIEVAL-READY CLEANED TEXT]
```text
Under Section 35A of the Banking Regulation Act, 1949, all regulatory entities must enforce automated transaction monitoring systems to detect suspicious fund flows.
```

### Case 2: Repeated Page Numbers & Confidentiality Boilerplate
- **Category**: Boilerplate & Headers/Footers
- **Raw**: 318 chars (56 words, 7 lines)
- **Cleaned**: 170 chars (27 words, 1 lines)
- **Reduction**: 148 chars removed (46.54%)

#### [BEFORE: RAW EXTRACTED TEXT]
```text
CONFIDENTIAL - FOR INTERNAL USE ONLY
Page 1 of 12
The bank shall determine the natural person who ultimately owns or controls a customer.
Page 1 of 12
Department of Supervision • Reserve Bank of India • Confidential Internal Copy
- 1 -
Beneficial own
```

#### [AFTER: RETRIEVAL-READY CLEANED TEXT]
```text
The bank shall determine the natural person who ultimately owns or controls a customer. Beneficial ownership threshold is pegged at 10 percent of shares or voting rights.
```

### Case 3: Web Navigation Breadcrumbs & Footer Disclaimers (HTML)
- **Category**: Navigation & Legal Boilerplate
- **Raw**: 261 chars (44 words, 9 lines)
- **Cleaned**: 90 chars (14 words, 1 lines)
- **Reduction**: 171 chars removed (65.52%)

#### [BEFORE: RAW EXTRACTED TEXT]
```text
Home > Circulars > Master Directions > AML
Skip to main content
Navigation Menu

All loan disbursals must be executed solely between the borrower and the regulated entity.

Back to top
Print this page
Copyright © 2024 Reserve Bank of India. All right
```

#### [AFTER: RETRIEVAL-READY CLEANED TEXT]
```text
All loan disbursals must be executed solely between the borrower and the regulated entity.
```

### Case 4: Unicode NFKC, Smart Quotes, Dashes, Ligatures & Zero-Width Spaces
- **Category**: Unicode & Character Encoding
- **Raw**: 164 chars (20 words, 2 lines)
- **Cleaned**: 164 chars (20 words, 1 lines)
- **Reduction**: 0 chars removed (0.0%)

#### [BEFORE: RAW EXTRACTED TEXT]
```text
The “Enhanced Due Diligence” guidelines—issued in 2024—require ‘high-risk’
PEPs to provide corroborating ﬁnancial statements (noting ​a​ ​threshold of INR 50,000…).
```

#### [AFTER: RETRIEVAL-READY CLEANED TEXT]
```text
The "Enhanced Due Diligence" guidelines-issued in 2024-require 'high-risk' PEPs to provide corroborating financial statements (noting a threshold of INR 50,000...).
```

### Case 5: Runaway Blank Lines & Multiple Spaces
- **Category**: Whitespace Normalization
- **Raw**: 151 chars (18 words, 11 lines)
- **Cleaned**: 124 chars (18 words, 5 lines)
- **Reduction**: 27 chars removed (17.88%)

#### [BEFORE: RAW EXTRACTED TEXT]
```text
Risk   Rating:    Tier 1 (Low).





Review   frequency   is   every   10   years   for   salaried   individuals.



Standard OVD verification applies.
```

#### [AFTER: RETRIEVAL-READY CLEANED TEXT]
```text
Risk Rating: Tier 1 (Low).

Review frequency is every 10 years for salaried individuals.

Standard OVD verification applies.
```

---

## 4. Corpus Document Before / After Samples

### Corpus Document 1: `circular_dor_2024_108.txt` (.txt)
- **Citation**: `[Source: circular_dor_2024_108.txt]`
- **Volume**: 4,747 raw chars -> 4,747 cleaned chars (0.0% noise eliminated)

**Cleaned Text Sample (First 280 characters):**
```text
RESERVE BANK OF INDIA FINANCIAL STABILITY AND COMPLIANCE DEPARTMENT CENTRAL OFFICE, MUMBAI Ref: RBI/2023-24/108 Circular No: DOR.AML.REC.66/14.01.001/2023-24 Date: January 04, 2024 To, All Scheduled Commercial Banks (excluding RRBs) All Small Finance Banks and Payment Banks All N...
```

### Corpus Document 2: `cyber_resilience_framework.pdf` (.pdf)
- **Citation**: `[Source: cyber_resilience_framework.pdf, Pages: 2]`
- **Volume**: 2,167 raw chars -> 2,167 cleaned chars (0.0% noise eliminated)

**Cleaned Text Sample (First 280 characters):**
```text
RESERVE BANK OF INDIA DEPARTMENT OF CYBER SECURITY AND INFORMATION TECHNOLOGY CENTRAL OFFICE, MUMBAI Circular No: RBI/2024-25/19 - DoS.CO.CSITE.No.03/11.01.005/2024-25 Date: March 12, 2024 Subject: Master Direction on Cyber Resilience and Digital Payment Security Controls 1. Mand...
```

### Corpus Document 3: `digital_lending_compliance_note.html` (.html)
- **Citation**: `[Source: digital_lending_compliance_note.html]`
- **Volume**: 1,708 raw chars -> 1,624 cleaned chars (4.92% noise eliminated)

**Cleaned Text Sample (First 280 characters):**
```text
Regulatory Advisory: Digital Lending Guidelines & Consumer Protection RegulSense Regulatory Advisory: Digital Lending & Fair Practices Code Reference: RBI/2022-23/DOR.CRE.REC.42/21.04.048/2022-23 Target Entities: Commercial Banks, Digital Lending Platforms, Lending Service Provid...
```

### Corpus Document 4: `guidelines_cdd_pml_rules.md` (.md)
- **Citation**: `[Source: guidelines_cdd_pml_rules.md]`
- **Volume**: 2,400 raw chars -> 2,392 cleaned chars (0.33% noise eliminated)

**Cleaned Text Sample (First 280 characters):**
```text
# Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules **Document Identifier**: REG-COMP-2024-V2 **Effective Date**: January 15, 2024 **Applicability**: Branch Operations, Credit Appraisal, and Compliance Teams **Supervisory Authority**: Reserve Bank o...
```
