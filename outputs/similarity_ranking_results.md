# RegulSense: Embedding Similarity Ranking & Query Retrieval Demonstration

- **Search Query**: *"What are the customer due diligence and KYC verification requirements for onboarding new bank accounts?"*
- **Target Embedding Model**: `all-minilm`
- **Embedding Vector Dimension**: `384`
- **Total Corpus Chunks Evaluated**: `15`
- **Chosen Similarity Metric**: **`Cosine Similarity`**

---

## 1. Top Most Similar Chunks (Highest Relevance)

These chunks exhibit the highest cosine similarity with the query's semantic intent:

| Rank | Cosine Similarity | Euclidean Dist | Chunk ID | Source Document | Section | Page | Text Preview |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :--- |
| **#1** | 🟢 **`0.6501`** | `0.8366` | `circular_dor_2024_108_txt_tokenaware_002` | `circular_dor_2024_108.txt` | 2. Customer Due Diligence | 1 | "Regulated entities must undertake client identification and verification procedures before..." |
| **#2** | 🟢 **`0.6501`** | `0.8366` | `sample_regulatory_circular_txt_tokenaware_002` | `sample_regulatory_circular.txt` | 2. Customer Due Diligence | 1 | "Regulated entities must undertake client identification and verification procedures before..." |
| **#3** | 🟢 **`0.5839`** | `0.9123` | `guidelines_cdd_pml_rules_md_tokenaware_001` | `guidelines_cdd_pml_rules.md` | Internal Compliance Guide | 1 | "# Internal Compliance Guidelines: CDD & Prevention of Money Laundering (PML) Rules  **Docu..." |

---

## 2. Least Similar Chunks (Unrelated / Different Domain)

These chunks show negligible or negative semantic correlation with the customer due diligence query:

| Rank | Cosine Similarity | Euclidean Dist | Chunk ID | Source Document | Section | Page | Text Preview |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :--- |
| **#13** | 🔴 **`0.4321`** | `1.0657` | `guidelines_cdd_pml_rules_md_tokenaware_002` | `guidelines_cdd_pml_rules.md` | 2. Customer Risk Categori | 1 | "2 years | PEPs, jewelers, bullion dealers, offshore trusts | Enhanced Due Diligence (EDD),..." |
| **#14** | 🔴 **`0.3569`** | `1.1341` | `digital_lending_compliance_note_html_tokenaware_002` | `digital_lending_compliance_note.html` | 3. Code of Conduct for Re | 1 | "or after 7:00 PM. Harassment, verbal intimidation, or unauthorized contact with borrowers'..." |
| **#15** | 🔴 **`0.3529`** | `1.1376` | `guidelines_cdd_pml_rules_md_tokenaware_003` | `guidelines_cdd_pml_rules.md` | 4. Beneficial Ownership T | 1 | "Trusts**: Natural person(s) who hold more than **15%** of property, capital, or profits.  ..." |

---

## 3. Full Corpus Ranking Ledger

Complete ranking of all evaluated chunks in descending order of semantic relevance:

| Rank | Score | Chunk ID | Document | Section | Page | Chk Idx | Tokens |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| #1 | **`0.6501`** | `circular_dor_2024_108_txt_tokenaware_002` | `circular_dor_2024_108.txt` | 2. Customer Due Diligence (C | 1 | 1 | 300 |
| #2 | **`0.6501`** | `sample_regulatory_circular_txt_tokenaware_002` | `sample_regulatory_circular.txt` | 2. Customer Due Diligence (C | 1 | 1 | 300 |
| #3 | **`0.5839`** | `guidelines_cdd_pml_rules_md_tokenaware_001` | `guidelines_cdd_pml_rules.md` | Internal Compliance Guidelin | 1 | 0 | 300 |
| #4 | **`0.5660`** | `circular_dor_2024_108_txt_tokenaware_001` | `circular_dor_2024_108.txt` | Preamble / Document Header | 1 | 0 | 300 |
| #5 | **`0.5660`** | `sample_regulatory_circular_txt_tokenaware_001` | `sample_regulatory_circular.txt` | Preamble / Document Header | 1 | 0 | 300 |
| #6 | **`0.4873`** | `circular_dor_2024_108_txt_tokenaware_004` | `circular_dor_2024_108.txt` | 5. Record Retention Obligati | 1 | 3 | 231 |
| #7 | **`0.4873`** | `sample_regulatory_circular_txt_tokenaware_004` | `sample_regulatory_circular.txt` | 5. Record Retention Obligati | 1 | 3 | 231 |
| #8 | **`0.4562`** | `cyber_resilience_framework_pdf_tokenaware_001` | `cyber_resilience_framework.pdf` | Preamble / Document Header | 1 | 0 | 300 |
| #9 | **`0.4433`** | `digital_lending_compliance_note_html_tokenaware_001` | `digital_lending_compliance_note.html` | Preamble / Document Header | 1 | 0 | 300 |
| #10 | **`0.4413`** | `circular_dor_2024_108_txt_tokenaware_003` | `circular_dor_2024_108.txt` | 3. Enhanced Due Diligence (E | 1 | 2 | 300 |
| #11 | **`0.4413`** | `sample_regulatory_circular_txt_tokenaware_003` | `sample_regulatory_circular.txt` | 3. Enhanced Due Diligence (E | 1 | 2 | 300 |
| #12 | **`0.4381`** | `cyber_resilience_framework_pdf_tokenaware_002` | `cyber_resilience_framework.pdf` | Preamble / Document Header | 1 | 1 | 206 |
| #13 | **`0.4321`** | `guidelines_cdd_pml_rules_md_tokenaware_002` | `guidelines_cdd_pml_rules.md` | 2. Customer Risk Categorizat | 1 | 1 | 300 |
| #14 | **`0.3569`** | `digital_lending_compliance_note_html_tokenaware_002` | `digital_lending_compliance_note.html` | 3. Code of Conduct for Recov | 1 | 1 | 87 |
| #15 | **`0.3529`** | `guidelines_cdd_pml_rules_md_tokenaware_003` | `guidelines_cdd_pml_rules.md` | 4. Beneficial Ownership Thre | 1 | 2 | 69 |

---

## 4. Metric Justification: Why Cosine Similarity? (Task 4)

### Justification for Choosing Cosine Similarity as the Retrieval Metric

When comparing high-dimensional text embeddings in a RAG pipeline, **Cosine Similarity** is the industry standard and mathematically optimal metric for several fundamental reasons:

#### 1. Direction Encodes Meaning, Not Magnitude
In transformer-based embedding models (such as `all-minilm`), semantic meaning and conceptual orientation are encoded in the **angular direction** of the high-dimensional vector. The magnitude (length $\|\\mathbf{v}\|_2$) of an unnormalized embedding vector is frequently influenced by confounding factors such as:
- Total token count / text chunk length
- Syntactic boilerplate density
- Frequency of rare vocabulary terms
Cosine similarity explicitly isolates vector direction:
$$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2}$$
By dividing by Euclidean norms, cosine similarity evaluates purely the angle $\\theta$ between the query and document vectors, regardless of document length.

#### 2. Invariance to Chunk Length Variation
If an unnormalized metric like raw Dot Product or Euclidean distance ($L_2$) were used:
- A longer chunk with 300 tokens might have a larger $L_2$ norm and artificially higher dot product than a concise 50-token chunk that is perfectly on-topic.
- With Euclidean distance, two identical statements of differing length would be separated by a large distance ($d > 0$) simply due to magnitude scaling.
Cosine similarity guarantees that a concise 1-sentence clause and an exhaustive paragraph on the exact same subject receive near-identical similarity scores.

#### 3. Bounded, Standardized Scale $[-1.0, +1.0]$
- Unlike Euclidean distance ($[0, \\infty)$ where distance scale depends on vector dimensionality) or unnormalized dot product ($(-\\infty, \\infty)$), cosine similarity is bounded strictly between **`-1.0`** (diametrically opposed) and **`+1.0`** (identical direction), with **`0.0`** representing orthogonality.
- This predictable, bounded range allows setting reliable retrieval relevance thresholds (e.g., score $\\ge 0.45$ for high relevance).

#### 4. Equivalence to Dot Product for Unit-Normalized Vectors
When embeddings are $L_2$-normalized prior to indexing ($\\|\\mathbf{u}\\|_2 = \\|\\mathbf{v}\\|_2 = 1$), the denominator becomes $1$, and cosine similarity reduces to simple dot product:
$$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\mathbf{u} \\cdot \\mathbf{v}$$
This enables vector search libraries (ChromaDB, FAISS, Milvus) to execute maximum inner product search (MIPS) using vectorized matrix operations and BLAS acceleration at extreme scale.

---
*Report automatically generated by `src/similarity_search.py` for RegulSense RAG Assistant.*