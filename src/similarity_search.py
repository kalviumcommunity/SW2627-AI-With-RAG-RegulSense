"""Similarity Search and Ranking Engine for RegulSense Banking Compliance Assistant.

This module implements:
1. Computing similarity metrics (Cosine Similarity and Euclidean Distance) between embedding vectors.
2. Embedding user compliance queries and comparing them against prepared corpus chunk vectors.
3. Ranking corpus chunks by semantic similarity, identifying most and least related results.
4. Providing detailed mathematical and architectural justification for choosing Cosine Similarity.
5. Exporting ranked results with source text, metadata, and scores to JSON and Markdown artifacts.
"""

from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("SimilaritySearch")

DEFAULT_QUERY = (
    "What are the customer due diligence and KYC verification requirements "
    "for onboarding new bank accounts?"
)

METRIC_JUSTIFICATION = r"""### Justification for Choosing Cosine Similarity as the Retrieval Metric

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
"""


@dataclass
class ScoredChunk:
    """Represents a corpus chunk scored and ranked against a query."""
    rank: int
    score: float
    euclidean_distance: float
    chunk_id: str
    source_document: str
    section: str
    page_number: int
    chunk_index: int
    token_count: int
    source_text: str
    metadata: Dict[str, Any]


@dataclass
class QueryRankingResult:
    """Encapsulates the complete ranking result for a user query."""
    query_text: str
    model: str
    vector_dimension: int
    total_corpus_chunks: int
    ranked_chunks: List[ScoredChunk]
    top_chunks: List[ScoredChunk]
    bottom_chunks: List[ScoredChunk]
    metric_name: str
    justification: str


def compute_cosine_similarity(vec1: List[float] | np.ndarray, vec2: List[float] | np.ndarray) -> float:
    """Computes cosine similarity between two numeric vectors.
    
    Formula:
        similarity = (u . v) / (||u||_2 * ||v||_2)
    """
    a = np.asarray(vec1, dtype=np.float64)
    b = np.asarray(vec2, dtype=np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(sim, -1.0, 1.0))


def compute_euclidean_distance(vec1: List[float] | np.ndarray, vec2: List[float] | np.ndarray) -> float:
    """Computes Euclidean (L2) distance between two numeric vectors."""
    a = np.asarray(vec1, dtype=np.float64)
    b = np.asarray(vec2, dtype=np.float64)
    return float(np.linalg.norm(a - b))


class SimilaritySearchEngine:
    """Engine for ranking corpus chunks against compliance queries."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ):
        """Initializes the engine reading configuration from environment."""
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("EMBEDDING_MODEL")

        if not self.model:
            raise ValueError("EMBEDDING_MODEL must be configured in environment or passed explicitly.")

        if client:
            self.client = client
        else:
            client_kwargs: Dict[str, Any] = {}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            self.client = OpenAI(**client_kwargs)

        logger.info("Initialized SimilaritySearchEngine (model='%s', endpoint='%s')", self.model, self.base_url)

    def load_corpus_chunks(self, json_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Loads embedded corpus chunks from outputs/embedded_corpus_chunks.json."""
        default_path = PROJECT_ROOT / "outputs" / "embedded_corpus_chunks.json"
        path = json_path or default_path

        if not path.exists():
            raise FileNotFoundError(
                f"Embedded corpus not found at {path}. Run src/corpus_embedder.py first."
            )

        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = data.get("embedded_chunks", [])
        logger.info("Loaded %d embedded chunks from %s", len(chunks), path)
        return chunks

    def embed_query(self, query_text: str) -> List[float]:
        """Generates embedding vector for the search query."""
        if not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        logger.info("Generating embedding for query: '%s'...", query_text[:60])
        response = self.client.embeddings.create(
            model=self.model,
            input=query_text,
        )
        vector = response.data[0].embedding
        logger.info("Query vector generated (dimension=%d)", len(vector))
        return vector

    def rank_chunks_for_query(
        self,
        query_text: str,
        corpus_chunks: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 3,
        bottom_k: int = 3,
        corpus_json_path: Optional[Path] = None,
    ) -> QueryRankingResult:
        """Compares query against corpus chunks, ranks by cosine similarity, and isolates top and bottom matches."""
        chunks = corpus_chunks or self.load_corpus_chunks(corpus_json_path)
        if not chunks:
            raise ValueError("Corpus chunks collection is empty.")

        query_vector = self.embed_query(query_text)
        scored_items: List[Tuple[float, float, Dict[str, Any]]] = []

        for c in chunks:
            chunk_vector = c.get("embedding", [])
            cos_sim = compute_cosine_similarity(query_vector, chunk_vector)
            l2_dist = compute_euclidean_distance(query_vector, chunk_vector)
            scored_items.append((cos_sim, l2_dist, c))

        # Sort descending by cosine similarity score
        scored_items.sort(key=lambda x: x[0], reverse=True)

        ranked_chunks: List[ScoredChunk] = []
        for rank, (score, l2, c) in enumerate(scored_items, start=1):
            meta = c.get("metadata", {})
            ranked_chunks.append(
                ScoredChunk(
                    rank=rank,
                    score=round(score, 4),
                    euclidean_distance=round(l2, 4),
                    chunk_id=c.get("chunk_id", f"chunk_{rank}"),
                    source_document=meta.get("filename", "unknown_doc"),
                    section=meta.get("section", "General"),
                    page_number=meta.get("page_number", 1),
                    chunk_index=meta.get("chunk_index", 0),
                    token_count=meta.get("token_count", 0),
                    source_text=c.get("source_text", ""),
                    metadata=meta,
                )
            )

        top_matches = ranked_chunks[:top_k]
        bottom_matches = ranked_chunks[-bottom_k:]

        return QueryRankingResult(
            query_text=query_text,
            model=self.model,
            vector_dimension=len(query_vector),
            total_corpus_chunks=len(ranked_chunks),
            ranked_chunks=ranked_chunks,
            top_chunks=top_matches,
            bottom_chunks=bottom_matches,
            metric_name="Cosine Similarity",
            justification=METRIC_JUSTIFICATION.strip(),
        )

    def export_ranking_artifacts(
        self,
        ranking_result: QueryRankingResult,
        output_markdown_path: Optional[Path] = None,
        output_json_path: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Saves similarity ranking demonstration report and structured JSON artifact to outputs/."""
        md_path = output_markdown_path or PROJECT_ROOT / "outputs" / "similarity_ranking_results.md"
        json_path = output_json_path or PROJECT_ROOT / "outputs" / "similarity_ranking_results.json"

        # 1. JSON Export
        json_data = {
            "query": {
                "text": ranking_result.query_text,
                "model": ranking_result.model,
                "vector_dimension": ranking_result.vector_dimension,
                "total_chunks_scored": ranking_result.total_corpus_chunks,
                "metric": ranking_result.metric_name,
            },
            "top_similar_chunks": [asdict(c) for c in ranking_result.top_chunks],
            "least_similar_chunks": [asdict(c) for c in ranking_result.bottom_chunks],
            "full_rankings": [asdict(c) for c in ranking_result.ranked_chunks],
            "metric_justification": ranking_result.justification,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
        logger.info("Saved similarity ranking JSON to %s", json_path)

        # 2. Markdown Report
        md_lines = [
            "# RegulSense: Embedding Similarity Ranking & Query Retrieval Demonstration",
            "",
            f"- **Search Query**: *\"{ranking_result.query_text}\"*",
            f"- **Target Embedding Model**: `{ranking_result.model}`",
            f"- **Embedding Vector Dimension**: `{ranking_result.vector_dimension}`",
            f"- **Total Corpus Chunks Evaluated**: `{ranking_result.total_corpus_chunks}`",
            f"- **Chosen Similarity Metric**: **`{ranking_result.metric_name}`**",
            "",
            "---",
            "",
            "## 1. Top Most Similar Chunks (Highest Relevance)",
            "",
            "These chunks exhibit the highest cosine similarity with the query's semantic intent:",
            "",
            "| Rank | Cosine Similarity | Euclidean Dist | Chunk ID | Source Document | Section | Page | Text Preview |",
            "| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :--- |",
        ]

        for c in ranking_result.top_chunks:
            snippet = c.source_text.replace("\n", " ")[:90] + "..."
            md_lines.append(
                f"| **#{c.rank}** | 🟢 **`{c.score:.4f}`** | `{c.euclidean_distance:.4f}` | `{c.chunk_id}` | `{c.source_document}` | {c.section[:25]} | {c.page_number} | \"{snippet}\" |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## 2. Least Similar Chunks (Unrelated / Different Domain)",
            "",
            "These chunks show negligible or negative semantic correlation with the customer due diligence query:",
            "",
            "| Rank | Cosine Similarity | Euclidean Dist | Chunk ID | Source Document | Section | Page | Text Preview |",
            "| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :--- |",
        ])

        for c in ranking_result.bottom_chunks:
            snippet = c.source_text.replace("\n", " ")[:90] + "..."
            md_lines.append(
                f"| **#{c.rank}** | 🔴 **`{c.score:.4f}`** | `{c.euclidean_distance:.4f}` | `{c.chunk_id}` | `{c.source_document}` | {c.section[:25]} | {c.page_number} | \"{snippet}\" |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## 3. Full Corpus Ranking Ledger",
            "",
            "Complete ranking of all evaluated chunks in descending order of semantic relevance:",
            "",
            "| Rank | Score | Chunk ID | Document | Section | Page | Chk Idx | Tokens |",
            "| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: |",
        ])

        for c in ranking_result.ranked_chunks:
            md_lines.append(
                f"| #{c.rank} | **`{c.score:.4f}`** | `{c.chunk_id}` | `{c.source_document}` | {c.section[:28]} | {c.page_number} | {c.chunk_index} | {c.token_count} |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## 4. Metric Justification: Why Cosine Similarity? (Task 4)",
            "",
            ranking_result.justification,
            "",
            "---",
            "*Report automatically generated by `src/similarity_search.py` for RegulSense RAG Assistant.*",
        ])

        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("Saved similarity ranking markdown report to %s", md_path)

        return md_path, json_path


def run_similarity_ranking_demo(
    query_text: Optional[str] = None,
) -> QueryRankingResult:
    """Executes similarity ranking demonstration and prints console verification."""
    search_query = query_text or DEFAULT_QUERY
    engine = SimilaritySearchEngine()
    ranking = engine.rank_chunks_for_query(search_query)
    md_path, json_path = engine.export_ranking_artifacts(ranking)

    # Console Output for User (Task 3 & 4)
    print("\n" + "=" * 80)
    print("REGULSENSE: EMBEDDING SIMILARITY RANKING DEMONSTRATION")
    print("=" * 80)
    print(f"Query: \"{ranking.query_text}\"")
    print(f"Model: {ranking.model} | Vector Dimension: {ranking.vector_dimension}")
    print(f"Metric: {ranking.metric_name} | Corpus Chunks Evaluated: {ranking.total_corpus_chunks}")
    print("-" * 80)
    print("TOP 3 MOST SIMILAR CHUNKS (Highest Semantic Alignment):")
    for c in ranking.top_chunks:
        print(f"  Rank #{c.rank} | Score: {c.score:.4f} | [{c.source_document}] Section: '{c.section[:35]}'")
        print(f"    Text: \"{c.source_text[:90].replace(chr(10), ' ')}...\"")
    print("-" * 80)
    print("BOTTOM 3 LEAST SIMILAR CHUNKS (Lowest Semantic Alignment):")
    for c in ranking.bottom_chunks:
        print(f"  Rank #{c.rank} | Score: {c.score:.4f} | [{c.source_document}] Section: '{c.section[:35]}'")
        print(f"    Text: \"{c.source_text[:90].replace(chr(10), ' ')}...\"")
    print("-" * 80)
    print("TASK 4 METRIC JUSTIFICATION SUMMARY:")
    print("  * Cosine similarity measures vector orientation (angle) rather than magnitude.")
    print("  * Eliminates text-length bias: longer chunks with larger norms don't dominate.")
    print("  * Standard bounded range [-1.0, 1.0] provides intuitive relevance thresholds.")
    print("-" * 80)
    print(f"Output Markdown Report: {md_path}")
    print(f"Output JSON Artifact:   {json_path}")
    print("=" * 80 + "\n")

    return ranking


if __name__ == "__main__":
    run_similarity_ranking_demo()
