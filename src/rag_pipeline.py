"""End-to-End Modular RAG Pipeline for RegulSense Banking Compliance Assistant.

This module implements the complete query-to-answer RAG flow:
1. Task 1 - Document the full flow:
   Architectural lifecycle spanning query embedding, dense semantic retrieval,
   grounded context assembly, LLM generation, and source attribution.
2. Task 2 - Represent each stage in code:
   Modular, standalone, individually testable stages:
   - Stage 1: `embed_query_stage` (dense vector projection via all-minilm)
   - Stage 2: `retrieve_chunks_stage` (ChromaDB top-k similarity search)
   - Stage 3: `assemble_context_stage` (token budgeting, provenance formatting, prompt rendering)
   - Stage 4: `generate_answer_stage` (grounded LLM inference via llama3:latest)
   - Stage 5: `attribute_sources_stage` (provenance citations extraction)
3. Task 3 - Run end to end:
   End-to-end execution on live regulatory compliance queries with generated answers
   and cited sources, outputting Markdown and JSON reports.
4. Task 4 - Separate responsibilities:
   Strict decoupling of stage functions allowing independent unit testing, mocking,
   and parameter tuning.
5. Task 5 - Commit structure with flow description:
   Exports full trace artifacts and aligns with architectural documentation.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.templates import (
    PromptTemplate,
    ChatPromptTemplate,
    REGUL_SENSE_SYSTEM_TEMPLATE,
    RAG_COMPLIANCE_USER_TEMPLATE,
    VARIATION_B_SYSTEM_PROMPT,
)
from src.retriever import VectorRetriever
from src.vector_db import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_DIMENSION,
    RetrievedRecord,
    VectorDatabaseManager,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RAGPipeline")

DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3:latest")
DEFAULT_SAMPLE_QUERY = (
    "What is the mandatory timeframe for banks to report Severity 1 cyber security "
    "incidents to CERT-In and RBI?"
)


# ==============================================================================
# Pipeline Data Contracts & Schema
# ==============================================================================

@dataclass
class QueryEmbedding:
    """Encapsulates the dense vector representation of a user query."""
    query_text: str
    vector: List[float]
    dimension: int
    latency_seconds: float


@dataclass
class RetrievedContextChunk:
    """Represents a candidate regulatory chunk retrieved for prompt grounding."""
    chunk_id: str
    source_document: str
    section: str
    page_number: int
    similarity_score: float
    text: str
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_document": self.source_document,
            "section": self.section,
            "page_number": self.page_number,
            "similarity_score": round(self.similarity_score, 4),
            "text": self.text,
            "token_count": self.token_count,
        }


@dataclass
class AssembledContext:
    """Represents context packaged with prompt templates and token budgets."""
    system_prompt: str
    user_prompt: str
    messages: List[Dict[str, str]]
    raw_context_text: str
    included_chunks: List[RetrievedContextChunk]
    total_context_tokens: int
    was_truncated: bool
    latency_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "messages": self.messages,
            "raw_context_text": self.raw_context_text,
            "included_chunks": [c.to_dict() for c in self.included_chunks],
            "total_context_tokens": self.total_context_tokens,
            "was_truncated": self.was_truncated,
            "latency_seconds": round(self.latency_seconds, 4),
        }


@dataclass
class GenerationResult:
    """Encapsulates the output of the LLM generation step."""
    answer_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    latency_seconds: float
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "latency_seconds": round(self.latency_seconds, 4),
            "model": self.model,
        }


@dataclass
class SourceCitation:
    """Attributed source reference detailing provenance and evidentiary basis."""
    chunk_id: str
    source_document: str
    section: str
    similarity_score: float
    excerpt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_document": self.source_document,
            "section": self.section,
            "similarity_score": round(self.similarity_score, 4),
            "excerpt": self.excerpt,
        }


@dataclass
class PipelineStageMetrics:
    """Aggregated latency and token consumption ledger across all stages."""
    embed_latency_seconds: float
    retrieve_latency_seconds: float
    assemble_latency_seconds: float
    generate_latency_seconds: float
    total_latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embed_latency_seconds": round(self.embed_latency_seconds, 4),
            "retrieve_latency_seconds": round(self.retrieve_latency_seconds, 4),
            "assemble_latency_seconds": round(self.assemble_latency_seconds, 4),
            "generate_latency_seconds": round(self.generate_latency_seconds, 4),
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class RAGResponse:
    """Final, comprehensive output produced by the end-to-end RAG pipeline."""
    query: str
    answer: str
    citations: List[SourceCitation]
    metrics: PipelineStageMetrics
    assembled_context: AssembledContext
    raw_generation: GenerationResult
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "metrics": self.metrics.to_dict(),
            "assembled_context": self.assembled_context.to_dict(),
            "raw_generation": self.raw_generation.to_dict(),
        }


# ==============================================================================
# Task 2 & Task 4: Modular, Independently Testable Pipeline Stages
# ==============================================================================

def embed_query_stage(
    query: str,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = DEFAULT_VECTOR_DIMENSION,
) -> QueryEmbedding:
    """Stage 1: Pre-flights and projects a user natural language query into vector space.

    Args:
        query: Compliance question text.
        client: Optional OpenAI-compatible client.
        model: Target embedding model identifier.
        dimension: Expected coordinate length.

    Returns:
        QueryEmbedding containing the validated embedding vector and latency.

    Raises:
        ValueError: If query is empty/whitespace or returned vector dimension mismatches.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        raise ValueError("Query text cannot be empty or whitespace.")

    if client is None:
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        )

    start_time = time.time()
    logger.info("Stage 1 [Embed]: Embedding query: '%s'...", clean_query[:60])
    response = client.embeddings.create(model=model, input=clean_query)
    elapsed = time.time() - start_time

    vector = response.data[0].embedding
    if len(vector) != dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {dimension} coordinates, "
            f"got {len(vector)}."
        )

    logger.info("Stage 1 [Embed]: Generated %d-dim vector in %.3fs", len(vector), elapsed)
    return QueryEmbedding(
        query_text=clean_query,
        vector=vector,
        dimension=len(vector),
        latency_seconds=elapsed,
    )


def retrieve_chunks_stage(
    query_text: str,
    query_vector: Optional[List[float]] = None,
    retriever: Optional[VectorRetriever] = None,
    top_k: int = 3,
    min_score: Optional[float] = None,
    where_filter: Optional[Dict[str, Any]] = None,
) -> Tuple[List[RetrievedContextChunk], float]:
    """Stage 2: Executes semantic nearest neighbor search against ChromaDB collection.

    Args:
        query_text: Natural language query.
        query_vector: Optional pre-computed query vector from Stage 1.
        retriever: Optional VectorRetriever instance.
        top_k: Number of most relevant candidates to retrieve.
        min_score: Optional cosine similarity floor (chunks below this are discarded).
        where_filter: Optional ChromaDB metadata filter.

    Returns:
        Tuple of (List of RetrievedContextChunk objects, retrieval latency in seconds).
    """
    if retriever is None:
        retriever = VectorRetriever()

    start_time = time.time()
    logger.info(
        "Stage 2 [Retrieve]: Searching top-%d chunks (min_score=%s, filter=%s)...",
        top_k,
        min_score,
        where_filter,
    )

    run_result = retriever.retrieve(
        query_text=query_text,
        top_k=top_k,
        query_vector=query_vector,
        where=where_filter,
    )
    elapsed = time.time() - start_time

    chunks: List[RetrievedContextChunk] = []
    for rec in run_result.chunks:
        # Score threshold filtering if specified
        if min_score is not None and rec.similarity_score < min_score:
            logger.debug(
                "Skipping chunk '%s' with score %.4f < threshold %.4f",
                rec.id,
                rec.similarity_score,
                min_score,
            )
            continue

        meta = rec.metadata or {}
        chunk_obj = RetrievedContextChunk(
            chunk_id=rec.id,
            source_document=str(meta.get("source_document") or meta.get("filename") or "Unknown Circular"),
            section=str(meta.get("section") or "General Provisions"),
            page_number=int(meta.get("page_number", 1)),
            similarity_score=float(rec.similarity_score),
            text=rec.document,
            token_count=int(meta.get("token_count", len(rec.document.split()))),
        )
        chunks.append(chunk_obj)

    logger.info(
        "Stage 2 [Retrieve]: Found %d eligible chunks (top score: %.4f) in %.3fs",
        len(chunks),
        chunks[0].similarity_score if chunks else 0.0,
        elapsed,
    )
    return chunks, elapsed


def assemble_context_stage(
    query: str,
    chunks: List[RetrievedContextChunk],
    max_context_tokens: int = 2048,
    assistant_name: str = "RegulSense",
    bank_entity: str = "Commercial Bank",
    system_template: Optional[PromptTemplate] = None,
    user_template: Optional[PromptTemplate] = None,
    encoding_name: str = "cl100k_base",
) -> AssembledContext:
    """Stage 3: Budgets and structures retrieved chunks into citation-ready prompt messages.

    Args:
        query: User compliance inquiry.
        chunks: Ranked candidate regulatory chunks.
        max_context_tokens: Maximum token budget reserved for context.
        assistant_name: Persona name.
        bank_entity: Banking entity name.
        system_template: Optional system PromptTemplate override.
        user_template: Optional user PromptTemplate override.
        encoding_name: Tokenizer encoding name.

    Returns:
        AssembledContext ready for LLM chat completions API.
    """
    start_time = time.time()
    try:
        encoder = tiktoken.get_encoding(encoding_name)
        count_tokens = lambda txt: len(encoder.encode(txt))
    except Exception:
        count_tokens = lambda txt: max(1, len(txt.split()) * 4 // 3)

    # 1. Format chunks into structured context blocks
    formatted_blocks: List[str] = []
    included_chunks: List[RetrievedContextChunk] = []
    current_tokens = 0
    was_truncated = False

    if not chunks:
        raw_context = (
            "[NO DIRECT REGULATORY EVIDENCE FOUND: No retrieved circular chunks met "
            "the relevance threshold. Fall back to standard out-of-scope guidance.]"
        )
    else:
        for idx, chunk in enumerate(chunks, start=1):
            block = (
                f"[CIRCULAR EVIDENCE {idx}]\n"
                f"Document: {chunk.source_document}\n"
                f"Section: {chunk.section} (Page {chunk.page_number})\n"
                f"Chunk ID: {chunk.chunk_id} | Relevance Score: {chunk.similarity_score:.4f}\n"
                f"Content:\n{chunk.text.strip()}\n"
            )
            block_tokens = count_tokens(block)
            if current_tokens + block_tokens > max_context_tokens and included_chunks:
                logger.warning(
                    "Stage 3 [Assemble]: Context budget reached (%d + %d > %d). Truncating remaining chunks.",
                    current_tokens,
                    block_tokens,
                    max_context_tokens,
                )
                was_truncated = True
                break

            formatted_blocks.append(block)
            included_chunks.append(chunk)
            current_tokens += block_tokens

        raw_context = "\n---\n".join(formatted_blocks)

    # 2. Render templates
    if system_template is None:
        sys_prompt = REGUL_SENSE_SYSTEM_TEMPLATE.render(
            assistant_name=assistant_name,
            bank_entity=bank_entity,
        )
    else:
        sys_prompt = system_template.render(
            assistant_name=assistant_name,
            bank_entity=bank_entity,
        )

    if user_template is None:
        usr_prompt = RAG_COMPLIANCE_USER_TEMPLATE.render(
            context=raw_context,
            question=query,
        )
    else:
        usr_prompt = user_template.render(
            context=raw_context,
            question=query,
        )

    messages = [
        {"role": "system", "content": sys_prompt.strip()},
        {"role": "user", "content": usr_prompt.strip()},
    ]

    total_context_tokens = count_tokens(raw_context)
    elapsed = time.time() - start_time

    logger.info(
        "Stage 3 [Assemble]: Formatted %d chunks (%d tokens, truncated=%s) in %.3fs",
        len(included_chunks),
        total_context_tokens,
        was_truncated,
        elapsed,
    )

    return AssembledContext(
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        messages=messages,
        raw_context_text=raw_context,
        included_chunks=included_chunks,
        total_context_tokens=total_context_tokens,
        was_truncated=was_truncated,
        latency_seconds=elapsed,
    )


def generate_answer_stage(
    messages: List[Dict[str, str]],
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> GenerationResult:
    """Stage 4: Invokes the LLM with grounded context to synthesize a factual compliance answer.

    Args:
        messages: Chat completion message array with system and user roles.
        client: Optional OpenAI client.
        model: Target LLM model name.
        temperature: Sampling temperature (0.2 for deterministic compliance).
        max_tokens: Completion token limit.

    Returns:
        GenerationResult containing generated text, token usage, and latency.
    """
    if client is None:
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        )

    start_time = time.time()
    logger.info("Stage 4 [Generate]: Requesting completion from model '%s' (T=%.2f)...", model, temperature)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start_time

    choice = response.choices[0]
    answer_text = choice.message.content.strip()
    finish_reason = getattr(choice, "finish_reason", "stop") or "stop"
    usage = response.usage

    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)

    logger.info(
        "Stage 4 [Generate]: Received answer (%d tokens, finish=%s) in %.3fs",
        completion_tokens,
        finish_reason,
        elapsed,
    )

    return GenerationResult(
        answer_text=answer_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        finish_reason=finish_reason,
        latency_seconds=elapsed,
        model=model,
    )


def attribute_sources_stage(
    chunks: List[RetrievedContextChunk],
    excerpt_length: int = 220,
) -> List[SourceCitation]:
    """Stage 5: Maps utilized context chunks into verifiable audit citations for examiners.

    Args:
        chunks: List of context chunks incorporated into the prompt.
        excerpt_length: Max character length for preview excerpt.

    Returns:
        List of structured SourceCitation objects.
    """
    citations: List[SourceCitation] = []
    for c in chunks:
        # Create single-line clean excerpt
        clean_text = " ".join(c.text.split())
        preview = clean_text[:excerpt_length] + ("..." if len(clean_text) > excerpt_length else "")
        citations.append(
            SourceCitation(
                chunk_id=c.chunk_id,
                source_document=c.source_document,
                section=c.section,
                similarity_score=c.similarity_score,
                excerpt=preview,
            )
        )
    return citations


# ==============================================================================
# Pipeline Orchestrator Class
# ==============================================================================

class RAGPipeline:
    """Orchestrates the query-to-answer pipeline connecting all 5 modular stages."""

    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        openai_client: Optional[OpenAI] = None,
        chat_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_dimension: Optional[int] = None,
        default_top_k: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 500,
        max_context_tokens: int = 2048,
    ):
        """Initializes the RAG pipeline with configured components."""
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.vector_dimension = vector_dimension or DEFAULT_VECTOR_DIMENSION
        self.chat_model = chat_model or os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.default_top_k = default_top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens

        # Initialize or receive client
        if openai_client:
            self.client = openai_client
        else:
            self.client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            )

        # Initialize or receive retriever
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = VectorRetriever(
                openai_client=self.client,
                embedding_model=self.embedding_model,
                vector_dimension=self.vector_dimension,
            )

        logger.info(
            "Initialized RAGPipeline (chat_model='%s', embedding_model='%s', dim=%d, default_k=%d)",
            self.chat_model,
            self.embedding_model,
            self.vector_dimension,
            self.default_top_k,
        )

    def embed(self, query: str) -> QueryEmbedding:
        """Runs Stage 1."""
        return embed_query_stage(
            query=query,
            client=self.client,
            model=self.embedding_model,
            dimension=self.vector_dimension,
        )

    def retrieve(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RetrievedContextChunk], float]:
        """Runs Stage 2."""
        return retrieve_chunks_stage(
            query_text=query,
            query_vector=query_vector,
            retriever=self.retriever,
            top_k=top_k or self.default_top_k,
            min_score=min_score,
            where_filter=where_filter,
        )

    def assemble(
        self,
        query: str,
        chunks: List[RetrievedContextChunk],
        max_context_tokens: Optional[int] = None,
    ) -> AssembledContext:
        """Runs Stage 3."""
        return assemble_context_stage(
            query=query,
            chunks=chunks,
            max_context_tokens=max_context_tokens or self.max_context_tokens,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Runs Stage 4."""
        return generate_answer_stage(
            messages=messages,
            client=self.client,
            model=self.chat_model,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )

    def attribute(self, chunks: List[RetrievedContextChunk]) -> List[SourceCitation]:
        """Runs Stage 5."""
        return attribute_sources_stage(chunks=chunks)

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RAGResponse:
        """Runs the complete query-to-answer pipeline end-to-end.

        Args:
            query: Natural language compliance query.
            top_k: Number of chunks to retrieve (defaults to default_top_k).
            min_score: Minimum cosine similarity threshold.
            where_filter: Metadata filter constraint.
            temperature: Generation temperature override.
            max_tokens: Generation max tokens override.

        Returns:
            RAGResponse object complete with answer, citations, and stage metrics.
        """
        pipeline_start = time.time()
        logger.info("Starting End-to-End RAG Pipeline Execution for query: '%s'", query)

        # Stage 1: Embed Query
        embedding_result = self.embed(query)

        # Stage 2: Retrieve Chunks
        chunks, retrieve_latency = self.retrieve(
            query=query,
            query_vector=embedding_result.vector,
            top_k=top_k,
            min_score=min_score,
            where_filter=where_filter,
        )

        # Stage 3: Assemble Context
        assembled_context = self.assemble(query=query, chunks=chunks)

        # Stage 4: Generate Grounded Answer
        generation_result = self.generate(
            messages=assembled_context.messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Stage 5: Attribute Sources
        citations = self.attribute(assembled_context.included_chunks)

        total_latency = time.time() - pipeline_start

        metrics = PipelineStageMetrics(
            embed_latency_seconds=embedding_result.latency_seconds,
            retrieve_latency_seconds=retrieve_latency,
            assemble_latency_seconds=assembled_context.latency_seconds,
            generate_latency_seconds=generation_result.latency_seconds,
            total_latency_seconds=total_latency,
            prompt_tokens=generation_result.prompt_tokens,
            completion_tokens=generation_result.completion_tokens,
            total_tokens=generation_result.total_tokens,
        )

        logger.info(
            "End-to-End RAG Pipeline Complete in %.3fs (Embed: %.3fs, Ret: %.3fs, Asm: %.3fs, Gen: %.3fs)",
            total_latency,
            embedding_result.latency_seconds,
            retrieve_latency,
            assembled_context.latency_seconds,
            generation_result.latency_seconds,
        )

        return RAGResponse(
            query=query,
            answer=generation_result.answer_text,
            citations=citations,
            metrics=metrics,
            assembled_context=assembled_context,
            raw_generation=generation_result,
        )


# ==============================================================================
# Reporting & Artifact Serialization (Tasks 3 & 5)
# ==============================================================================

def generate_markdown_report(response: RAGResponse) -> str:
    """Generates a comprehensive Markdown documentation report of the pipeline run."""
    lines: List[str] = [
        "# RegulSense: End-to-End RAG Pipeline Execution & Grounding Report",
        "",
        f"- **Execution Timestamp**: `{response.timestamp}`",
        f"- **Chat Generation Model**: `{response.raw_generation.model}`",
        f"- **Total End-to-End Latency**: `{response.metrics.total_latency_seconds:.3f}s`",
        f"- **Total Tokens Consumed**: `{response.metrics.total_tokens}` (`{response.metrics.prompt_tokens}` prompt + `{response.metrics.completion_tokens}` completion)",
        "",
        "---",
        "",
        "## 1. User Inquiry & Synthesized Grounded Response (Task 3)",
        "",
        f"### **Compliance Question**:",
        f"> *\"{response.query}\"*",
        "",
        f"### **RegulSense Model Answer**:",
        f"{response.answer}",
        "",
        "---",
        "",
        "## 2. Attributed Sources & Evidentiary Provenance (Task 3)",
        "",
        "The following regulatory circular chunks were retrieved and injected as context:",
        "",
        "| Rank | Source Document | Section | Chunk ID | Similarity Score | Excerpt Preview |",
        "| :---: | :--- | :--- | :--- | :---: | :--- |",
    ]

    for i, c in enumerate(response.citations, start=1):
        lines.append(
            f"| #{i} | `{c.source_document}` | {c.section} | `{c.chunk_id}` | `{c.similarity_score:.4f}` | \"{c.excerpt}\" |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Modular Stage Execution Breakdown & Latency Ledger (Task 2 & 4)",
        "",
        "| Pipeline Stage | Function / Stage | Latency | Tokens / Output | Technical Responsibility |",
        "| :--- | :--- | :---: | :---: | :--- |",
        f"| **Stage 1: Embed** | `embed_query_stage()` | `{response.metrics.embed_latency_seconds:.4f}s` | 384 coordinates | Project query into dense vector space |",
        f"| **Stage 2: Retrieve** | `retrieve_chunks_stage()` | `{response.metrics.retrieve_latency_seconds:.4f}s` | {len(response.citations)} chunks | ChromaDB cosine similarity search |",
        f"| **Stage 3: Assemble** | `assemble_context_stage()` | `{response.metrics.assemble_latency_seconds:.4f}s` | {response.assembled_context.total_context_tokens} context tokens | Format headers, enforce budget, render prompts |",
        f"| **Stage 4: Generate** | `generate_answer_stage()` | `{response.metrics.generate_latency_seconds:.4f}s` | {response.metrics.completion_tokens} tokens | Grounded LLM inference ($T=0.2$) |",
        f"| **Stage 5: Attribute** | `attribute_sources_stage()` | `< 0.001s` | {len(response.citations)} citations | Provenance audit trail extraction |",
        f"| **Total Pipeline** | `RAGPipeline.run()` | **`{response.metrics.total_latency_seconds:.4f}s`** | **`{response.metrics.total_tokens}` tokens** | Complete end-to-end query resolution |",
        "",
        "---",
        "",
        "## 4. Assembled Context Preview (Prompt Injected into Generator)",
        "",
        "```text",
        response.assembled_context.raw_context_text,
        "```",
        "",
        "---",
        "*Report automatically generated by `src/rag_pipeline.py` for RegulSense Banking Compliance Assistant.*",
    ])
    return "\n".join(lines)


def export_pipeline_artifacts(
    response: RAGResponse,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Saves Markdown and JSON pipeline execution reports."""
    out_path = Path(output_dir or (PROJECT_ROOT / "outputs"))
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "rag_pipeline_execution.json"
    md_file = out_path / "rag_pipeline_execution.md"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(response.to_dict(), f, indent=2, ensure_ascii=False)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(response))

    logger.info("Saved RAG pipeline JSON report to %s", json_file)
    logger.info("Saved RAG pipeline Markdown report to %s", md_file)
    return md_file, json_file


def run_sample_pipeline(
    query: Optional[str] = None,
    top_k: int = 3,
    min_score: Optional[float] = 0.40,
) -> RAGResponse:
    """Executes the pipeline end-to-end on a representative query and saves reports."""
    target_query = query or DEFAULT_SAMPLE_QUERY

    print("=" * 80)
    print("REGULSENSE: END-TO-END RAG PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Sample Query: \"{target_query}\"")
    print(f"Retrieval Parameters: top_k={top_k}, min_score={min_score}")
    print("-" * 80)

    pipeline = RAGPipeline(default_top_k=top_k)
    response = pipeline.run(
        query=target_query,
        top_k=top_k,
        min_score=min_score,
    )

    md_path, json_path = export_pipeline_artifacts(response)

    print("\n[GENERATED GROUNDED ANSWER]:")
    print(response.answer)
    print("\n[ATTRIBUTED SOURCES]:")
    for i, cit in enumerate(response.citations, start=1):
        print(f"  {i}. [{cit.source_document}] Section: {cit.section} (Score: {cit.similarity_score:.4f}, ID: {cit.chunk_id})")
        print(f"     Excerpt: {cit.excerpt}")

    print("\n[STAGE LATENCY LEDGER]:")
    print(f"  Stage 1 (Embed):    {response.metrics.embed_latency_seconds:.4f}s")
    print(f"  Stage 2 (Retrieve): {response.metrics.retrieve_latency_seconds:.4f}s")
    print(f"  Stage 3 (Assemble): {response.metrics.assemble_latency_seconds:.4f}s")
    print(f"  Stage 4 (Generate): {response.metrics.generate_latency_seconds:.4f}s")
    print(f"  Total Pipeline:     {response.metrics.total_latency_seconds:.4f}s")
    print(f"  Token Consumption:  {response.metrics.total_tokens} tokens ({response.metrics.prompt_tokens} prompt + {response.metrics.completion_tokens} completion)")
    print("-" * 80)
    print(f"Exported Markdown: {md_path}")
    print(f"Exported JSON:     {json_path}")
    print("=" * 80)

    return response


if __name__ == "__main__":
    cli_query = sys.argv[1] if len(sys.argv) > 1 else None
    run_sample_pipeline(query=cli_query)
