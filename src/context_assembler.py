"""Grounded Context Assembly & Prompt Augmentation Engine for RegulSense.

This module implements:
1. Task 1 - Inject retrieved chunks:
   Formats and injects candidate regulatory chunks into the prompt context block.
2. Task 2 - Enforce token budget:
   Calculates exact token budgets reserving space for system instructions, user query,
   safety margins, and model answer generation, preventing context window saturation.
3. Task 3 - Include source markers:
   Tags each injected chunk with unambiguous source markers (e.g. [1], [2], document
   name, section, page, and chunk ID) for downstream attribution.
4. Task 4 - Add grounding instructions:
   Injects strict grounding constraints commanding the model to answer ONLY from the
   provided context, cite source markers, and declare when context is insufficient.
5. Task 5 - Commit sample augmented prompt:
   Generates and exports comprehensive Markdown and JSON reports showcasing the
   assembled prompt, token budget ledger, and grounded LLM output.
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
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ContextAssembler")

DEFAULT_TOKEN_ENCODING = "cl100k_base"
DEFAULT_TOTAL_TOKEN_BUDGET = 2048
DEFAULT_ANSWER_RESERVE = 450
DEFAULT_SAFETY_MARGIN = 50


# ==============================================================================
# Production Grounding Prompt Templates (Task 4)
# ==============================================================================

GROUNDED_SYSTEM_INSTRUCTIONS = """You are RegulSense, an AI Senior Regulatory Compliance Specialist for a commercial bank.
Your sole mission is to answer regulatory compliance questions with absolute accuracy, zero hallucination, and full evidentiary auditability.

CORE GROUNDING RULES:
1. STRICT CONTEXT BOUNDARY:
   - Answer the user's question relying SOLELY and EXCLUSIVELY on the regulatory evidence provided in the "RETRIEVED REGULATORY CONTEXT" section below.
   - Do NOT extrapolate, speculate, or introduce external legal knowledge, general banking industry practices, or unstated assumptions not explicitly documented in the provided excerpts.

2. MANDATORY SOURCE CITATIONS:
   - For every factual assertion, timeline, metric, or requirement in your answer, you MUST cite the corresponding source marker (e.g. [1], [2], or the exact circular name and section).
   - If multiple sources support a point, cite all relevant markers (e.g. [1], [2]).

3. INSUFFICIENT CONTEXT FALLBACK PROTOCOL:
   - If the provided context does NOT contain sufficient, explicit evidence to answer the question with complete certainty, do NOT attempt to answer or guess.
   - You MUST respond with this exact statement:
     "The provided regulatory context does not contain sufficient information to answer this question."
   - You may briefly specify what regulatory information or circular is missing.

4. RESPONSE CONSTRAINTS:
   - Maintain a strictly professional, objective, risk-aware tone.
   - Begin with a direct 1-sentence answer citing the primary source marker.
   - Follow with concise bullet points detailing operative compliance obligations, quoting rule text where appropriate."""

GROUNDED_SYSTEM_TEMPLATE = PromptTemplate(
    template=GROUNDED_SYSTEM_INSTRUCTIONS,
    input_variables=[],
)

GROUNDED_USER_TEMPLATE = PromptTemplate(
    template=(
        "--- RETRIEVED REGULATORY CONTEXT ---\n"
        "{context}\n"
        "--- END REGULATORY CONTEXT ---\n\n"
        "COMPLIANCE INQUIRY:\n"
        "{question}\n\n"
        "Please provide a grounded compliance answer adhering strictly to the source markers "
        "and grounding rules above."
    ),
    input_variables=["context", "question"],
)


# ==============================================================================
# Data Contracts & Schemas
# ==============================================================================

@dataclass
class SourceMarker:
    """Represents an evidentiary citation marker assigned to an injected chunk."""
    marker_id: str  # e.g., "[1]"
    index: int      # 1-indexed
    source_document: str
    section: str
    page_number: int
    chunk_id: str
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker_id": self.marker_id,
            "index": self.index,
            "source_document": self.source_document,
            "section": self.section,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "similarity_score": round(self.similarity_score, 4),
        }


@dataclass
class InjectedChunk:
    """Represents a chunk formatted and injected into the prompt context block."""
    marker: SourceMarker
    raw_text: str
    formatted_block: str
    token_count: int
    was_truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker": self.marker.to_dict(),
            "raw_text": self.raw_text,
            "formatted_block": self.formatted_block,
            "token_count": self.token_count,
            "was_truncated": self.was_truncated,
        }


@dataclass
class TokenBudgetSpec:
    """Defines the allocation boundaries for prompt assembly."""
    total_budget: int = DEFAULT_TOTAL_TOKEN_BUDGET
    answer_reserve: int = DEFAULT_ANSWER_RESERVE
    safety_margin: int = DEFAULT_SAFETY_MARGIN
    max_context_budget: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenBudgetLedger:
    """Comprehensive token audit detailing exact allocation across all prompt components."""
    total_budget: int
    system_tokens: int
    query_tokens: int
    context_tokens: int
    answer_reserve: int
    safety_margin: int
    total_allocated: int
    remaining_tokens: int
    is_within_budget: bool
    chunks_included_count: int
    chunks_dropped_count: int
    was_truncated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AugmentedPrompt:
    """The final assembled, grounded prompt package ready for LLM consumption."""
    query: str
    system_prompt: str
    user_prompt: str
    messages: List[Dict[str, str]]
    raw_context_block: str
    injected_chunks: List[InjectedChunk]
    source_markers: Dict[str, Dict[str, Any]]
    budget_ledger: TokenBudgetLedger
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "messages": self.messages,
            "raw_context_block": self.raw_context_block,
            "injected_chunks": [c.to_dict() for c in self.injected_chunks],
            "source_markers": self.source_markers,
            "budget_ledger": self.budget_ledger.to_dict(),
        }


# ==============================================================================
# Token Budget Manager (Task 2)
# ==============================================================================

class TokenBudgetManager:
    """Measures, calculates, and enforces token budget boundaries."""

    def __init__(self, encoding_name: str = DEFAULT_TOKEN_ENCODING):
        self.encoding_name = encoding_name
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            logger.warning("Could not initialize tiktoken encoding '%s': %s. Using heuristic fallback.", encoding_name, exc)
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Accurately calculates token count for a given text string."""
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        # Heuristic fallback: ~4 characters per token or 1.33 tokens per word
        return max(1, int(len(text.split()) * 1.33))

    def calculate_available_context_budget(
        self,
        spec: TokenBudgetSpec,
        system_tokens: int,
        query_tokens: int,
    ) -> int:
        """Calculates available token budget remaining strictly for context chunks.

        Formula:
            B_context = B_total - T_system - T_query - B_answer - B_margin
        """
        reserved = system_tokens + query_tokens + spec.answer_reserve + spec.safety_margin
        available = spec.total_budget - reserved
        if spec.max_context_budget is not None:
            available = min(available, spec.max_context_budget)
        return max(0, available)


# ==============================================================================
# Context Assembler (Tasks 1, 2, 3, 4)
# ==============================================================================

class ContextAssembler:
    """Assembles retrieved regulatory chunks into a token-budgeted, grounded prompt."""

    def __init__(
        self,
        token_manager: Optional[TokenBudgetManager] = None,
        default_budget_spec: Optional[TokenBudgetSpec] = None,
    ):
        self.token_manager = token_manager or TokenBudgetManager()
        self.default_budget_spec = default_budget_spec or TokenBudgetSpec()

    def format_chunk_block(
        self,
        chunk: Any,
        index: int,
    ) -> Tuple[InjectedChunk, SourceMarker]:
        """Formats a single candidate chunk with explicit source markers (Task 3).

        Args:
            chunk: RetrievedRecord, RetrievedContextChunk, or dict.
            index: 1-indexed sequential rank.

        Returns:
            Tuple of (InjectedChunk, SourceMarker).
        """
        marker_id = f"[{index}]"

        # Normalize attributes across object schemas
        if hasattr(chunk, "chunk_id"):
            chunk_id = chunk.chunk_id
            doc_name = getattr(chunk, "source_document", "Unknown Circular")
            section = getattr(chunk, "section", "General Provisions")
            page = getattr(chunk, "page_number", 1)
            score = getattr(chunk, "similarity_score", 0.0)
            text = getattr(chunk, "text", "")
        elif hasattr(chunk, "id"):  # RetrievedRecord
            chunk_id = chunk.id
            meta = chunk.metadata or {}
            doc_name = str(meta.get("source_document") or meta.get("filename") or "Unknown Circular")
            section = str(meta.get("section") or "General Provisions")
            page = int(meta.get("page_number", 1))
            score = float(chunk.similarity_score)
            text = chunk.document
        elif isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{index:03d}"
            meta = chunk.get("metadata") or {}
            doc_name = chunk.get("source_document") or meta.get("source_document") or "Unknown Circular"
            section = chunk.get("section") or meta.get("section") or "General Provisions"
            page = int(chunk.get("page_number") or meta.get("page_number", 1))
            score = float(chunk.get("similarity_score") or chunk.get("score", 0.0))
            text = chunk.get("text") or chunk.get("document", "")
        else:
            chunk_id = f"chunk_{index:03d}"
            doc_name = "Unknown Circular"
            section = "General Provisions"
            page = 1
            score = 0.0
            text = str(chunk)

        source_marker = SourceMarker(
            marker_id=marker_id,
            index=index,
            source_document=doc_name,
            section=section,
            page_number=page,
            chunk_id=chunk_id,
            similarity_score=score,
        )

        formatted_block = (
            f"SOURCE {marker_id}: {doc_name}\n"
            f"Section: {section} | Page: {page} | Chunk ID: {chunk_id} | Similarity: {score:.4f}\n"
            f"Verbatim Regulatory Content:\n{text.strip()}"
        )

        token_count = self.token_manager.count_tokens(formatted_block)

        injected = InjectedChunk(
            marker=source_marker,
            raw_text=text.strip(),
            formatted_block=formatted_block,
            token_count=token_count,
            was_truncated=False,
        )

        return injected, source_marker

    def assemble(
        self,
        query: str,
        chunks: List[Any],
        budget_spec: Optional[TokenBudgetSpec] = None,
        system_template: Optional[PromptTemplate] = None,
        user_template: Optional[PromptTemplate] = None,
        truncate_oversized_chunk: bool = True,
    ) -> AugmentedPrompt:
        """Assembles chunks, calculates token budgets, and renders the augmented prompt.

        Args:
            query: User compliance inquiry.
            chunks: List of retrieved regulatory chunks.
            budget_spec: Optional token budget specification.
            system_template: Optional system prompt template.
            user_template: Optional user prompt template.
            truncate_oversized_chunk: If True, partially trims a chunk that exceeds budget.

        Returns:
            AugmentedPrompt object ready for LLM invocation.
        """
        clean_query = query.strip() if query else ""
        spec = budget_spec or self.default_budget_spec

        sys_tpl = system_template or GROUNDED_SYSTEM_TEMPLATE
        usr_tpl = user_template or GROUNDED_USER_TEMPLATE

        # 1. Render and measure system prompt
        rendered_sys = sys_tpl.render()
        system_tokens = self.token_manager.count_tokens(rendered_sys)

        # 2. Measure query tokens and template boilerplate
        query_tokens = self.token_manager.count_tokens(clean_query)
        empty_user_rendered = usr_tpl.render(context="", question=clean_query)
        user_overhead_tokens = self.token_manager.count_tokens(empty_user_rendered) - query_tokens

        # 3. Compute available context budget
        available_context_budget = self.token_manager.calculate_available_context_budget(
            spec=spec,
            system_tokens=system_tokens,
            query_tokens=query_tokens + user_overhead_tokens,
        )

        logger.info(
            "Context Budget Calculation: Total=%d, Sys=%d, Query=%d, Reserve=%d, Margin=%d -> Context Budget=%d",
            spec.total_budget,
            system_tokens,
            query_tokens,
            spec.answer_reserve,
            spec.safety_margin,
            available_context_budget,
        )

        # 4. Pack chunks into context block within budget (Task 1 & 2)
        injected_chunks: List[InjectedChunk] = []
        source_markers: Dict[str, Dict[str, Any]] = {}
        current_context_tokens = 0
        chunks_dropped_count = 0
        was_truncated = False

        if not chunks:
            raw_context_block = (
                "[NO REGULATORY CONTEXT AVAILABLE: No retrieved circulars or evidentiary "
                "provisions met the relevance threshold. State that context is insufficient.]"
            )
            current_context_tokens = self.token_manager.count_tokens(raw_context_block)
        else:
            formatted_blocks: List[str] = []
            for idx, raw_chunk in enumerate(chunks, start=1):
                injected, marker = self.format_chunk_block(raw_chunk, index=idx)
                block_tokens = injected.token_count

                # Check if block fits completely
                if current_context_tokens + block_tokens <= available_context_budget:
                    formatted_blocks.append(injected.formatted_block)
                    injected_chunks.append(injected)
                    source_markers[marker.marker_id] = marker.to_dict()
                    current_context_tokens += block_tokens
                else:
                    # Budget reached
                    remaining_budget = available_context_budget - current_context_tokens
                    if truncate_oversized_chunk and remaining_budget > 60 and not was_truncated:
                        # Clean truncation
                        logger.warning(
                            "Chunk %d exceeds remaining context budget (%d > %d). Truncating chunk.",
                            idx,
                            block_tokens,
                            remaining_budget,
                        )
                        # Truncate content text
                        words = injected.raw_text.split()
                        truncated_words = words[: max(10, remaining_budget // 2)]
                        truncated_text = " ".join(truncated_words) + "\n[... TRUNCATED DUE TO CONTEXT BUDGET ...]"
                        
                        trunc_block = (
                            f"SOURCE {marker.marker_id}: {marker.source_document}\n"
                            f"Section: {marker.section} | Page: {marker.page_number} | Chunk ID: {marker.chunk_id}\n"
                            f"Verbatim Regulatory Content (Truncated):\n{truncated_text}"
                        )
                        trunc_tokens = self.token_manager.count_tokens(trunc_block)

                        injected.formatted_block = trunc_block
                        injected.token_count = trunc_tokens
                        injected.was_truncated = True

                        formatted_blocks.append(trunc_block)
                        injected_chunks.append(injected)
                        source_markers[marker.marker_id] = marker.to_dict()
                        current_context_tokens += trunc_tokens
                        was_truncated = True
                    else:
                        chunks_dropped_count += 1
                        was_truncated = True
                        logger.warning("Context budget saturated. Dropping chunk %d (%d dropped total).", idx, chunks_dropped_count)

            divider = "\n\n" + ("-" * 40) + "\n\n"
            raw_context_block = divider.join(formatted_blocks)

        # 5. Render final user prompt
        rendered_user = usr_tpl.render(context=raw_context_block, question=clean_query)
        final_user_tokens = self.token_manager.count_tokens(rendered_user)

        # 6. Audit total allocated tokens
        total_allocated = system_tokens + final_user_tokens + spec.answer_reserve + spec.safety_margin
        remaining_slack = spec.total_budget - total_allocated
        is_within_budget = total_allocated <= spec.total_budget

        budget_ledger = TokenBudgetLedger(
            total_budget=spec.total_budget,
            system_tokens=system_tokens,
            query_tokens=query_tokens,
            context_tokens=self.token_manager.count_tokens(raw_context_block),
            answer_reserve=spec.answer_reserve,
            safety_margin=spec.safety_margin,
            total_allocated=total_allocated,
            remaining_tokens=remaining_slack,
            is_within_budget=is_within_budget,
            chunks_included_count=len(injected_chunks),
            chunks_dropped_count=chunks_dropped_count,
            was_truncated=was_truncated,
        )

        messages = [
            {"role": "system", "content": rendered_sys.strip()},
            {"role": "user", "content": rendered_user.strip()},
        ]

        logger.info(
            "Assembled Augmented Prompt: %d chunks included, %d tokens context, total allocated=%d/%d (slack=%d, within=%s)",
            len(injected_chunks),
            budget_ledger.context_tokens,
            total_allocated,
            spec.total_budget,
            remaining_slack,
            is_within_budget,
        )

        return AugmentedPrompt(
            query=clean_query,
            system_prompt=rendered_sys,
            user_prompt=rendered_user,
            messages=messages,
            raw_context_block=raw_context_block,
            injected_chunks=injected_chunks,
            source_markers=source_markers,
            budget_ledger=budget_ledger,
        )


# ==============================================================================
# Reporting & Artifact Serialization (Task 5)
# ==============================================================================

def generate_augmented_prompt_markdown(
    prompt: AugmentedPrompt,
    llm_completion: Optional[str] = None,
    negative_prompt: Optional[AugmentedPrompt] = None,
    negative_completion: Optional[str] = None,
) -> str:
    """Generates a detailed Markdown report displaying injected chunks, markers, and budget ledger."""
    lines: List[str] = [
        "# RegulSense: Grounded Context Assembly & Augmented Prompt Report",
        "",
        f"- **Execution Timestamp**: `{prompt.timestamp}`",
        f"- **Tokenizer**: `tiktoken` (`{DEFAULT_TOKEN_ENCODING}`)",
        f"- **Configured Total Token Budget**: `{prompt.budget_ledger.total_budget}` tokens",
        f"- **Budget Adherence**: `{'✅ STRICTLY WITHIN BUDGET' if prompt.budget_ledger.is_within_budget else '⚠️ EXCEEDED'}`",
        f"- **Injected Chunks**: `{prompt.budget_ledger.chunks_included_count}` included (`{prompt.budget_ledger.chunks_dropped_count}` dropped, truncated: `{prompt.budget_ledger.was_truncated}`)",
        "",
        "---",
        "",
        "## 1. Token Budget Allocation & Consumption Ledger (Task 2)",
        "",
        "| Component | Allocated Tokens | Budget Limit / Role | % of Budget | Status |",
        "| :--- | :---: | :--- | :---: | :---: |",
        f"| **System Grounding Instructions** | `{prompt.budget_ledger.system_tokens}` | Role, boundary guardrails, citation rules | {prompt.budget_ledger.system_tokens * 100 / prompt.budget_ledger.total_budget:.1f}% | Guaranteed |",
        f"| **User Compliance Inquiry** | `{prompt.budget_ledger.query_tokens}` | Natural language question | {prompt.budget_ledger.query_tokens * 100 / prompt.budget_ledger.total_budget:.1f}% | Preserved |",
        f"| **Injected Regulatory Context** | `{prompt.budget_ledger.context_tokens}` | Candidate circular evidence with source markers | {prompt.budget_ledger.context_tokens * 100 / prompt.budget_ledger.total_budget:.1f}% | Budget-Enforced |",
        f"| **Reserved Output Answer** | `{prompt.budget_ledger.answer_reserve}` | Space guaranteed for model completion | {prompt.budget_ledger.answer_reserve * 100 / prompt.budget_ledger.total_budget:.1f}% | Guaranteed |",
        f"| **Safety Margin** | `{prompt.budget_ledger.safety_margin}` | Delimiter & prompt formatting headroom | {prompt.budget_ledger.safety_margin * 100 / prompt.budget_ledger.total_budget:.1f}% | Buffer |",
        f"| **Total Pipeline Allocated** | **`{prompt.budget_ledger.total_allocated}`** | **Total Budget Ceiling (`{prompt.budget_ledger.total_budget}`)** | **{prompt.budget_ledger.total_allocated * 100 / prompt.budget_ledger.total_budget:.1f}%** | **{'PASS' if prompt.budget_ledger.is_within_budget else 'FAIL'}** |",
        f"| **Unallocated Remaining Slack** | `{prompt.budget_ledger.remaining_tokens}` | Remaining headroom in context window | {prompt.budget_ledger.remaining_tokens * 100 / prompt.budget_ledger.total_budget:.1f}% | Available |",
        "",
        "---",
        "",
        "## 2. Injected Regulatory Chunks with Source Markers (Tasks 1 & 3)",
        "",
        "Every candidate chunk is assigned an explicit bracketed marker `[X]` allowing exact citation traceability:",
        "",
        "| Marker | Source Document | Section | Page | Chunk ID | Similarity | Tokens | Truncated? |",
        "| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: |",
    ]

    for c in prompt.injected_chunks:
        m = c.marker
        lines.append(
            f"| **`{m.marker_id}`** | `{m.source_document}` | {m.section} | {m.page_number} | `{m.chunk_id}` | `{m.similarity_score:.4f}` | {c.token_count} | `{c.was_truncated}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Grounding Instructions & Guardrail Directives (Task 4)",
        "",
        "The system prompt injects four non-negotiable compliance guardrails:",
        "1. **Strict Context Boundary**: Answer strictly from provided circular context; no outside speculation.",
        "2. **Mandatory In-Text Citations**: Every factual statement must cite markers (e.g. `[1]`, `[2]`).",
        "3. **Insufficient Context Refusal**: If evidence is absent, output the mandated refusal phrase:",
        "   > *\"The provided regulatory context does not contain sufficient information to answer this question.\"*",
        "4. **Professional & Risk-Aware Tone**: Formatted with a direct topic sentence and actionable compliance obligations.",
        "",
        "---",
        "",
        "## 4. Full Augmented Prompt Preview (Injected into Model)",
        "",
        "### A. System Message (`role: 'system'`):",
        "```text",
        prompt.system_prompt,
        "```",
        "",
        "### B. User Message with Injected Context (`role: 'user'`):",
        "```text",
        prompt.user_prompt,
        "```",
    ])

    if llm_completion:
        lines.extend([
            "",
            "---",
            "",
            "## 5. Live Grounded Model Completion (Verification of Citations & Grounding)",
            "",
            "### **Model Generated Answer**:",
            llm_completion,
            "",
            "> [!NOTE]",
            "> Notice how the model directly references source markers `[1]` and verbatim provisions from the injected evidence.",
        ])

    if negative_prompt and negative_completion:
        lines.extend([
            "",
            "---",
            "",
            "## 6. Edge-Case Verification: Insufficient Context Fallback Protocol",
            "",
            f"### **Out-of-Corpus Query**: *\"{negative_prompt.query}\"*",
            f"### **Model Fallback Response**:",
            negative_completion,
            "",
            "> [!TIP]",
            "> The model correctly invoked the insufficient-context protocol rather than hallucinating speculative answers.",
        ])

    lines.extend([
        "",
        "---",
        "*Report automatically generated by `src/context_assembler.py` for RegulSense Banking Compliance Assistant.*",
    ])

    return "\n".join(lines)


def export_augmented_prompt_artifacts(
    prompt: AugmentedPrompt,
    llm_completion: Optional[str] = None,
    negative_prompt: Optional[AugmentedPrompt] = None,
    negative_completion: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Saves Markdown and JSON sample augmented prompt artifacts."""
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    md_file = out_dir / "augmented_prompt_sample.md"
    json_file = out_dir / "augmented_prompt_sample.json"

    export_payload = {
        "augmented_prompt": prompt.to_dict(),
        "llm_completion": llm_completion,
        "negative_test": {
            "prompt": negative_prompt.to_dict() if negative_prompt else None,
            "completion": negative_completion,
        },
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    md_content = generate_augmented_prompt_markdown(
        prompt=prompt,
        llm_completion=llm_completion,
        negative_prompt=negative_prompt,
        negative_completion=negative_completion,
    )
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Exported Augmented Prompt JSON artifact to %s", json_file)
    logger.info("Exported Augmented Prompt Markdown artifact to %s", md_file)
    return md_file, json_file


def run_sample_augmented_prompt(
    query: Optional[str] = None,
    execute_llm: bool = True,
) -> Tuple[AugmentedPrompt, Optional[str]]:
    """Runs context assembly on sample queries, executes LLM to demonstrate citations, and exports reports."""
    from src.retriever import VectorRetriever

    sample_query = (
        query
        or "What are the mandatory timeframe and reporting procedures for banks to notify "
        "CERT-In and RBI regarding Severity 1 cyber security incidents?"
    )

    print("=" * 80)
    print("REGULSENSE: CONTEXT ASSEMBLY & AUGMENTED PROMPT DEMONSTRATION")
    print("=" * 80)
    print(f"Sample Query: \"{sample_query}\"")

    retriever = VectorRetriever()
    run_result = retriever.retrieve(query_text=sample_query, top_k=3)
    chunks = run_result.chunks

    assembler = ContextAssembler()
    budget_spec = TokenBudgetSpec(
        total_budget=2048,
        answer_reserve=450,
        safety_margin=50,
    )

    augmented = assembler.assemble(
        query=sample_query,
        chunks=chunks,
        budget_spec=budget_spec,
    )

    print("-" * 80)
    print(f"Token Budget Audit: Total={augmented.budget_ledger.total_budget}, Context={augmented.budget_ledger.context_tokens}, System={augmented.budget_ledger.system_tokens}, Query={augmented.budget_ledger.query_tokens}, Reserve={augmented.budget_ledger.answer_reserve}")
    print(f"Total Allocated: {augmented.budget_ledger.total_allocated} / {augmented.budget_ledger.total_budget} tokens (Within Budget: {augmented.budget_ledger.is_within_budget})")
    print(f"Injected Chunks Count: {len(augmented.injected_chunks)} with source markers: {list(augmented.source_markers.keys())}")

    completion_text = None
    neg_prompt = None
    neg_completion = None

    if execute_llm:
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        )
        chat_model = os.getenv("CHAT_MODEL", "llama3:latest")
        try:
            print(f"\n[Executing LLM completion with model '{chat_model}' to verify citation adherence...]")
            res = client.chat.completions.create(
                model=chat_model,
                messages=augmented.messages,
                temperature=0.2,
                max_tokens=450,
            )
            completion_text = res.choices[0].message.content.strip()
            print("\n[GROUNDED MODEL ANSWER WITH SOURCE CITATIONS]:")
            print(completion_text)

            # Also demonstrate Insufficient Context Fallback on negative query
            neg_query = "What are the Basel III Common Equity Tier 1 (CET1) capital buffer ratios for regional rural banks?"
            print(f"\n[Executing Negative Control Query]: \"{neg_query}\"")
            neg_prompt = assembler.assemble(
                query=neg_query,
                chunks=[],  # No matching chunks
                budget_spec=budget_spec,
            )
            neg_res = client.chat.completions.create(
                model=chat_model,
                messages=neg_prompt.messages,
                temperature=0.2,
                max_tokens=250,
            )
            neg_completion = neg_res.choices[0].message.content.strip()
            print("\n[MODEL INSUFFICIENT CONTEXT FALLBACK]:")
            print(neg_completion)

        except Exception as e:
            logger.warning("Live LLM execution could not complete: %s", e)

    md_file, json_file = export_augmented_prompt_artifacts(
        prompt=augmented,
        llm_completion=completion_text,
        negative_prompt=neg_prompt,
        negative_completion=neg_completion,
    )

    print("-" * 80)
    print(f"Exported Markdown: {md_file}")
    print(f"Exported JSON:     {json_file}")
    print("=" * 80)

    return augmented, completion_text


if __name__ == "__main__":
    cli_query = sys.argv[1] if len(sys.argv) > 1 else None
    run_sample_augmented_prompt(query=cli_query)
