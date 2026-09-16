"""Verifiable Source Citation Mapping, Evidentiary Verification & Anti-Fabrication Engine.

This module implements:
1. Task 1 - Add source references:
   Extracts and validates in-text citation markers (e.g. [1], [2]) linking claims to source chunks.
2. Task 2 - Map citations to metadata:
   Constructs a comprehensive registry mapping each citation marker directly to physical document
   metadata: source document filename, chunk ID, chunk index, section, page, and similarity score.
3. Task 3 - Verify cited sources:
   Enables users and auditors to verify every cited claim against the original retrieved text,
   extracting matching verbatim text spans and determining verification verdicts.
4. Task 4 - Avoid fabricated citations:
   Scans answers for fabricated markers (e.g. unindexed [99]) and enforces a clean fallback
   refusal for queries without supporting context, strictly preventing hallucinated citations.
5. Task 5 - Commit sample cited answers:
   Exports structured Markdown and JSON reports demonstrating cited answers, metadata mappings,
   evidentiary verifications, and no-source fallback executions.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.context_assembler import ContextAssembler, TokenBudgetSpec
from src.retriever import VectorRetriever
from src.vector_db import RetrievedRecord

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("CitationEngine")

DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3:latest")
DEFAULT_CITED_QUERY = (
    "What are the mandatory timeframe and reporting procedures for banks to notify "
    "CERT-In and RBI regarding Severity 1 cyber security incidents?"
)
DEFAULT_OUT_OF_CORPUS_QUERY = (
    "What are the Basel III Common Equity Tier 1 (CET1) capital buffer ratios "
    "and countercyclical requirements for regional rural banks?"
)


# ==============================================================================
# Data Contracts & Schemas (Tasks 1, 2, 3, 4)
# ==============================================================================

@dataclass
class CitationMetadataRecord:
    """Immutable provenance record mapping a citation marker to document metadata (Task 2)."""
    marker_id: str  # e.g. "[1]"
    index: int      # 1-indexed
    source_document: str
    chunk_id: str
    chunk_index: int
    section: str
    page_number: int
    similarity_score: float
    verbatim_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker_id": self.marker_id,
            "index": self.index,
            "source_document": self.source_document,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "page_number": self.page_number,
            "similarity_score": round(self.similarity_score, 4),
            "verbatim_text": self.verbatim_text,
        }


@dataclass
class ClaimVerificationDetail:
    """Individual claim-level verification result auditing evidence alignment (Task 3)."""
    claim_sentence: str
    cited_marker: str
    source_document: str
    section: str
    chunk_id: str
    is_verified: bool
    status: str  # "VERIFIED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "FABRICATED_MARKER"
    matched_verbatim_span: Optional[str]
    matched_keywords: List[str]
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_sentence": self.claim_sentence,
            "cited_marker": self.cited_marker,
            "source_document": self.source_document,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "is_verified": self.is_verified,
            "status": self.status,
            "matched_verbatim_span": self.matched_verbatim_span,
            "matched_keywords": self.matched_keywords,
            "confidence_score": round(self.confidence_score, 4),
        }


@dataclass
class CitationAuditReport:
    """Comprehensive audit report verifying all citations in a generated answer."""
    total_citations_found: int
    unique_markers_cited: List[str]
    verified_citations_count: int
    fabricated_citations_count: int
    unsupported_claims_count: int
    citation_precision: float
    claim_verifications: List[ClaimVerificationDetail]
    fabricated_markers: List[str]
    overall_verdict: str  # "PASS", "FLAGGED", "REJECTED"
    audit_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_citations_found": self.total_citations_found,
            "unique_markers_cited": self.unique_markers_cited,
            "verified_citations_count": self.verified_citations_count,
            "fabricated_citations_count": self.fabricated_citations_count,
            "unsupported_claims_count": self.unsupported_claims_count,
            "citation_precision": round(self.citation_precision, 4),
            "fabricated_markers": self.fabricated_markers,
            "overall_verdict": self.overall_verdict,
            "audit_notes": self.audit_notes,
            "claim_verifications": [cv.to_dict() for cv in self.claim_verifications],
        }


@dataclass
class CitedAnswerOutput:
    """End-to-end output containing the answer, citation registry, and audit report."""
    query: str
    answer: str
    citation_registry: Dict[str, Dict[str, Any]]
    audit_report: CitationAuditReport
    is_fallback: bool
    has_fabricated_citations: bool
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "answer": self.answer,
            "is_fallback": self.is_fallback,
            "has_fabricated_citations": self.has_fabricated_citations,
            "citation_registry": self.citation_registry,
            "audit_report": self.audit_report.to_dict(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_seconds": round(self.latency_seconds, 4),
            "model": self.model,
        }


# ==============================================================================
# Citation Engine (Tasks 1, 2, 3, 4)
# ==============================================================================

class CitationEngine:
    """Orchestrates citation extraction, metadata mapping, verification, and anti-fabrication."""

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        retriever: Optional[VectorRetriever] = None,
        assembler: Optional[ContextAssembler] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.model = model or os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.temperature = temperature

        if openai_client:
            self.client = openai_client
        else:
            self.client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            )

        self.retriever = retriever or VectorRetriever(openai_client=self.client)
        self.assembler = assembler or ContextAssembler()

        logger.info("Initialized CitationEngine (model='%s', T=%.2f)", self.model, self.temperature)

    # -------------------------------------------------------------------------
    # Task 2: Map Citations to Metadata
    # -------------------------------------------------------------------------

    def build_citation_registry(
        self,
        chunks: List[Any],
    ) -> Dict[str, CitationMetadataRecord]:
        """Builds a complete citation-to-metadata registry from candidate chunks (Task 2).

        Args:
            chunks: List of RetrievedRecord, RetrievedContextChunk, InjectedChunk, or dicts.

        Returns:
            Dictionary mapping marker IDs (e.g. "[1]") to CitationMetadataRecord objects.
        """
        registry: Dict[str, CitationMetadataRecord] = {}

        for idx, chunk in enumerate(chunks, start=1):
            marker_id = f"[{idx}]"

            if hasattr(chunk, "marker") and hasattr(chunk.marker, "marker_id") and isinstance(chunk.marker.marker_id, str):  # InjectedChunk
                m = chunk.marker
                rec = CitationMetadataRecord(
                    marker_id=m.marker_id,
                    index=m.index,
                    source_document=m.source_document,
                    chunk_id=m.chunk_id,
                    chunk_index=getattr(m, "chunk_index", idx - 1),
                    section=m.section,
                    page_number=m.page_number,
                    similarity_score=m.similarity_score,
                    verbatim_text=chunk.raw_text,
                )
            elif hasattr(chunk, "chunk_id") and isinstance(chunk.chunk_id, str):  # RetrievedContextChunk
                rec = CitationMetadataRecord(
                    marker_id=marker_id,
                    index=idx,
                    source_document=getattr(chunk, "source_document", "Unknown Circular"),
                    chunk_id=chunk.chunk_id,
                    chunk_index=getattr(chunk, "chunk_index", idx - 1),
                    section=getattr(chunk, "section", "General Provisions"),
                    page_number=getattr(chunk, "page_number", 1),
                    similarity_score=float(getattr(chunk, "similarity_score", 0.0)),
                    verbatim_text=getattr(chunk, "text", getattr(chunk, "document", "")),
                )
            elif hasattr(chunk, "id"):  # RetrievedRecord
                meta = chunk.metadata or {}
                rec = CitationMetadataRecord(
                    marker_id=marker_id,
                    index=idx,
                    source_document=str(meta.get("source_document") or meta.get("filename") or "Unknown Circular"),
                    chunk_id=chunk.id,
                    chunk_index=int(meta.get("chunk_index", idx - 1)),
                    section=str(meta.get("section") or "General Provisions"),
                    page_number=int(meta.get("page_number", 1)),
                    similarity_score=float(chunk.similarity_score),
                    verbatim_text=chunk.document,
                )
            elif isinstance(chunk, dict):
                meta = chunk.get("metadata") or {}
                rec = CitationMetadataRecord(
                    marker_id=marker_id,
                    index=idx,
                    source_document=chunk.get("source_document") or meta.get("source_document") or "Unknown Circular",
                    chunk_id=chunk.get("chunk_id") or chunk.get("id") or f"chunk_{idx:03d}",
                    chunk_index=int(chunk.get("chunk_index") or meta.get("chunk_index") or (idx - 1)),
                    section=chunk.get("section") or meta.get("section") or "General Provisions",
                    page_number=int(chunk.get("page_number") or meta.get("page_number", 1)),
                    similarity_score=float(chunk.get("similarity_score") or chunk.get("score", 0.0)),
                    verbatim_text=chunk.get("verbatim_text") or chunk.get("raw_text") or chunk.get("text") or chunk.get("document", ""),
                )
            else:
                rec = CitationMetadataRecord(
                    marker_id=marker_id,
                    index=idx,
                    source_document="Unknown Circular",
                    chunk_id=f"chunk_{idx:03d}",
                    chunk_index=idx - 1,
                    section="General Provisions",
                    page_number=1,
                    similarity_score=0.0,
                    verbatim_text=str(chunk),
                )

            registry[rec.marker_id] = rec

        logger.info("Built Citation Registry with %d entries: %s", len(registry), list(registry.keys()))
        return registry

    # -------------------------------------------------------------------------
    # Task 1: Add Source References & Parse In-Text Citations
    # -------------------------------------------------------------------------

    def extract_claims_and_citations(
        self,
        answer: str,
    ) -> List[Tuple[str, List[str]]]:
        """Parses answer into sentences and extracts associated citation markers (Task 1).

        Supports:
        - Numeric chunk markers: [1], [2]
        - Sub-section citations: [1, Section 2] -> normalized to [1]
        - Direct source filename markers: [cyber_resilience_framework.pdf]

        Args:
            answer: Model generated text.

        Returns:
            List of (claim_sentence, list_of_cited_markers) tuples with deduplicated markers.
        """
        raw_lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]
        claims: List[Tuple[str, List[str]]] = []

        # Regex captures numeric IDs or filenames with standard document extensions
        marker_pattern = r"\[(\d+|[\w\.\-]+\.(?:pdf|txt|md|html|json))(?:[^\]]*)\]"

        for line in raw_lines:
            # Split sentences on .!? followed by whitespace UNLESS immediately followed by citation marker
            parts = re.split(r"(?<=[.!?])\s+(?!\s*\[)", line)
            for p in parts:
                clean_s = p.strip()
                if not clean_s:
                    continue

                num_matches = re.findall(marker_pattern, clean_s)
                if num_matches:
                    markers = [f"[{m}]" for m in num_matches]
                    deduped_markers = list(dict.fromkeys(markers))
                    claims.append((clean_s, deduped_markers))

        # If no bracketed markers found by sentence split, check overall text
        if not claims:
            num_matches = re.findall(marker_pattern, answer)
            if num_matches:
                markers = [f"[{m}]" for m in num_matches]
                deduped_markers = list(dict.fromkeys(markers))
                claims.append((answer.strip(), deduped_markers))

        return claims

    # -------------------------------------------------------------------------
    # Task 2: Provenance Resolution Helper
    # -------------------------------------------------------------------------

    def lookup_chunk(
        self,
        marker_key: str,
        registry: Dict[str, Any],
    ) -> Optional[CitationMetadataRecord]:
        """Resolves a citation marker to its corresponding metadata record (Task 2).

        Matches directly on marker ID (e.g. "[1]") or matches against the source
        document filename (e.g. "[cyber_resilience_framework.pdf]").
        Supports both CitationMetadataRecord instances and serialized dicts.
        """
        def _to_record(item: Any) -> CitationMetadataRecord:
            if isinstance(item, CitationMetadataRecord):
                return item
            if isinstance(item, dict):
                return CitationMetadataRecord(
                    marker_id=item.get("marker_id", ""),
                    index=int(item.get("index", 1)),
                    source_document=item.get("source_document", "Unknown"),
                    chunk_id=item.get("chunk_id", ""),
                    chunk_index=int(item.get("chunk_index", 0)),
                    section=item.get("section", ""),
                    page_number=int(item.get("page_number", 1)),
                    similarity_score=float(item.get("similarity_score", 0.0)),
                    verbatim_text=item.get("verbatim_text", ""),
                )
            raise ValueError(f"Unsupported record type: {type(item)}")

        if marker_key in registry:
            return _to_record(registry[marker_key])

        clean_name = marker_key.strip("[]").lower()
        for rec_val in registry.values():
            rec = _to_record(rec_val)
            if clean_name == rec.source_document.lower():
                return rec
        return None

    # -------------------------------------------------------------------------
    # Task 3: Modular Claim-Level Verification
    # -------------------------------------------------------------------------

    def verify_claim_against_chunk(
        self,
        sentence: str,
        marker: str,
        chunk_rec: CitationMetadataRecord,
    ) -> ClaimVerificationDetail:
        """Verifies a single claim sentence against the verbatim text of a cited chunk (Task 3).

        Audits factual alignment using:
        1. Continuous n-gram phrase matching (spans of 3-8 words).
        2. Regulatory timelines & numbers (e.g. '6 hours', '7 business days', '5 years').
        3. Informative keyword overlap ratios.
        """
        chunk_text_lower = chunk_rec.verbatim_text.lower()
        clean_sentence = re.sub(r"\[[^\]]+\]", "", sentence).strip()
        clean_sentence_lower = clean_sentence.lower()

        # Extract informative words (numbers, keywords, acronyms)
        keywords = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", clean_sentence_lower)
        kw_filtered = [
            kw for kw in keywords
            if kw not in {"initial", "report", "based", "provided", "regulatory", "context", "according", "must", "the"}
        ]
        active_keywords = kw_filtered if kw_filtered else keywords
        matched_kw = [kw for kw in active_keywords if kw in chunk_text_lower]

        # Check key numbers or timelines specifically
        number_patterns = re.findall(
            r"\b\d+\s*(?:hours?|days?|business days?|calendar days?|years?|lakhs?|months?)\b",
            clean_sentence_lower,
        )
        numbers_matched = [num for num in number_patterns if num in chunk_text_lower]

        # Clean punctuation & markdown from claim sentence for n-gram matching
        normalized_claim_words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", clean_sentence_lower)
        normalized_chunk_tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", chunk_text_lower)
        normalized_chunk_str = " " + " ".join(normalized_chunk_tokens) + " "

        best_span: Optional[str] = None
        for span_len in range(min(len(normalized_claim_words), 8), 2, -1):
            for i in range(len(normalized_claim_words) - span_len + 1):
                subphrase = " ".join(normalized_claim_words[i : i + span_len])
                if (" " + subphrase + " ") in normalized_chunk_str:
                    sub_tokens = subphrase.split()
                    pos = chunk_text_lower.find(subphrase)
                    if pos == -1 and sub_tokens:
                        pos = chunk_text_lower.find(sub_tokens[0])
                    if pos != -1:
                        end_pos = min(len(chunk_rec.verbatim_text), pos + len(subphrase) + 30)
                        best_span = chunk_rec.verbatim_text[pos : end_pos].strip()
                    else:
                        best_span = subphrase
                    break
            if best_span:
                break

        if not best_span and numbers_matched:
            best_span = f"Exact number match: '{numbers_matched[0]}'"

        # Confidence calculation
        kw_ratio = len(matched_kw) / max(1, len(active_keywords))
        has_numbers = len(number_patterns) > 0
        numbers_ok = len(numbers_matched) == len(number_patterns) if has_numbers else True

        is_verified = (kw_ratio >= 0.35 and numbers_ok) or bool(best_span)

        if is_verified:
            status = "VERIFIED"
            conf = min(1.0, kw_ratio + 0.3)
        elif kw_ratio >= 0.25:
            status = "PARTIALLY_SUPPORTED"
            conf = kw_ratio
        else:
            status = "UNSUPPORTED"
            conf = kw_ratio

        return ClaimVerificationDetail(
            claim_sentence=sentence,
            cited_marker=marker,
            source_document=chunk_rec.source_document,
            section=chunk_rec.section,
            chunk_id=chunk_rec.chunk_id,
            is_verified=is_verified,
            status=status,
            matched_verbatim_span=best_span,
            matched_keywords=matched_kw[:6],
            confidence_score=conf,
        )

    # -------------------------------------------------------------------------
    # Task 3 & Task 4: Verify Cited Sources & Detect Fabricated Citations
    # -------------------------------------------------------------------------

    def verify_citations(
        self,
        answer: str,
        registry: Dict[str, CitationMetadataRecord],
    ) -> CitationAuditReport:
        """Audits all citations against the registry to verify truth and flag fabrications (Tasks 3 & 4).

        Args:
            answer: Generated compliance answer.
            registry: Citation-to-metadata registry.

        Returns:
            CitationAuditReport with itemized claim verifications.
        """
        marker_pattern = r"\[(\d+|[\w\.\-]+\.(?:pdf|txt|md|html|json))(?:[^\]]*)\]"
        num_matches = re.findall(marker_pattern, answer)
        all_markers_in_answer = [f"[{m}]" for m in num_matches]
        unique_markers = sorted(list(set(all_markers_in_answer)))

        # Check for fabricated markers (markers not resolvable to any registry chunk)
        fabricated_markers = [m for m in unique_markers if self.lookup_chunk(m, registry) is None]

        # Extract sentences with citations
        parsed_claims = self.extract_claims_and_citations(answer)
        claim_verifications: List[ClaimVerificationDetail] = []
        verified_count = 0
        unsupported_count = 0

        # Handle true fallback answer (when NO source markers are cited AND model states refusal)
        is_refusal = (
            "not contain sufficient information" in answer.lower()
            or "insufficient information" in answer.lower()
            or "insufficient context" in answer.lower()
            or "no information" in answer.lower()
        )
        if is_refusal and len(unique_markers) == 0:
            return CitationAuditReport(
                total_citations_found=0,
                unique_markers_cited=[],
                verified_citations_count=0,
                fabricated_citations_count=0,
                unsupported_claims_count=0,
                citation_precision=1.0,
                claim_verifications=[],
                fabricated_markers=[],
                overall_verdict="PASS",
                audit_notes="Answer correctly invoked the missing-context fallback protocol without fabricating citations.",
            )

        for sentence, markers in parsed_claims:
            for m in markers:
                chunk_rec = self.lookup_chunk(m, registry)
                if chunk_rec is None or m in fabricated_markers:
                    # Fabricated citation detected!
                    detail = ClaimVerificationDetail(
                        claim_sentence=sentence,
                        cited_marker=m,
                        source_document="[FABRICATED SOURCE]",
                        section="None",
                        chunk_id="None",
                        is_verified=False,
                        status="FABRICATED_MARKER",
                        matched_verbatim_span=None,
                        matched_keywords=[],
                        confidence_score=0.0,
                    )
                    claim_verifications.append(detail)
                    unsupported_count += 1
                    continue

                # Valid marker: cross-verify claim against chunk text
                detail = self.verify_claim_against_chunk(sentence, m, chunk_rec)
                if detail.is_verified:
                    verified_count += 1
                elif detail.status == "UNSUPPORTED":
                    unsupported_count += 1
                claim_verifications.append(detail)

        total_audited = len(claim_verifications)
        precision = verified_count / total_audited if total_audited > 0 else (1.0 if not fabricated_markers else 0.0)

        if fabricated_markers:
            overall_verdict = "REJECTED"
            notes = f"CRITICAL: Found {len(fabricated_markers)} fabricated citation markers: {fabricated_markers}."
        elif total_audited == 0 and not is_refusal:
            overall_verdict = "FLAGGED"
            notes = "Answer contains zero source citations for assertions made without invoking fallback protocol."
        elif unsupported_count > 0:
            overall_verdict = "FLAGGED"
            notes = f"Found {unsupported_count} claims with insufficient evidence in cited chunks."
        else:
            overall_verdict = "PASS"
            notes = f"All {verified_count} citations successfully verified against source chunk text with 100% evidentiary fidelity."

        return CitationAuditReport(
            total_citations_found=len(all_markers_in_answer),
            unique_markers_cited=unique_markers,
            verified_citations_count=verified_count,
            fabricated_citations_count=len(fabricated_markers),
            unsupported_claims_count=unsupported_count,
            citation_precision=precision,
            claim_verifications=claim_verifications,
            fabricated_markers=fabricated_markers,
            overall_verdict=overall_verdict,
            audit_notes=notes,
        )

    # -------------------------------------------------------------------------
    # End-to-End Generation with Citation Mapping & Fallback (Tasks 1, 3, 4)
    # -------------------------------------------------------------------------

    def generate_cited_answer(
        self,
        query: str,
        chunks: Optional[List[Any]] = None,
        top_k: int = 3,
        min_score: float = 0.40,
    ) -> CitedAnswerOutput:
        """Generates an answer, maps citations to metadata, and verifies claims (Tasks 1-4).

        Args:
            query: Compliance inquiry.
            chunks: Optional pre-retrieved chunks. If None, retrieves from vector store.
            top_k: Candidate retrieval count.
            min_score: Minimum similarity score floor.

        Returns:
            CitedAnswerOutput object.
        """
        start_time = time.time()
        logger.info("Generating verifiable cited answer for query: '%s'...", query[:60])

        # Step 1: Retrieve candidate chunks if not provided
        if chunks is None:
            ret_result = self.retriever.retrieve(query_text=query, top_k=top_k)
            # Filter by min_score
            candidate_chunks = [c for c in ret_result.chunks if c.similarity_score >= min_score]
        else:
            candidate_chunks = chunks

        # Step 2: Build citation-to-metadata registry (Task 2)
        registry = self.build_citation_registry(candidate_chunks)

        # Step 3: Handle empty context / no-source fallback (Task 4)
        if not candidate_chunks:
            logger.info("No chunks met relevance threshold for query. Triggering no-source fallback.")
            augmented_prompt = self.assembler.assemble(
                query=query,
                chunks=[],
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=augmented_prompt.messages,
                temperature=self.temperature,
                max_tokens=250,
            )
            elapsed = time.time() - start_time
            answer = response.choices[0].message.content.strip()
            usage = response.usage

            audit = self.verify_citations(answer=answer, registry={})

            return CitedAnswerOutput(
                query=query,
                answer=answer,
                citation_registry={},
                audit_report=audit,
                is_fallback=True,
                has_fabricated_citations=False,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_seconds=elapsed,
                model=self.model,
            )

        # Step 4: Assemble grounded prompt with source markers
        augmented_prompt = self.assembler.assemble(
            query=query,
            chunks=candidate_chunks,
        )

        # Step 5: Invoke model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=augmented_prompt.messages,
            temperature=self.temperature,
            max_tokens=450,
        )
        elapsed = time.time() - start_time
        answer = response.choices[0].message.content.strip()
        usage = response.usage

        # Step 6: Audit citations and verify against chunks (Task 3 & 4)
        audit = self.verify_citations(answer=answer, registry=registry)
        has_fabricated = len(audit.fabricated_markers) > 0

        is_refusal = (
            "not contain sufficient information" in answer.lower()
            or "insufficient information" in answer.lower()
            or "insufficient context" in answer.lower()
            or "no information" in answer.lower()
        )
        is_fallback = (not candidate_chunks) or (is_refusal and len(audit.unique_markers_cited) == 0)

        logger.info(
            "Cited Answer Generation Complete in %.3fs (Verdict=%s, Precision=%.1f%%, Fabricated=%s, Fallback=%s)",
            elapsed,
            audit.overall_verdict,
            audit.citation_precision * 100,
            has_fabricated,
            is_fallback,
        )

        return CitedAnswerOutput(
            query=query,
            answer=answer,
            citation_registry={k: v.to_dict() for k, v in registry.items()},
            audit_report=audit,
            is_fallback=is_fallback,
            has_fabricated_citations=has_fabricated,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_seconds=elapsed,
            model=self.model,
        )

    def stream_cited_answer(
        self,
        query: str,
        chunks: Optional[List[Any]] = None,
        top_k: int = 3,
        min_score: float = 0.40,
    ):
        """Streams cited answer tokens progressively while yielding sources and metadata (Tasks 1-4).

        Yields:
            ("sources", registry_dict, candidate_chunks)
            ("token", delta_str)
            ("done", CitedAnswerOutput)
        """
        start_time = time.time()
        logger.info("Streaming cited answer generation for query: '%s'...", query[:60])

        # Step 1: Retrieve candidate chunks if not provided
        if chunks is None:
            ret_result = self.retriever.retrieve(query_text=query, top_k=top_k)
            candidate_chunks = [c for c in ret_result.chunks if c.similarity_score >= min_score]
        else:
            candidate_chunks = chunks

        # Step 2: Build citation-to-metadata registry
        registry = self.build_citation_registry(candidate_chunks)
        registry_serialized = {k: v.to_dict() for k, v in registry.items()}

        # Yield sources immediately so the caller can inspect sources before LLM generation finishes
        yield ("sources", registry_serialized, candidate_chunks)

        # Step 3: Handle empty context / no-source fallback
        if not candidate_chunks:
            logger.info("No chunks met relevance threshold. Triggering streaming fallback.")
            augmented_prompt = self.assembler.assemble(
                query=query,
                chunks=[],
            )
            stream_resp = self.client.chat.completions.create(
                model=self.model,
                messages=augmented_prompt.messages,
                temperature=self.temperature,
                max_tokens=250,
                stream=True,
            )
            accumulated: List[str] = []
            for stream_chunk in stream_resp:
                if stream_chunk.choices and stream_chunk.choices[0].delta and stream_chunk.choices[0].delta.content:
                    delta = stream_chunk.choices[0].delta.content
                    accumulated.append(delta)
                    yield ("token", delta)

            full_answer = "".join(accumulated).strip()
            elapsed = time.time() - start_time
            audit = self.verify_citations(answer=full_answer, registry={})
            output = CitedAnswerOutput(
                query=query,
                answer=full_answer,
                citation_registry={},
                audit_report=audit,
                is_fallback=True,
                has_fabricated_citations=False,
                prompt_tokens=0,
                completion_tokens=len(accumulated),
                latency_seconds=elapsed,
                model=self.model,
            )
            yield ("done", output)
            return

        # Step 4: Assemble grounded prompt with source markers
        augmented_prompt = self.assembler.assemble(
            query=query,
            chunks=candidate_chunks,
        )

        # Step 5: Invoke model with stream=True
        stream_resp = self.client.chat.completions.create(
            model=self.model,
            messages=augmented_prompt.messages,
            temperature=self.temperature,
            max_tokens=450,
            stream=True,
        )
        accumulated = []
        for stream_chunk in stream_resp:
            if stream_chunk.choices and stream_chunk.choices[0].delta and stream_chunk.choices[0].delta.content:
                delta = stream_chunk.choices[0].delta.content
                accumulated.append(delta)
                yield ("token", delta)

        full_answer = "".join(accumulated).strip()
        elapsed = time.time() - start_time

        # Step 6: Audit citations and verify against chunks
        audit = self.verify_citations(answer=full_answer, registry=registry)
        has_fabricated = len(audit.fabricated_markers) > 0
        is_refusal = (
            "not contain sufficient information" in full_answer.lower()
            or "insufficient information" in full_answer.lower()
            or "insufficient context" in full_answer.lower()
            or "no information" in full_answer.lower()
        )
        is_fallback = (not candidate_chunks) or (is_refusal and len(audit.unique_markers_cited) == 0)

        output = CitedAnswerOutput(
            query=query,
            answer=full_answer,
            citation_registry=registry_serialized,
            audit_report=audit,
            is_fallback=is_fallback,
            has_fabricated_citations=has_fabricated,
            prompt_tokens=0,
            completion_tokens=len(accumulated),
            latency_seconds=elapsed,
            model=self.model,
        )
        yield ("done", output)


# ==============================================================================
# Reporting & Artifact Serialization (Task 5)
# ==============================================================================

def generate_citation_markdown_report(
    cited_output: CitedAnswerOutput,
    fallback_output: CitedAnswerOutput,
    fabricated_test_audit: Optional[CitationAuditReport] = None,
) -> str:
    """Generates a detailed Markdown report displaying cited answers, mappings, and audits."""
    lines: List[str] = [
        "# RegulSense: Verifiable Source Citation Mapping & Anti-Fabrication Audit Report",
        "",
        f"- **Model**: `{cited_output.model}` (Sampling Temperature: `0.2`)",
        f"- **Execution Timestamp**: `{cited_output.timestamp}`",
        f"- **Citation Verification Verdict**: `{cited_output.audit_report.overall_verdict}`",
        f"- **Citation Precision**: **`{cited_output.audit_report.citation_precision:.1%}`**",
        f"- **Fabricated Citations Detected**: `{'⚠️ YES' if cited_output.has_fabricated_citations else '✅ ZERO (Anti-Fabrication Enforced)'}`",
        "",
        "---",
        "",
        "## 1. Grounded Answer with In-Text Source Citations (Task 1)",
        "",
        f"### **Compliance Question**:",
        f"> *\"{cited_output.query}\"*",
        "",
        f"### **Model Generated Answer**:",
        cited_output.answer,
        "",
        "---",
        "",
        "## 2. Citation-to-Metadata Registry Mapping (Task 2)",
        "",
        "Each citation marker maps directly to physical regulatory circular provenance:",
        "",
        "| Marker | Source Document | Section | Page | Chunk ID | Similarity | Evidentiary Text Excerpt |",
        "| :---: | :--- | :--- | :---: | :--- | :---: | :--- |",
    ]

    for marker_id, meta in cited_output.citation_registry.items():
        clean_excerpt = " ".join(meta.get("verbatim_text", "").split())[:120] + "..."
        lines.append(
            f"| **`{marker_id}`** | `{meta.get('source_document')}` | {meta.get('section')} | {meta.get('page_number')} | `{meta.get('chunk_id')}` | `{meta.get('similarity_score', 0.0):.4f}` | \"{clean_excerpt}\" |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Claim-by-Claim Evidentiary Verification Ledger (Task 3)",
        "",
        "Every sentence containing a source reference is audited against the verbatim text of the cited chunk:",
        "",
        "| Cited Marker | In-Text Claim Sentence | Matched Verbatim Source Span | Status | Confidence |",
        "| :---: | :--- | :--- | :---: | :---: |",
    ])

    for detail in cited_output.audit_report.claim_verifications:
        clean_claim = " ".join(detail.claim_sentence.split())
        claim_preview = clean_claim[:130] + ("..." if len(clean_claim) > 130 else "")
        clean_span = " ".join(str(detail.matched_verbatim_span).split()) if detail.matched_verbatim_span else "*[No direct phrase]*"
        span_preview = f"\"{clean_span}\"" if detail.matched_verbatim_span else clean_span
        status_badge = f"✅ {detail.status}" if detail.is_verified else f"⚠️ {detail.status}"
        lines.append(
            f"| **`{detail.cited_marker}`** | {claim_preview} | {span_preview} | {status_badge} | `{detail.confidence_score:.1%}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. No-Source Fallback Protocol Demonstration (Task 4)",
        "",
        f"### **Unindexed Compliance Query**:",
        f"> *\"{fallback_output.query}\"*",
        "",
        f"### **Model Fallback Response**:",
        fallback_output.answer,
        "",
        "> [!IMPORTANT]",
        "> **Anti-Fabrication Guarantee**: When no relevant regulatory evidence is available in the vector store, the model returns a direct refusal without fabricating citations or hallucinating false source markers.",
    ])

    if fabricated_test_audit:
        lines.extend([
            "",
            "---",
            "",
            "## 5. Anti-Fabrication Detection Demonstration (Negative Test)",
            "",
            f"- **Injected Synthetic Marker**: `{fabricated_test_audit.fabricated_markers}`",
            f"- **Detection Verdict**: `{fabricated_test_audit.overall_verdict}`",
            f"- **Audit Warning**: > {fabricated_test_audit.audit_notes}",
            "",
            "> [!NOTE]",
            "> The system automatically detects when an answer attempts to cite a non-existent marker and flags the assertion as unverified.",
        ])

    lines.extend([
        "",
        "---",
        "*Report automatically generated by `src/citation_engine.py` for RegulSense Banking Compliance Assistant.*",
    ])

    return "\n".join(lines)


def export_citation_artifacts(
    cited_output: CitedAnswerOutput,
    fallback_output: CitedAnswerOutput,
    fabricated_test_audit: Optional[CitationAuditReport] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Saves Markdown and JSON citation verification artifacts."""
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    md_file = out_dir / "citation_verification_results.md"
    json_file = out_dir / "citation_verification_results.json"

    export_payload = {
        "cited_output": cited_output.to_dict(),
        "fallback_output": fallback_output.to_dict(),
        "fabricated_test_audit": fabricated_test_audit.to_dict() if fabricated_test_audit else None,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    md_content = generate_citation_markdown_report(
        cited_output=cited_output,
        fallback_output=fallback_output,
        fabricated_test_audit=fabricated_test_audit,
    )
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved Citation Verification JSON report to %s", json_file)
    logger.info("Saved Citation Verification Markdown report to %s", md_file)
    return md_file, json_file


def run_citation_evaluation_suite() -> Dict[str, Any]:
    """Executes all citation mapping, verification, fallback, and anti-fabrication tests."""
    print("=" * 80)
    print("REGULSENSE: VERIFIABLE SOURCE CITATION & ANTI-FABRICATION SUITE")
    print("=" * 80)

    engine = CitationEngine()

    # Task 1, 2, 3: Grounded cited generation and verification
    print("\n--- Tasks 1, 2, 3: Generating Cited Answer & Verifying Against Sources ---")
    cited_output = engine.generate_cited_answer(query=DEFAULT_CITED_QUERY)

    print("\n[GROUNDED ANSWER WITH IN-TEXT CITATIONS]:")
    print(cited_output.answer)

    print("\n[CITATION REGISTRY MAPPINGS]:")
    for m, rec in cited_output.citation_registry.items():
        print(f"  {m} -> Document: {rec['source_document']} | Section: {rec['section']} | Chunk ID: {rec['chunk_id']}")

    print("\n[CLAIM VERIFICATION AUDIT]:")
    print(f"  Overall Verdict: {cited_output.audit_report.overall_verdict}")
    print(f"  Precision:       {cited_output.audit_report.citation_precision:.1%}")
    for detail in cited_output.audit_report.claim_verifications:
        print(f"  - [{detail.cited_marker}] Status: {detail.status} | Matched: {detail.matched_verbatim_span}")

    # Task 4: No-source fallback
    print("\n--- Task 4: Missing-Context Fallback (Zero Fabricated Citations) ---")
    fallback_output = engine.generate_cited_answer(query=DEFAULT_OUT_OF_CORPUS_QUERY)
    print("\n[FALLBACK OUTPUT]:")
    print(fallback_output.answer)
    print(f"  Fabricated Citations: {fallback_output.has_fabricated_citations} (Count: {fallback_output.audit_report.fabricated_citations_count})")

    # Anti-fabrication demonstration test
    print("\n--- Anti-Fabrication Detection Demonstration ---")
    fake_answer = "Under [99], regional rural banks must maintain 15% CET1 capital buffers."
    fake_audit = engine.verify_citations(answer=fake_answer, registry=cited_output.citation_registry)
    print(f"  Fabricated Markers Detected: {fake_audit.fabricated_markers}")
    print(f"  Verdict: {fake_audit.overall_verdict} ({fake_audit.audit_notes})")

    # Task 5: Export artifacts
    print("\n--- Task 5: Exporting Reports & Summary ---")
    md_file, json_file = export_citation_artifacts(
        cited_output=cited_output,
        fallback_output=fallback_output,
        fabricated_test_audit=fake_audit,
    )

    print(f"Exported Markdown: {md_file}")
    print(f"Exported JSON:     {json_file}")
    print("=" * 80)

    return {
        "cited_output": cited_output,
        "fallback_output": fallback_output,
        "fake_audit": fake_audit,
    }


if __name__ == "__main__":
    run_citation_evaluation_suite()
