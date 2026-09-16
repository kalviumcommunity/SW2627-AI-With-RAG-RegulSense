"""Thread-safe LRU & TTL Query Cache for RegulSense RAG assistant.

Implements Task 1:
- Normalizes and hashes incoming user inquiries with relevant retrieval settings.
- Avoids redundant RAG retrieval, guardrail evaluations, and LLM inference calls.
- Preserves full verifiable citation mappings, chunk metadata, and token usage metrics.
- Provides thread-safe eviction, TTL expiration, and telemetry tracking.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("QueryCache")


@dataclass
class CachedQueryEntry:
    """Represents a cached RAG response with its citations, sources, and usage metadata."""
    cache_key: str
    normalized_query: str
    top_k: int
    answer: str
    citations: List[str]
    sources: List[Dict[str, Any]]
    status: str
    is_refusal: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    original_latency_seconds: float
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Checks whether the cache entry has exceeded its time-to-live."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serializes entry to dictionary for telemetry and inspection."""
        data = asdict(self)
        data["created_at_iso"] = datetime.fromtimestamp(self.created_at, timezone.utc).isoformat()
        data["is_expired"] = self.is_expired
        return data


class QueryCache:
    """Thread-safe, in-memory LRU cache with TTL expiration for RAG queries."""

    def __init__(
        self,
        max_size: int = 500,
        default_ttl_seconds: float = 3600.0,
        enable_metrics: bool = True,
    ):
        """Initializes the query cache.

        Args:
            max_size: Maximum number of cached query results before LRU eviction.
            default_ttl_seconds: Time-to-live in seconds for cached entries (default: 1 hour).
            enable_metrics: Whether to maintain hit/miss statistics.
        """
        self.max_size = max(1, max_size)
        self.default_ttl = default_ttl_seconds
        self.enable_metrics = enable_metrics

        self._cache: OrderedDict[str, CachedQueryEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Telemetry counters
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._evictions: int = 0
        self._expirations: int = 0
        self._total_cost_saved_usd: float = 0.0

        logger.info(
            "Initialized QueryCache (max_size=%d, default_ttl=%.1fs)",
            self.max_size,
            self.default_ttl,
        )

    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalizes user query text by trimming whitespace and normalizing case."""
        if not query:
            return ""
        # Strip outer whitespace and collapse internal whitespace
        normalized = re.sub(r"\s+", " ", query.strip())
        return normalized.lower()

    @classmethod
    def generate_cache_key(
        cls,
        query: str,
        top_k: int = 3,
        model: Optional[str] = None,
        filter_str: Optional[str] = None,
    ) -> str:
        """Generates a deterministic MD5 hash key from normalized query settings."""
        norm_q = cls.normalize_query(query)
        payload = {
            "q": norm_q,
            "k": top_k,
            "m": (model or "").lower().strip(),
            "f": (filter_str or "").strip(),
        }
        raw_repr = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw_repr.encode("utf-8")).hexdigest()[:24]

    def get(
        self,
        query: str,
        top_k: int = 3,
        model: Optional[str] = None,
        filter_str: Optional[str] = None,
    ) -> Optional[CachedQueryEntry]:
        """Retrieves a cached entry if present and not expired."""
        cache_key = self.generate_cache_key(query=query, top_k=top_k, model=model, filter_str=filter_str)

        with self._lock:
            self._total_requests += 1

            if cache_key not in self._cache:
                self._cache_misses += 1
                logger.debug("Cache MISS for key '%s' (query: '%s...')", cache_key, query[:40])
                return None

            entry = self._cache[cache_key]

            # Check TTL expiration
            if entry.is_expired:
                self._expirations += 1
                self._cache_misses += 1
                del self._cache[cache_key]
                logger.debug("Cache EXPIRED for key '%s' (TTL=%.1fs)", cache_key, entry.ttl_seconds)
                return None

            # Move to end for LRU policy
            self._cache.move_to_end(cache_key)
            entry.hit_count += 1
            self._cache_hits += 1
            self._total_cost_saved_usd += entry.estimated_cost_usd

            logger.info(
                "Cache HIT for key '%s' (hits=%d, saved=$%.5f, query='%s...')",
                cache_key,
                entry.hit_count,
                entry.estimated_cost_usd,
                query[:40],
            )
            return entry

    def put(
        self,
        query: str,
        top_k: int,
        answer: str,
        citations: List[str],
        sources: List[Dict[str, Any]],
        status: str = "success",
        is_refusal: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        original_latency_seconds: float = 0.0,
        model: Optional[str] = None,
        filter_str: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> CachedQueryEntry:
        """Stores a completed RAG response in the cache with LRU eviction."""
        norm_q = self.normalize_query(query)
        cache_key = self.generate_cache_key(query=query, top_k=top_k, model=model, filter_str=filter_str)
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        entry = CachedQueryEntry(
            cache_key=cache_key,
            normalized_query=norm_q,
            top_k=top_k,
            answer=answer,
            citations=citations or [],
            sources=sources or [],
            status=status,
            is_refusal=is_refusal,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=round(estimated_cost_usd, 6),
            original_latency_seconds=round(original_latency_seconds, 4),
            ttl_seconds=effective_ttl,
            metadata=extra_metadata or {},
        )

        with self._lock:
            # If replacing an existing key, delete it first to update position
            if cache_key in self._cache:
                del self._cache[cache_key]
            elif len(self._cache) >= self.max_size:
                # Evict oldest entry (first item in OrderedDict)
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug("LRU Evicted cache entry '%s' (hits=%d)", evicted_key, evicted_entry.hit_count)

            self._cache[cache_key] = entry
            logger.info("Cached query response '%s' (size=%d/%d)", cache_key, len(self._cache), self.max_size)

        return entry

    def invalidate(
        self,
        query: Optional[str] = None,
        top_k: int = 3,
        model: Optional[str] = None,
    ) -> bool:
        """Invalidates a specific cached query or clears all entries if query is None."""
        with self._lock:
            if query is None:
                count = len(self._cache)
                self._cache.clear()
                logger.info("Cleared entire query cache (%d entries removed)", count)
                return True

            cache_key = self.generate_cache_key(query=query, top_k=top_k, model=model)
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info("Invalidated cache entry for key '%s'", cache_key)
                return True
            return False

    def clear(self):
        """Clears all cached entries and resets size."""
        self.invalidate(query=None)

    def get_stats(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry on cache utilization and savings."""
        with self._lock:
            total = self._total_requests
            hits = self._cache_hits
            misses = self._cache_misses
            hit_rate = (hits / total * 100.0) if total > 0 else 0.0

            return {
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "total_requests": total,
                "hits": hits,
                "misses": misses,
                "hit_rate_pct": round(hit_rate, 2),
                "evictions": self._evictions,
                "expirations": self._expirations,
                "total_cost_saved_usd": round(self._total_cost_saved_usd, 6),
                "default_ttl_seconds": self.default_ttl,
            }
