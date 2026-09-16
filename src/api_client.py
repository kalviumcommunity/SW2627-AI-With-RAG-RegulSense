"""Client library for communicating with the RegulSense RAG Backend API.

Provides structured methods for:
- Health checking (/api/v1/health)
- Compliance queries (/api/v1/query)
- Runtime document uploads (/api/v1/upload)
- Configuration introspection (/api/v1/config)
- Graceful connection error, timeout, and HTTP validation error handling.
- Direct-mode fallback via FastAPI TestClient when standalone server process is offline.
"""

from dataclasses import dataclass, field
import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

logger = logging.getLogger("RegulSenseClient")


@dataclass
class ClientResponse:
    """Standardized wrapper for all client API responses."""
    success: bool
    status_code: int
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    latency_seconds: float = 0.0

    @property
    def is_refusal(self) -> bool:
        """Returns True if the backend returned a safe guardrail refusal."""
        return self.data.get("status") == "refusal"

    @property
    def answer(self) -> str:
        """Returns answer string if available."""
        return self.data.get("answer", "")

    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Returns list of source items if available."""
        return self.data.get("sources", [])

    @property
    def citations(self) -> List[str]:
        """Returns citation markers cited in answer."""
        return self.data.get("citations", [])


class RegulSenseAPIClient:
    """Client for interacting with the RegulSense Regulatory Compliance RAG Backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_timeout: float = 60.0,
        enable_direct_fallback: bool = True,
    ):
        """Initializes API client.

        Args:
            base_url: Base URL of backend API (e.g. http://localhost:8000).
            default_timeout: Default request timeout in seconds.
            enable_direct_fallback: Whether to fall back to in-process FastAPI TestClient
                                   if HTTP server connection fails.
        """
        env_url = os.getenv("REGULSENSE_API_URL", "http://localhost:8000").rstrip("/")
        self.base_url = (base_url or env_url).rstrip("/")
        self.default_timeout = default_timeout
        self.enable_direct_fallback = enable_direct_fallback
        self._direct_client = None

    def _get_direct_client(self):
        """Lazily initializes in-process TestClient fallback."""
        if self._direct_client is None:
            try:
                from fastapi.testclient import TestClient
                from src.api import app
                self._direct_client = TestClient(app)
                logger.info("Initialized direct in-process TestClient fallback.")
            except Exception as exc:
                logger.warning("Could not initialize direct TestClient fallback: %s", exc)
                return None
        return self._direct_client

    def check_health(self, timeout: float = 3.0, use_direct: bool = False) -> ClientResponse:
        """Checks API health and readiness status."""
        start_time = time.time()
        url = f"{self.base_url}/api/v1/health"

        if use_direct:
            client = self._get_direct_client()
            if client:
                try:
                    resp = client.get("/api/v1/health")
                    elapsed = time.time() - start_time
                    return ClientResponse(
                        success=resp.status_code == 200,
                        status_code=resp.status_code,
                        data=resp.json(),
                        latency_seconds=round(elapsed, 4),
                    )
                except Exception as exc:
                    return ClientResponse(
                        success=False,
                        status_code=500,
                        error_code="DIRECT_CLIENT_ERROR",
                        error_message=str(exc),
                        latency_seconds=round(time.time() - start_time, 4),
                    )

        try:
            resp = requests.get(url, timeout=timeout)
            elapsed = time.time() - start_time
            if resp.status_code == 200:
                return ClientResponse(
                    success=True,
                    status_code=200,
                    data=resp.json(),
                    latency_seconds=round(elapsed, 4),
                )
            else:
                return ClientResponse(
                    success=False,
                    status_code=resp.status_code,
                    data=resp.json() if "application/json" in resp.headers.get("Content-Type", "") else {},
                    error_code=f"HTTP_{resp.status_code}",
                    error_message=f"Health check failed with HTTP {resp.status_code}",
                    latency_seconds=round(elapsed, 4),
                )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            elapsed = time.time() - start_time
            if self.enable_direct_fallback:
                logger.info("Backend HTTP unreachable, falling back to direct mode for health check.")
                return self.check_health(timeout=timeout, use_direct=True)

            return ClientResponse(
                success=False,
                status_code=503,
                error_code="BACKEND_OFFLINE",
                error_message=f"Backend service unreachable at {self.base_url}. Ensure uvicorn is running.",
                latency_seconds=round(elapsed, 4),
            )
        except Exception as exc:
            return ClientResponse(
                success=False,
                status_code=500,
                error_code="CLIENT_ERROR",
                error_message=str(exc),
                latency_seconds=round(time.time() - start_time, 4),
            )

    def submit_query(
        self,
        question: str,
        top_k: int = 3,
        include_metadata: bool = True,
        timeout: Optional[float] = None,
        use_direct: bool = False,
    ) -> ClientResponse:
        """Submits a regulatory question to the /api/v1/query endpoint."""
        start_time = time.time()
        eff_timeout = timeout or self.default_timeout

        # Client-side input validation
        cleaned_q = question.strip() if question else ""
        if not cleaned_q:
            return ClientResponse(
                success=False,
                status_code=400,
                error_code="EMPTY_QUESTION",
                error_message="Question cannot be empty or whitespace only.",
                latency_seconds=0.0,
            )
        if len(cleaned_q) < 3:
            return ClientResponse(
                success=False,
                status_code=400,
                error_code="QUESTION_TOO_SHORT",
                error_message="Question must be at least 3 characters long.",
                latency_seconds=0.0,
            )

        payload = {
            "question": cleaned_q,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }

        if use_direct:
            client = self._get_direct_client()
            if client:
                try:
                    resp = client.post("/api/v1/query", json=payload)
                    elapsed = time.time() - start_time
                    try:
                        data = resp.json()
                    except Exception:
                        data = {"raw_text": resp.text}

                    if resp.status_code == 200:
                        return ClientResponse(
                            success=True,
                            status_code=200,
                            data=data,
                            latency_seconds=round(elapsed, 4),
                        )
                    else:
                        return ClientResponse(
                            success=False,
                            status_code=resp.status_code,
                            data=data,
                            error_code=data.get("error_code", f"HTTP_{resp.status_code}"),
                            error_message=data.get("message", f"Request failed with HTTP {resp.status_code}"),
                            latency_seconds=round(elapsed, 4),
                        )
                except Exception as exc:
                    return ClientResponse(
                        success=False,
                        status_code=500,
                        error_code="DIRECT_EXECUTION_ERROR",
                        error_message=str(exc),
                        latency_seconds=round(time.time() - start_time, 4),
                    )

        url = f"{self.base_url}/api/v1/query"
        try:
            resp = requests.post(url, json=payload, timeout=eff_timeout)
            elapsed = time.time() - start_time
            try:
                data = resp.json()
            except Exception:
                data = {"raw_text": resp.text}

            if resp.status_code == 200:
                return ClientResponse(
                    success=True,
                    status_code=200,
                    data=data,
                    latency_seconds=round(elapsed, 4),
                )
            else:
                err_code = data.get("error_code", f"HTTP_{resp.status_code}")
                err_msg = data.get("message", f"Query failed with status {resp.status_code}")
                return ClientResponse(
                    success=False,
                    status_code=resp.status_code,
                    data=data,
                    error_code=err_code,
                    error_message=err_msg,
                    latency_seconds=round(elapsed, 4),
                )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            elapsed = time.time() - start_time
            if self.enable_direct_fallback:
                logger.info("Backend HTTP unreachable, falling back to direct mode for query.")
                return self.submit_query(
                    question=question,
                    top_k=top_k,
                    include_metadata=include_metadata,
                    timeout=timeout,
                    use_direct=True,
                )

            return ClientResponse(
                success=False,
                status_code=503,
                error_code="BACKEND_OFFLINE",
                error_message=f"Backend service unreachable at {self.base_url}. Start backend with 'uvicorn src.api:app --port 8000'.",
                latency_seconds=round(elapsed, 4),
            )
        except Exception as exc:
            return ClientResponse(
                success=False,
                status_code=500,
                error_code="CLIENT_ERROR",
                error_message=str(exc),
                latency_seconds=round(time.time() - start_time, 4),
            )

    def submit_query_stream(
        self,
        question: str,
        top_k: int = 3,
        include_metadata: bool = True,
        timeout: Optional[float] = None,
        use_direct: bool = False,
    ):
        """Streams a compliance inquiry via Server-Sent Events (SSE) or in-process generator.

        Yields dictionaries with 'type':
        - {"type": "sources", "sources": [...]}
        - {"type": "token", "token": "..."}
        - {"type": "done", "status": "success"|"refusal", "answer": str, "citations": [...], "metadata": {...}}
        - {"type": "error", "error_code": "...", "message": "..."}
        """
        eff_timeout = timeout or self.default_timeout

        cleaned_q = question.strip() if question else ""
        if not cleaned_q:
            yield {
                "type": "error",
                "error_code": "EMPTY_QUESTION",
                "message": "Question cannot be empty or whitespace only.",
            }
            return
        if len(cleaned_q) < 3:
            yield {
                "type": "error",
                "error_code": "QUESTION_TOO_SHORT",
                "message": "Question must be at least 3 characters long.",
            }
            return

        payload = {
            "question": cleaned_q,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }

        if use_direct:
            try:
                from src.api import app
                stream_gen = app.state.guardrail.execute_stream(query=cleaned_q, top_k=top_k)

                for item in stream_gen:
                    event_type = item[0]
                    if event_type == "sources":
                        _, registry_serialized, candidate_chunks = item
                        sources_list: List[Dict[str, Any]] = []
                        if registry_serialized:
                            for marker, meta in registry_serialized.items():
                                sources_list.append({
                                    "marker": marker,
                                    "source_document": meta.get("source_document", "Unknown"),
                                    "chunk_id": meta.get("chunk_id", "None"),
                                    "section": meta.get("section", ""),
                                    "page_number": meta.get("page_number", 1),
                                    "similarity_score": meta.get("similarity_score", 0.0),
                                    "verbatim_text": meta.get("verbatim_text", ""),
                                })
                        elif candidate_chunks:
                            for idx, c in enumerate(candidate_chunks, start=1):
                                doc = getattr(c, "metadata", {}).get("source_document", "Unknown") if hasattr(c, "metadata") else "Unknown"
                                cid = getattr(c, "id", None) or getattr(c, "chunk_id", f"chunk_{idx}")
                                sec = getattr(c, "metadata", {}).get("section", "") if hasattr(c, "metadata") else ""
                                txt = getattr(c, "document", "") or getattr(c, "verbatim_text", "")
                                score = float(getattr(c, "similarity_score", 0.0))
                                sources_list.append({
                                    "marker": f"[{idx}]",
                                    "source_document": doc,
                                    "chunk_id": cid,
                                    "section": sec,
                                    "page_number": 1,
                                    "similarity_score": score,
                                    "verbatim_text": txt,
                                })
                        yield {"type": "sources", "sources": sources_list}
                    elif event_type == "token":
                        _, delta = item
                        yield {"type": "token", "token": delta}
                    elif event_type == "done":
                        _, guard_res = item
                        yield {
                            "type": "done",
                            "status": "refusal" if guard_res.is_refusal else "success",
                            "answer": guard_res.answer,
                            "citations": guard_res.citations,
                            "is_refusal": guard_res.is_refusal,
                            "metadata": {
                                "latency_seconds": guard_res.latency_seconds,
                                "model": guard_res.model,
                                "top_k": top_k,
                                "guardrail_status": guard_res.quality_assessment.status if guard_res.quality_assessment else "UNKNOWN",
                                "top_similarity_score": round(guard_res.quality_assessment.top_score, 4) if guard_res.quality_assessment else 0.0,
                            },
                        }
            except Exception as exc:
                logger.error("Direct streaming error: %s", exc, exc_info=True)
                yield {
                    "type": "error",
                    "error_code": "DIRECT_STREAM_ERROR",
                    "message": str(exc),
                }
            return

        url = f"{self.base_url}/api/v1/query/stream"
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=eff_timeout)
            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_code = err_data.get("error_code", f"HTTP_{resp.status_code}")
                    err_msg = err_data.get("message", f"Streaming request failed with HTTP {resp.status_code}")
                except Exception:
                    err_code = f"HTTP_{resp.status_code}"
                    err_msg = f"Streaming request failed with status {resp.status_code}: {resp.text[:200]}"
                yield {"type": "error", "error_code": err_code, "message": err_msg}
                return

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data:"):
                    json_str = line[5:].strip()
                    if not json_str:
                        continue
                    try:
                        parsed = json.loads(json_str)
                        evt = parsed.get("event")
                        evt_data = parsed.get("data", {})
                        if evt == "sources":
                            yield {"type": "sources", "sources": evt_data.get("sources", [])}
                        elif evt == "token":
                            yield {"type": "token", "token": evt_data.get("token", "")}
                        elif evt == "done":
                            yield {"type": "done", **evt_data}
                        elif evt == "error":
                            yield {"type": "error", **evt_data}
                    except json.JSONDecodeError:
                        continue

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if self.enable_direct_fallback:
                logger.info("Backend HTTP unreachable for stream, falling back to direct mode.")
                yield from self.submit_query_stream(
                    question=question,
                    top_k=top_k,
                    include_metadata=include_metadata,
                    timeout=timeout,
                    use_direct=True,
                )
                return

            yield {
                "type": "error",
                "error_code": "BACKEND_OFFLINE",
                "message": f"Backend service unreachable at {self.base_url}. Start backend with 'uvicorn src.api:app --port 8000'.",
            }
        except Exception as exc:
            logger.error("Client error during streaming query: %s", exc, exc_info=True)
            yield {
                "type": "error",
                "error_code": "STREAM_CLIENT_ERROR",
                "message": str(exc),
            }

    def upload_document(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        timeout: Optional[float] = None,
        use_direct: bool = False,
    ) -> ClientResponse:
        """Uploads and indexes a regulatory document via /api/v1/upload."""
        start_time = time.time()
        eff_timeout = timeout or (self.default_timeout * 2)

        data_fields: Dict[str, Any] = {}
        if chunk_size is not None:
            data_fields["chunk_size"] = str(chunk_size)
        if chunk_overlap is not None:
            data_fields["chunk_overlap"] = str(chunk_overlap)

        files = {"file": (filename, io.BytesIO(file_bytes))}

        if use_direct:
            client = self._get_direct_client()
            if client:
                try:
                    resp = client.post("/api/v1/upload", files={"file": (filename, file_bytes)}, data=data_fields)
                    elapsed = time.time() - start_time
                    try:
                        res_json = resp.json()
                    except Exception:
                        res_json = {"raw_text": resp.text}
                    return ClientResponse(
                        success=resp.status_code == 200,
                        status_code=resp.status_code,
                        data=res_json,
                        error_code=res_json.get("error_code") if resp.status_code != 200 else None,
                        error_message=res_json.get("message") if resp.status_code != 200 else None,
                        latency_seconds=round(elapsed, 4),
                    )
                except Exception as exc:
                    return ClientResponse(
                        success=False,
                        status_code=500,
                        error_code="DIRECT_UPLOAD_ERROR",
                        error_message=str(exc),
                        latency_seconds=round(time.time() - start_time, 4),
                    )

        url = f"{self.base_url}/api/v1/upload"
        try:
            resp = requests.post(url, files=files, data=data_fields, timeout=eff_timeout)
            elapsed = time.time() - start_time
            try:
                res_json = resp.json()
            except Exception:
                res_json = {"raw_text": resp.text}

            return ClientResponse(
                success=resp.status_code == 200,
                status_code=resp.status_code,
                data=res_json,
                error_code=res_json.get("error_code") if resp.status_code != 200 else None,
                error_message=res_json.get("message") if resp.status_code != 200 else None,
                latency_seconds=round(elapsed, 4),
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if self.enable_direct_fallback:
                return self.upload_document(
                    file_bytes=file_bytes,
                    filename=filename,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    timeout=timeout,
                    use_direct=True,
                )
            return ClientResponse(
                success=False,
                status_code=503,
                error_code="BACKEND_OFFLINE",
                error_message=f"Backend service unreachable at {self.base_url}.",
                latency_seconds=round(time.time() - start_time, 4),
            )
        except Exception as exc:
            return ClientResponse(
                success=False,
                status_code=500,
                error_code="CLIENT_ERROR",
                error_message=str(exc),
                latency_seconds=round(time.time() - start_time, 4),
            )

    def get_usage_summary(self, timeout: Optional[float] = None, use_direct: bool = False) -> ClientResponse:
        """Retrieves aggregated usage metrics, cache hit rate, token counts, and cost economics (Task 4)."""
        start_time = time.time()
        eff_timeout = timeout or self.default_timeout

        if use_direct:
            client = self._get_direct_client()
            if client:
                try:
                    resp = client.get("/api/v1/analytics/usage")
                    elapsed = time.time() - start_time
                    return ClientResponse(
                        success=resp.status_code == 200,
                        status_code=resp.status_code,
                        data=resp.json(),
                        latency_seconds=round(elapsed, 4),
                    )
                except Exception as exc:
                    return ClientResponse(
                        success=False,
                        status_code=500,
                        error_code="DIRECT_ANALYTICS_ERROR",
                        error_message=str(exc),
                        latency_seconds=round(time.time() - start_time, 4),
                    )

        url = f"{self.base_url}/api/v1/analytics/usage"
        try:
            resp = requests.get(url, timeout=eff_timeout)
            elapsed = time.time() - start_time
            return ClientResponse(
                success=resp.status_code == 200,
                status_code=resp.status_code,
                data=resp.json() if resp.status_code == 200 else {},
                error_code=resp.json().get("error_code") if resp.status_code != 200 else None,
                error_message=resp.json().get("message") if resp.status_code != 200 else None,
                latency_seconds=round(elapsed, 4),
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if self.enable_direct_fallback:
                return self.get_usage_summary(timeout=timeout, use_direct=True)
            return ClientResponse(
                success=False,
                status_code=503,
                error_code="BACKEND_OFFLINE",
                error_message=f"Backend service unreachable at {self.base_url}.",
                latency_seconds=round(time.time() - start_time, 4),
            )
        except Exception as exc:
            return ClientResponse(
                success=False,
                status_code=500,
                error_code="CLIENT_ERROR",
                error_message=str(exc),
                latency_seconds=round(time.time() - start_time, 4),
            )

    def get_cache_stats(self, timeout: Optional[float] = None, use_direct: bool = False) -> ClientResponse:
        """Retrieves in-memory query cache statistics and capacity (Task 1)."""
        start_time = time.time()
        eff_timeout = timeout or self.default_timeout

        if use_direct:
            client = self._get_direct_client()
            if client:
                try:
                    resp = client.get("/api/v1/analytics/cache")
                    elapsed = time.time() - start_time
                    return ClientResponse(
                        success=resp.status_code == 200,
                        status_code=resp.status_code,
                        data=resp.json(),
                        latency_seconds=round(elapsed, 4),
                    )
                except Exception as exc:
                    return ClientResponse(
                        success=False,
                        status_code=500,
                        error_code="DIRECT_CACHE_ERROR",
                        error_message=str(exc),
                        latency_seconds=round(time.time() - start_time, 4),
                    )

        url = f"{self.base_url}/api/v1/analytics/cache"
        try:
            resp = requests.get(url, timeout=eff_timeout)
            elapsed = time.time() - start_time
            return ClientResponse(
                success=resp.status_code == 200,
                status_code=resp.status_code,
                data=resp.json() if resp.status_code == 200 else {},
                latency_seconds=round(elapsed, 4),
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if self.enable_direct_fallback:
                return self.get_cache_stats(timeout=timeout, use_direct=True)
            return ClientResponse(
                success=False,
                status_code=503,
                error_code="BACKEND_OFFLINE",
                error_message=f"Backend service unreachable at {self.base_url}.",
                latency_seconds=round(time.time() - start_time, 4),
            )
        except Exception as exc:
            return ClientResponse(
                success=False,
                status_code=500,
                error_code="CLIENT_ERROR",
                error_message=str(exc),
                latency_seconds=round(time.time() - start_time, 4),
            )

    def clear_cache(self, timeout: Optional[float] = None, use_direct: bool = False) -> ClientResponse:
        """Clears all cached query entries on the backend (Task 1)."""
        start_time = time.time()
        eff_timeout = timeout or self.default_timeout

        if use_direct:
            client = self._get_direct_client()
            if client:
                try:
                    resp = client.post("/api/v1/analytics/cache/clear")
                    elapsed = time.time() - start_time
                    return ClientResponse(
                        success=resp.status_code == 200,
                        status_code=resp.status_code,
                        data=resp.json(),
                        latency_seconds=round(elapsed, 4),
                    )
                except Exception as exc:
                    return ClientResponse(
                        success=False,
                        status_code=500,
                        error_code="DIRECT_CLEAR_ERROR",
                        error_message=str(exc),
                        latency_seconds=round(time.time() - start_time, 4),
                    )

        url = f"{self.base_url}/api/v1/analytics/cache/clear"
        try:
            resp = requests.post(url, timeout=eff_timeout)
            elapsed = time.time() - start_time
            return ClientResponse(
                success=resp.status_code == 200,
                status_code=resp.status_code,
                data=resp.json() if resp.status_code == 200 else {},
                latency_seconds=round(elapsed, 4),
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if self.enable_direct_fallback:
                return self.clear_cache(timeout=timeout, use_direct=True)
            return ClientResponse(
                success=False,
                status_code=503,
                error_code="BACKEND_OFFLINE",
                error_message=f"Backend service unreachable at {self.base_url}.",
                latency_seconds=round(time.time() - start_time, 4),
            )
        except Exception as exc:
            return ClientResponse(
                success=False,
                status_code=500,
                error_code="CLIENT_ERROR",
                error_message=str(exc),
                latency_seconds=round(time.time() - start_time, 4),
            )
