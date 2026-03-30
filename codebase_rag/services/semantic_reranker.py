from __future__ import annotations

import asyncio

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from .. import logs as ls
from ..config import settings
from ..types_defs import SemanticSearchResult

_DEEPINFRA_CLIENT: httpx.AsyncClient | None = None
_DEEPINFRA_CLIENT_LOCK = asyncio.Lock()


class _DeepInfraRerankResponse(BaseModel):
    scores: list[float] = Field(default_factory=list)


def _candidate_document(c: SemanticSearchResult) -> str:
    # Keep it short but information-dense for the reranker.
    return f"{c['qualified_name']} (type={c['type']}, name={c['name']})"


async def _deepinfra_rerank(
    *, query: str, candidates: list[SemanticSearchResult]
) -> list[float] | None:
    api_key = settings.DEEPINFRA_API_KEY
    if not api_key or not api_key.strip():
        return None

    # DeepInfra expects queries/documents arrays to be same length.
    documents = [_candidate_document(c) for c in candidates]
    payload = {
        "queries": [query for _ in documents],
        "documents": documents,
    }

    base = (settings.DEEPINFRA_BASE_URL or "").rstrip("/")
    if not base:
        return None
    url = f"{base}/{settings.DEEPINFRA_RERANK_MODEL}"
    headers = {
        "Authorization": f"bearer {api_key}",
        "Content-Type": "application/json",
    }

    async def _get_client(*, timeout: httpx.Timeout) -> httpx.AsyncClient:
        global _DEEPINFRA_CLIENT
        async with _DEEPINFRA_CLIENT_LOCK:
            if _DEEPINFRA_CLIENT is None or _DEEPINFRA_CLIENT.is_closed:
                _DEEPINFRA_CLIENT = httpx.AsyncClient(timeout=timeout)
            else:
                _DEEPINFRA_CLIENT.timeout = timeout
            return _DEEPINFRA_CLIENT

    try:
        timeout = httpx.Timeout(float(settings.DEEPINFRA_TIMEOUT_SECONDS))
        max_retries = max(0, int(settings.DEEPINFRA_MAX_RETRIES))
        client = await _get_client(timeout=timeout)

        scores = await _execute_rerank_with_retries(
            client=client,
            url=url,
            payload=payload,
            headers=headers,
            max_retries=max_retries,
            candidate_count=len(candidates),
        )
        return scores
    except Exception as e:
        logger.warning(ls.SEMANTIC_RERANK_FAILED.format(error=e))
        return None


async def _execute_rerank_with_retries(
    *,
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
    max_retries: int,
    candidate_count: int,
) -> list[float] | None:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                await asyncio.sleep(0.2 * (2**attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            parsed = _DeepInfraRerankResponse.model_validate(data)
            if len(parsed.scores) != candidate_count:
                return None
            return [float(s) for s in parsed.scores]
        except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(0.2 * (2**attempt))
                continue
            break
    if last_error:
        raise last_error
    return None


async def aclose_deepinfra_client() -> None:
    """
    Optional cleanup hook for app shutdown.
    """
    global _DEEPINFRA_CLIENT
    async with _DEEPINFRA_CLIENT_LOCK:
        if _DEEPINFRA_CLIENT is not None and not _DEEPINFRA_CLIENT.is_closed:
            await _DEEPINFRA_CLIENT.aclose()
        _DEEPINFRA_CLIENT = None


async def rerank_semantic_results(
    *, query: str, candidates: list[SemanticSearchResult], top_k: int
) -> list[int] | None:
    """
    Reranker for semantic search candidates.

    Returns:
        Ordered list of node_ids (best-first), or None on failure.
    """
    if not candidates or top_k <= 0:
        return []

    try:
        provider = (settings.SEMANTIC_RERANK_PROVIDER or "deepinfra").lower()
        scores: list[float] | None = None

        if provider == "deepinfra":
            scores = await _deepinfra_rerank(query=query, candidates=candidates)
        else:
            return None

        if not scores:
            return None

        scored = []
        for c, s in zip(candidates, scores, strict=False):
            nid = c.get("node_id")
            if nid is None:
                continue
            scored.append((int(nid), float(s)))

        if not scored:
            return None

        scored.sort(key=lambda t: t[1], reverse=True)
        return [nid for nid, _ in scored[:top_k]]
    except Exception as e:
        logger.warning(ls.SEMANTIC_RERANK_FAILED.format(error=e))
        return None

