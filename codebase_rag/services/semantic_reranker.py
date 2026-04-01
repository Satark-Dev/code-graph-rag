from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from uuid import uuid4

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from .. import logs as ls
from ..config import settings
from ..types_defs import SemanticSearchResult


class _DeepInfraRerankResponse(BaseModel):
    scores: list[float] = Field(default_factory=list)


class _DeepInfraRerankRequest(BaseModel):
    queries: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)


class _DeepInfraHeaders(BaseModel):
    authorization: str
    content_type: str = "application/json"

    def to_httpx_headers(self) -> dict[str, str]:
        return {
            "Authorization": self.authorization,
            "Content-Type": self.content_type,
        }


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
    request = _DeepInfraRerankRequest(
        queries=[query for _ in documents],
        documents=documents,
    )

    base = (settings.DEEPINFRA_BASE_URL or "").rstrip("/")
    if not base:
        return None
    url = f"{base}/{settings.DEEPINFRA_RERANK_MODEL}"
    headers = _DeepInfraHeaders(authorization=f"bearer {api_key}")

    try:
        timeout = httpx.Timeout(float(settings.DEEPINFRA_TIMEOUT_SECONDS))
        max_retries = max(0, int(settings.DEEPINFRA_MAX_RETRIES))
        async with httpx.AsyncClient(timeout=timeout) as client:
            scores = await _execute_rerank_with_retries(
                client=client,
                url=url,
                request=request,
                headers=headers,
                max_retries=max_retries,
                candidate_count=len(candidates),
            )
            try:
                # Best-effort: emit tool usage so pricing for Qwen reranker comes from model_pricing.
                from ..observability.hook import observability_hook
                from ..utils.token_utils import count_tokens

                input_tokens = count_tokens(query) + sum(count_tokens(d) for d in documents)
                await observability_hook.log_tool_usage(
                    tool_name="rerank",
                    tool_call_id=f"rerank_{uuid4().hex}",
                    model_name=str(settings.DEEPINFRA_RERANK_MODEL or "unknown"),
                    input_tokens=int(input_tokens),
                    output_tokens=0,
                )
            except Exception as e:
                logger.debug("Observability rerank usage log failed: {}", e)
            return scores
    except Exception as e:
        from ..utils.error_handling import log_and_fallback

        return log_and_fallback(
            label="Semantic rerank (provider=deepinfra)",
            error=e,
            default=None,
            level="warning",
            include_traceback=False,
        )


async def _execute_rerank_with_retries(
    *,
    client: httpx.AsyncClient,
    url: str,
    request: _DeepInfraRerankRequest,
    headers: _DeepInfraHeaders,
    max_retries: int,
    candidate_count: int,
) -> list[float] | None:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(
                url,
                json=request.model_dump(mode="json"),
                headers=headers.to_httpx_headers(),
            )
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
    # No-op: DeepInfra client is not cached globally anymore.
    return None


_RerankProvider = Callable[..., Awaitable[list[float] | None]]


def _get_provider_registry() -> dict[str, _RerankProvider]:
    # Keeping this as a function makes it easy to unit test/patch and avoids import-order surprises.
    return {
        "deepinfra": _deepinfra_rerank,
    }


async def rerank_semantic_results(
    *, query: str, candidates: list[SemanticSearchResult], top_k: int
) -> list[int] | None:
    """
    Reranker for semantic search candidates.

    Returns:
        Ordered list of node_ids (best-first).

        Returns an empty list for no-op cases (no candidates, top_k <= 0).
        Returns None when the configured provider is unavailable/unsupported or on failure.
    """
    if not candidates or top_k <= 0:
        return []

    try:
        provider = (settings.SEMANTIC_RERANK_PROVIDER or "deepinfra").lower()
        provider_fn = _get_provider_registry().get(provider)
        if provider_fn is None:
            return None
        scores = await provider_fn(query=query, candidates=candidates)

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
        from ..utils.error_handling import log_and_fallback

        return log_and_fallback(
            label=f"Semantic rerank (provider={provider!r})",
            error=e,
            default=None,
            level="warning",
            include_traceback=False,
        )

