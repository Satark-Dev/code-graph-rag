from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from loguru import logger

from . import constants as cs
from . import logs as ls
from .config import settings
from .utils.dependencies import has_pgvector
from .utils.token_utils import count_tokens


class EmbeddingCache:
    __slots__ = ("_cache", "_path")

    def __init__(self, path: Path | None = None) -> None:
        self._cache: dict[str, list[float]] = {}
        self._path = path

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, content: str) -> list[float] | None:
        return self._cache.get(self._content_hash(content))

    def put(self, content: str, embedding: list[float]) -> None:
        self._cache[self._content_hash(content)] = embedding

    def get_many(self, snippets: list[str]) -> dict[int, list[float]]:
        results: dict[int, list[float]] = {}
        for i, snippet in enumerate(snippets):
            if (cached := self.get(snippet)) is not None:
                results[i] = cached
        return results

    def put_many(self, snippets: list[str], embeddings: list[list[float]]) -> None:
        for snippet, embedding in zip(snippets, embeddings):
            self.put(snippet, embedding)

    def save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(ls.EMBEDDING_CACHE_SAVE_FAILED, path=self._path, error=e)

    def load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.debug(
                ls.EMBEDDING_CACHE_LOADED, count=len(self._cache), path=self._path
            )
        except Exception as e:
            logger.warning(ls.EMBEDDING_CACHE_LOAD_FAILED, path=self._path, error=e)
            self._cache = {}

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


_embedding_cache_by_base: dict[Path, EmbeddingCache] = {}


def get_embedding_cache(repo_path: Path | None = None) -> EmbeddingCache:
    # Use the provided repo_path, or fall back to current directory for CLI/local use.
    # Deprecated settings.TARGET_REPO_PATH is no longer used.
    base_path = (repo_path or Path.cwd()).resolve()
    if base_path not in _embedding_cache_by_base:
        cache_path = base_path / cs.EMBEDDING_CACHE_FILENAME
        cache = EmbeddingCache(path=cache_path)
        cache.load()
        _embedding_cache_by_base[base_path] = cache
    return _embedding_cache_by_base[base_path]


def clear_embedding_cache(repo_path: Path | None = None) -> None:
    """
    Clears the in-memory embedding cache.

    If repo_path is provided, only clears that repo's cache; otherwise clears all caches.
    """
    if repo_path is None:
        _embedding_cache_by_base.clear()
        return
    base_path = repo_path.resolve()
    _embedding_cache_by_base.pop(base_path, None)


def _truncate(s: str, max_length: int | None) -> str:
    if max_length is None:
        return s
    ml = int(max_length)
    if ml <= 0:
        return s
    return s[:ml]


def _require_semantic_enabled() -> None:
    if not has_pgvector():
        raise RuntimeError(
            "Semantic search requires 'semantic' extra: uv sync --extra semantic"
        )


def _resolve_openai_embeddings_client_config() -> tuple[str, str]:
    """
    Resolve API key + base_url for embeddings requests.

    Precedence:
    - api_key: EMBEDDINGS_API_KEY, then ORCHESTRATOR_API_KEY, then CYPHER_API_KEY
    - base_url: EMBEDDINGS_ENDPOINT, then ORCHESTRATOR_ENDPOINT, then OPENAI_DEFAULT_ENDPOINT
    """
    api_key = (
        (settings.EMBEDDINGS_API_KEY or "").strip()
        or (settings.ORCHESTRATOR_API_KEY or "").strip()
        or (settings.CYPHER_API_KEY or "").strip()
    )
    if not api_key:
        raise ValueError(
            "Missing API key for embeddings. Set EMBEDDINGS_API_KEY (preferred) "
            "or ORCHESTRATOR_API_KEY (fallback) in .env."
        )

    base_url = (
        (settings.EMBEDDINGS_ENDPOINT or "").strip()
        or (settings.ORCHESTRATOR_ENDPOINT or "").strip()
        or cs.OPENAI_DEFAULT_ENDPOINT
    )
    return api_key, base_url


def _openai_client():
    from openai import OpenAI

    api_key, base_url = _resolve_openai_embeddings_client_config()
    return OpenAI(api_key=api_key, base_url=base_url)


def _best_effort_log_embedding_usage(
    *,
    tool_call_id: str,
    input_texts: list[str],
) -> None:
    try:
        from .observability.hook import observability_hook

        input_tokens = sum(count_tokens(t) for t in input_texts)
        coro = observability_hook.log_tool_usage(
            tool_name="embeddings",
            tool_call_id=tool_call_id,
            model_name=settings.EMBEDDINGS_MODEL,
            input_tokens=int(input_tokens),
            output_tokens=0,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # We are running inside a thread from asyncio.to_thread().
            # Running asyncio.run(coro) here is unsafe because the underlying
            # Kafka producer is bound to the main event loop.
            from .services.kafka.producer import kafka_service

            k_loop = getattr(kafka_service, "_loop", None)
            if k_loop is not None and k_loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, k_loop)
            else:
                coro.close()
    except (ImportError, ValueError) as e:
        logger.debug("Observability embeddings usage log skipped: {}", e)
    except Exception as e:
        logger.debug("Observability embeddings usage log failed: {}", e)


def embed_code(
    code: str,
    max_length: int | None = None,
    *,
    repo_path: Path | None = None,
    cache: EmbeddingCache | None = None,
) -> list[float]:
    _require_semantic_enabled()

    cache = cache or get_embedding_cache(repo_path=repo_path)
    if (cached := cache.get(code)) is not None:
        return cached

    client = _openai_client()
    snippet = _truncate(code, max_length or settings.EMBEDDING_MAX_LENGTH)
    resp = client.embeddings.create(model=settings.EMBEDDINGS_MODEL, input=snippet)
    embedding = list(resp.data[0].embedding)
    cache.put(code, embedding)

    _best_effort_log_embedding_usage(
        tool_call_id=f"emb_{uuid4().hex}",
        input_texts=[snippet],
    )
    return embedding


def embed_code_batch(
    snippets: list[str],
    max_length: int | None = None,
    batch_size: int = cs.EMBEDDING_DEFAULT_BATCH_SIZE,
    *,
    repo_path: Path | None = None,
    cache: EmbeddingCache | None = None,
) -> list[list[float]]:
    _require_semantic_enabled()

    if not snippets:
        return []

    cache = cache or get_embedding_cache(repo_path=repo_path)
    cached_results = cache.get_many(snippets)
    if len(cached_results) == len(snippets):
        logger.debug(ls.EMBEDDING_CACHE_HIT, count=len(snippets))
        return [cached_results[i] for i in range(len(snippets))]

    ml = max_length or settings.EMBEDDING_MAX_LENGTH
    uncached_indices = [i for i in range(len(snippets)) if i not in cached_results]
    uncached_snippets = [_truncate(snippets[i], ml) for i in uncached_indices]

    client = _openai_client()
    all_new_embeddings: list[list[float]] = []
    for start in range(0, len(uncached_snippets), int(batch_size)):
        batch = uncached_snippets[start : start + int(batch_size)]
        resp = client.embeddings.create(model=settings.EMBEDDINGS_MODEL, input=batch)
        # API returns data in the same order as inputs
        all_new_embeddings.extend([list(d.embedding) for d in resp.data])

    cache.put_many([snippets[i] for i in uncached_indices], all_new_embeddings)
    _best_effort_log_embedding_usage(
        tool_call_id=f"emb_batch_{uuid4().hex}",
        input_texts=uncached_snippets,
    )

    results: list[list[float]] = [[] for _ in snippets]
    for i, emb in cached_results.items():
        results[i] = emb
    for idx, orig_i in enumerate(uncached_indices):
        results[orig_i] = all_new_embeddings[idx]
    return results


def prewarm_embeddings() -> None:
    # No local model to prewarm; validate config early.
    _ = _openai_client()
