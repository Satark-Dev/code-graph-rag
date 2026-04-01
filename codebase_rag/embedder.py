from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loguru import logger

from . import constants as cs
from . import logs as ls
from .config import settings
from .utils.dependencies import has_pgvector


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


_embedding_cache: EmbeddingCache | None = None


def get_embedding_cache(repo_path: Path | None = None) -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        # (H) Use the provided repo_path, or fall back to current directory for CLI/local use.
        # Deprecated settings.TARGET_REPO_PATH is no longer used.
        base_path = repo_path or Path.cwd()
        cache_path = base_path / cs.EMBEDDING_CACHE_FILENAME
        _embedding_cache = EmbeddingCache(path=cache_path)
        _embedding_cache.load()
    return _embedding_cache


def clear_embedding_cache() -> None:
    global _embedding_cache
    if _embedding_cache is not None:
        _embedding_cache.clear()
        _embedding_cache = None


def _truncate(s: str, max_length: int | None) -> str:
    if max_length is None:
        return s
    ml = int(max_length)
    if ml <= 0:
        return s
    return s[:ml]


def _openai_client():
    from openai import OpenAI

    api_key = settings.EMBEDDINGS_API_KEY or settings.ORCHESTRATOR_API_KEY or settings.CYPHER_API_KEY
    if not api_key:
        raise ValueError(
            "Missing OpenAI API key for embeddings. Set EMBEDDINGS_API_KEY or ORCHESTRATOR_API_KEY in .env."
        )
    base_url = settings.EMBEDDINGS_ENDPOINT or settings.ORCHESTRATOR_ENDPOINT or cs.OPENAI_DEFAULT_ENDPOINT
    return OpenAI(api_key=api_key, base_url=base_url)


def embed_code(code: str, max_length: int | None = None) -> list[float]:
    if not has_pgvector():
        raise RuntimeError(
            "Semantic search requires 'semantic' extra: uv sync --extra semantic"
        )

    cache = get_embedding_cache()
    if (cached := cache.get(code)) is not None:
        return cached

    client = _openai_client()
    snippet = _truncate(code, max_length or settings.EMBEDDING_MAX_LENGTH)
    resp = client.embeddings.create(model=settings.EMBEDDINGS_MODEL, input=snippet)
    embedding = list(resp.data[0].embedding)
    cache.put(code, embedding)
    return embedding


def embed_code_batch(
    snippets: list[str],
    max_length: int | None = None,
    batch_size: int = cs.EMBEDDING_DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    if not has_pgvector():
        raise RuntimeError(
            "Semantic search requires 'semantic' extra: uv sync --extra semantic"
        )

    if not snippets:
        return []

    cache = get_embedding_cache()
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

    results: list[list[float]] = [[] for _ in snippets]
    for i, emb in cached_results.items():
        results[i] = emb
    for idx, orig_i in enumerate(uncached_indices):
        results[orig_i] = all_new_embeddings[idx]
    return results


def prewarm_embeddings() -> None:
    # No local model to prewarm; validate config early.
    _ = _openai_client()
