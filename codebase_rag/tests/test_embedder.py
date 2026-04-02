from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.embedder import EmbeddingCache, clear_embedding_cache


@pytest.fixture(autouse=True)
def reset_cache() -> Generator[None, None, None]:
    clear_embedding_cache()
    yield
    clear_embedding_cache()

@pytest.fixture
def mock_openai_embeddings_client() -> MagicMock:
    client = MagicMock()
    # Emulate OpenAI embeddings response: resp.data[i].embedding
    emb = [0.0] * 1536
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=emb)]
    )
    return client


def test_embed_code_returns_1536_dimensional_vector(
    mock_openai_embeddings_client: MagicMock, monkeypatch
) -> None:
    monkeypatch.setattr("codebase_rag.embedder.has_pgvector", lambda: True)
    with patch("codebase_rag.embedder._openai_client", return_value=mock_openai_embeddings_client):
        from codebase_rag.embedder import embed_code

        result = embed_code("def hello(): pass")

    assert isinstance(result, list)
    assert len(result) == 1536


def test_embed_code_uses_cache(
    mock_openai_embeddings_client: MagicMock, monkeypatch
) -> None:
    monkeypatch.setattr("codebase_rag.embedder.has_pgvector", lambda: True)
    from codebase_rag.embedder import embed_code, get_embedding_cache

    cache = get_embedding_cache()
    cache.put("cached_code", [0.42] * 1536)

    with patch("codebase_rag.embedder._openai_client", return_value=mock_openai_embeddings_client):
        result = embed_code("cached_code")

    assert result == [0.42] * 1536
    mock_openai_embeddings_client.embeddings.create.assert_not_called()


def test_embed_code_batch_empty_list(monkeypatch) -> None:
    monkeypatch.setattr("codebase_rag.embedder.has_pgvector", lambda: True)
    from codebase_rag.embedder import embed_code_batch

    assert embed_code_batch([]) == []


def test_embed_code_batch_returns_correct_count(
    mock_openai_embeddings_client: MagicMock, monkeypatch
) -> None:
    monkeypatch.setattr("codebase_rag.embedder.has_pgvector", lambda: True)
    from codebase_rag.embedder import embed_code_batch

    emb = [0.0] * 1536
    mock_openai_embeddings_client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=emb), SimpleNamespace(embedding=emb), SimpleNamespace(embedding=emb)]
    )
    with patch("codebase_rag.embedder._openai_client", return_value=mock_openai_embeddings_client):
        results = embed_code_batch(["a", "b", "c"], batch_size=3)

    assert len(results) == 3
    assert all(len(r) == 1536 for r in results)


def test_embed_code_raises_without_pgvector(monkeypatch) -> None:
    monkeypatch.setattr("codebase_rag.embedder.has_pgvector", lambda: False)
    from codebase_rag.embedder import embed_code

    with pytest.raises(RuntimeError, match="Semantic search requires"):
        embed_code("x = 1")


# --- The rest of this file intentionally contains only OpenAI embedding tests. ---


def test_embedding_cache_put_and_get() -> None:
    cache = EmbeddingCache()
    embedding = [0.1, 0.2, 0.3]
    cache.put("def foo(): pass", embedding)
    assert cache.get("def foo(): pass") == embedding


def test_embedding_cache_miss_returns_none() -> None:
    cache = EmbeddingCache()
    assert cache.get("unknown code") is None


def test_embedding_cache_different_content_different_key() -> None:
    cache = EmbeddingCache()
    cache.put("code_a", [1.0])
    cache.put("code_b", [2.0])
    assert cache.get("code_a") == [1.0]
    assert cache.get("code_b") == [2.0]


def test_embedding_cache_overwrite() -> None:
    cache = EmbeddingCache()
    cache.put("code_a", [1.0])
    cache.put("code_a", [9.9])
    assert cache.get("code_a") == [9.9]


def test_embedding_cache_len() -> None:
    cache = EmbeddingCache()
    assert len(cache) == 0
    cache.put("a", [1.0])
    assert len(cache) == 1
    cache.put("b", [2.0])
    assert len(cache) == 2


def test_embedding_cache_clear() -> None:
    cache = EmbeddingCache()
    cache.put("a", [1.0])
    cache.put("b", [2.0])
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_embedding_cache_get_many() -> None:
    cache = EmbeddingCache()
    cache.put("a", [1.0])
    cache.put("b", [2.0])
    results = cache.get_many(["a", "c", "b"])
    assert results == {0: [1.0], 2: [2.0]}


def test_embedding_cache_put_many() -> None:
    cache = EmbeddingCache()
    cache.put_many(["x", "y"], [[1.0], [2.0]])
    assert cache.get("x") == [1.0]
    assert cache.get("y") == [2.0]


def test_embedding_cache_save_and_load() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.json"
        cache = EmbeddingCache(path=cache_path)
        cache.put("hello", [0.5, 0.6])
        cache.save()

        assert cache_path.exists()

        cache2 = EmbeddingCache(path=cache_path)
        cache2.load()
        assert cache2.get("hello") == [0.5, 0.6]


def test_embedding_cache_load_nonexistent_path() -> None:
    cache = EmbeddingCache(path=Path("/nonexistent/path/cache.json"))
    cache.load()
    assert len(cache) == 0


def test_embedding_cache_load_corrupt_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "corrupt.json"
        cache_path.write_text("not valid json data", encoding="utf-8")
        cache = EmbeddingCache(path=cache_path)
        cache.load()
        assert len(cache) == 0


def test_embedding_cache_save_no_path() -> None:
    cache = EmbeddingCache(path=None)
    cache.put("a", [1.0])
    cache.save()


def test_embedding_cache_load_no_path() -> None:
    cache = EmbeddingCache(path=None)
    cache.load()
    assert len(cache) == 0
