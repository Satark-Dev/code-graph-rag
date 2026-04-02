from __future__ import annotations

import pytest

from codebase_rag.services.index_guard import RepoNotIndexedError, assert_repo_indexed


class _FakeIngestor:
    def __init__(self, rows):
        self._rows = rows

    def fetch_all(self, _query: str, _params=None):
        return self._rows


@pytest.mark.anyio
async def test_assert_repo_indexed_raises_when_empty() -> None:
    ingestor = _FakeIngestor([{"count": 0}])
    with pytest.raises(RepoNotIndexedError) as e:
        await assert_repo_indexed(ingestor=ingestor, target_repo_path="/tmp/myrepo")
    assert e.value.project_name == "myrepo"


@pytest.mark.anyio
async def test_assert_repo_indexed_passes_when_present() -> None:
    ingestor = _FakeIngestor([{"count": 5}])
    await assert_repo_indexed(ingestor=ingestor, target_repo_path="/tmp/myrepo")

