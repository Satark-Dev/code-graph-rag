from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class IngestorFetchAllProtocol(Protocol):
    def fetch_all(self, query: str, params: dict | None = None) -> list[dict] | list:
        ...


class RepoNotIndexedError(RuntimeError):
    def __init__(self, *, project_name: str):
        super().__init__(project_name)
        self.project_name = project_name


class RepoIndexQueryError(RuntimeError):
    pass


async def assert_repo_indexed(
    *, ingestor: IngestorFetchAllProtocol, target_repo_path: str
) -> None:
    """
    Prevent /api/chat from running until a repo is indexed in Memgraph.
    """
    project_name = Path(target_repo_path).name
    prefix = f"{project_name}."
    query = (
        "MATCH (n) "
        "WHERE n.qualified_name STARTS WITH $prefix "
        "RETURN count(n) AS count"
    )
    try:
        rows = await asyncio.to_thread(ingestor.fetch_all, query, {"prefix": prefix})
    except Exception as e:
        raise RepoIndexQueryError(str(e)) from e

    count = 0
    if isinstance(rows, list) and rows:
        first = rows[0]
        if isinstance(first, dict) and isinstance(first.get("count"), (int, float)):
            count = int(first["count"])

    if count <= 0:
        raise RepoNotIndexedError(project_name=project_name)

