from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

from ... import constants as cs
from ...config import load_cgrignore_patterns
from ...graph_updater import GraphUpdater
from ...parser_loader import load_parsers
from ...request_context import org_id_context
from .index_job_payload import IndexJobPayload


def _resolve_repo_path(repo_path: str) -> str:
    return os.path.abspath(repo_path)


async def process_index_job_message(
    *,
    payload: IndexJobPayload,
    ingestor: Any,
    key: str | None,
) -> bool:
    """
    Returns True if the message offset may be committed (success or poison skip).
    Returns False to leave the offset uncommitted (retry after rebalance/restart).
    """
    repo_path = _resolve_repo_path(payload.repo_path)
    if not os.path.isdir(repo_path):
        logger.warning("Index job repo_path not found; skipping: {}", repo_path)
        return True

    ctx_token = org_id_context.set(payload.org_id)
    try:
        from pathlib import Path

        repo = Path(repo_path).resolve()
        cgrignore = load_cgrignore_patterns(repo)
        cli_excludes = frozenset(payload.exclude or [])
        exclude_paths = cli_excludes | cgrignore.exclude or None
        unignore_paths = cgrignore.unignore or None

        if payload.clean:
            logger.info("Cleaning graph database and hash cache before index job...")
            ingestor.clean_database()
            cache_path = repo / cs.HASH_CACHE_FILENAME
            cache_path.unlink(missing_ok=True)

        ingestor.ensure_constraints()
        parsers, queries = load_parsers()

        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=repo,
            parsers=parsers,
            queries=queries,
            unignore_paths=unignore_paths,
            exclude_paths=exclude_paths,
            project_name=repo.name,
        )

        # Run synchronous GraphUpdater in a thread to avoid blocking the event loop.
        await asyncio.to_thread(updater.run)
        logger.info("Index job completed for {}", repo_path)
        return True
    except Exception:
        logger.exception("Index job failed for {}", repo_path)
        return False
    finally:
        org_id_context.reset(ctx_token)

