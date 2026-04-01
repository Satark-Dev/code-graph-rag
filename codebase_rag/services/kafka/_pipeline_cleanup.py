from __future__ import annotations

from loguru import logger

from .repo_manager import RepoManager


async def release_repo_lease_if_unused(*, repo_path: str, repo_lease_id: str) -> bool:
    """
    Drop one pipeline lease on a shared clone. Deletes the checkout only when no leases remain.

    Each evidence job must use a distinct ``repo_lease_id``; remediation calls this after success
    so parallel findings under the same ``invocation_id`` do not delete the repo early.
    """
    mgr = RepoManager()
    deleted = await mgr.release_and_cleanup_if_unused(
        repo_path=repo_path,
        lease_id=repo_lease_id,
    )
    if deleted:
        logger.info(
            "Repo checkout removed after last pipeline lease released repo_path={}",
            repo_path,
        )
    return deleted
