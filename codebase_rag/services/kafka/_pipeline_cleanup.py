from __future__ import annotations

from loguru import logger

from ...utils.tool_call_store import fetch_latest_stage_status
from .repo_manager import RepoManager


def _stage_done(*, invocation_id: str, stage: str) -> bool:
    try:
        status = fetch_latest_stage_status(run_id=invocation_id, stage=stage)
        return status == "success"
    except Exception:  # noqa: BLE001
        return False


async def maybe_cleanup_repo_after_chat_pipeline(
    *,
    invocation_id: str,
    repo_path: str,
) -> bool:
    """
    Cleanup policy:
    - Only delete the local repo checkout when BOTH scoring and remediation stages exist
      for the invocation_id.
    - Uses repo in-use markers to avoid deleting while other invocations are running.
    """
    if not (_stage_done(invocation_id=invocation_id, stage="scoring") and _stage_done(invocation_id=invocation_id, stage="remediation")):
        return False

    mgr = RepoManager()
    deleted = await mgr.release_and_cleanup_if_unused(
        repo_path=repo_path,
        invocation_id=invocation_id,
    )
    if deleted:
        logger.info("Chat pipeline cleanup completed invocation_id={}", invocation_id)
    return deleted

