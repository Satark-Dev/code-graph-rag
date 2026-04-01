from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from loguru import logger

from ...config import settings

class RepoManager:
    """
    Manages local repository clones for sharing between indexing and chat jobs.
    Tracks directories created during the instance's lifetime and ensures
    deterministic pathing to allow reuse.
    """

    def __init__(self) -> None:
        self._tracked_dirs: set[str] = set()
        self._lock = asyncio.Lock()

    def get_base_temp_dir(self) -> Path:
        base_dir = Path(settings.KAFKA_REPO_ROOT)
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def get_repo_path(self, org_id: str, branch: str) -> Path:
        """Derives a deterministic local path for a (org, branch) pair."""
        safe_branch = branch.replace("/", "_").replace("\\", "_")
        return self.get_base_temp_dir() / f"{org_id}_{safe_branch}"

    def _in_use_dir(self, repo_path: Path) -> Path:
        return repo_path / ".cgr_in_use"

    async def mark_in_use(self, *, repo_path: str, lease_id: str) -> None:
        """
        Mark a repo checkout as held by one concurrent pipeline (one file per lease).

        Use a **unique** ``lease_id`` per evidence job (e.g. UUID). Many jobs may share the
        same ``invocation_id`` while reusing one clone; a single shared marker would delete
        the repo as soon as the first pipeline finished.
        """
        lid = lease_id.strip()
        if not lid:
            raise ValueError("lease_id is required")
        p = Path(repo_path).resolve()
        async with self._lock:
            d = self._in_use_dir(p)
            d.mkdir(parents=True, exist_ok=True)
            (d / lid).write_text("1", encoding="utf-8")

    async def release_and_cleanup_if_unused(self, *, repo_path: str, lease_id: str) -> bool:
        """
        Remove this lease's marker. If no lease files remain under ``.cgr_in_use``, delete the repo.
        Returns True if the repo directory was removed.
        """
        lid = lease_id.strip()
        if not lid:
            raise ValueError("lease_id is required")
        p = Path(repo_path).resolve()
        async with self._lock:
            d = self._in_use_dir(p)
            marker = d / lid
            if marker.exists():
                marker.unlink(missing_ok=True)
            remaining: list[Path] = []
            if d.is_dir():
                remaining = [x for x in d.iterdir() if x.is_file()]
            if not remaining:
                try:
                    shutil.rmtree(p, ignore_errors=True)
                    logger.info("Cleaned up local repository checkout at {}", str(p))
                    return True
                except Exception as e:  # noqa: BLE001
                    logger.warning("Repo cleanup failed for {}: {}", str(p), e)
            return False

    def require_existing_local_clone(self, org_id: str, branch: str | None) -> str:
        """
        Return the deterministic local path only if a non-empty directory already exists.

        Chat jobs must use this (no git clone). Index jobs use ``ensure_cloned`` to populate
        the same path first.
        """
        if not branch or not str(branch).strip():
            raise ValueError(
                "Branch name is required; got empty/None. "
                "Ensure the finding has a branch asset with a valid name."
            )
        branch = branch.strip()
        target_path = self.get_repo_path(org_id, branch)
        target_str = str(target_path.resolve())
        if not target_path.is_dir():
            raise ValueError(
                f"Local repository clone not found at {target_str}. "
                "Run an index job for this org and branch first so the repository is cloned "
                "and indexed under KAFKA_REPO_ROOT."
            )
        if not any(target_path.iterdir()):
            raise ValueError(
                f"Local repository path {target_str} exists but is empty. "
                "Run an index job for this org and branch first."
            )
        return target_str

    async def ensure_cloned(self, repo_url: str, org_id: str, branch: str | None) -> str:
        """
        Ensures the repository is cloned into the deterministic path.
        If it already exists, skips cloning and returns the path.
        """
        if not branch or not branch.strip():
            raise ValueError(
                "Branch name is required for cloning; got empty/None. "
                "Ensure the finding has a branch asset with a valid name."
            )

        branch = branch.strip()
        target_path = self.get_repo_path(org_id, branch)
        target_str = str(target_path.resolve())

        async with self._lock:
            self._tracked_dirs.add(target_str)
            if target_path.exists() and any(target_path.iterdir()):
                logger.info(
                    "Found existing repository at {}; skipping clone.", target_str
                )
                return target_str

            # Create parent if missing
            target_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Cloning {} (branch: {}) into {}...",
                repo_url,
                branch,
                target_str,
            )
            git_cmd = ["git", "clone", "--depth", "1"]
            git_cmd.extend(["-b", branch])
            git_cmd.extend([repo_url, target_str])

            try:
                # Using native asyncio subprocess for better event loop integration and cancellation
                process = await asyncio.create_subprocess_exec(
                    *git_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await process.communicate()

                if process.returncode != 0:
                    error_msg = stderr.decode().strip()
                    logger.error("Git clone failed for {}: {}", repo_url, error_msg)
                    if target_path.exists():
                        shutil.rmtree(target_path, ignore_errors=True)
                    raise ValueError(f"Failed to clone repository: {error_msg}")

                return target_str
            except (asyncio.CancelledError, Exception) as e:
                # Cleanup partially created folder on failure or cancellation
                if target_path.exists():
                    shutil.rmtree(target_path, ignore_errors=True)
                if isinstance(e, asyncio.CancelledError):
                    logger.warning("Cloning cancelled for {}", repo_url)
                    raise
                logger.error("Unexpected error during clone for {}: {}", repo_url, e)
                raise ValueError(f"Failed to clone repository: {e}") from e

    def cleanup_all(self) -> None:
        """Recursively deletes all directories tracked by this instance."""
        for path_str in list(self._tracked_dirs):
            path = Path(path_str)
            if path.exists():
                try:
                    logger.info("Cleaning up tracked repository directory: {}", path_str)
                    shutil.rmtree(path)
                except Exception as e:
                    logger.warning(
                        "Failed to clean up tracked directory {}: {}", path_str, e
                    )
        self._tracked_dirs.clear()
