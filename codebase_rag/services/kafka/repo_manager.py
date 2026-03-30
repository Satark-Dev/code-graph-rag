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

    def get_repo_path(self, org_id: str, branch: str | None) -> Path:
        """Derives a deterministic local path for a (org, branch) pair."""
        # Using org_id and branch name to create a stable folder name
        safe_branch = (branch or "default").replace("/", "_").replace("\\", "_")
        return self.get_base_temp_dir() / f"{org_id}_{safe_branch}"

    async def ensure_cloned(self, repo_url: str, org_id: str, branch: str | None) -> str:
        """
        Ensures the repository is cloned into the deterministic path.
        If it already exists, skips cloning and returns the path.
        """
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
                branch or "default",
                target_str,
            )
            git_cmd = ["git", "clone", "--depth", "1"]
            if branch:
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
