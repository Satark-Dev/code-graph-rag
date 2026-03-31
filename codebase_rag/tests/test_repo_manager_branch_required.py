import asyncio
from pathlib import Path

import pytest

import codebase_rag.services.kafka.repo_manager as repo_manager_mod


def test_repo_manager_require_existing_local_clone_requires_branch() -> None:
    manager = repo_manager_mod.RepoManager()

    with pytest.raises(ValueError, match="Branch name is required"):
        manager.require_existing_local_clone(org_id="org", branch=None)

    with pytest.raises(ValueError, match="Branch name is required"):
        manager.require_existing_local_clone(org_id="org", branch="   ")


def test_repo_manager_require_existing_local_clone_missing_or_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repos"
    root.mkdir()
    monkeypatch.setattr(repo_manager_mod.settings, "KAFKA_REPO_ROOT", str(root))

    manager = repo_manager_mod.RepoManager()

    with pytest.raises(ValueError, match="Local repository clone not found"):
        manager.require_existing_local_clone(org_id="org1", branch="main")

    expected = root / "org1_main"
    expected.mkdir()
    with pytest.raises(ValueError, match="exists but is empty"):
        manager.require_existing_local_clone(org_id="org1", branch="main")

    (expected / ".keep").write_text("x")
    assert manager.require_existing_local_clone(org_id="org1", branch="main") == str(
        expected.resolve()
    )


def test_repo_manager_ensure_cloned_requires_branch_name() -> None:
    from codebase_rag.services.kafka.repo_manager import RepoManager

    manager = RepoManager()

    async def _raises(branch: str | None) -> None:
        await manager.ensure_cloned(
            repo_url="https://example.com/repo.git", org_id="org", branch=branch
        )

    for branch in (None, "", "   "):
        with pytest.raises(ValueError, match="Branch name is required"):
            asyncio.run(_raises(branch))

