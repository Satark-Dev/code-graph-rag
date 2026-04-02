import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from codebase_rag.mcp.server import get_project_root


class TestGetProjectRoot:
    """Test suite for get_project_root() function."""

    def test_uses_environment_variable_when_set(self, tmp_path: Path) -> None:
        """Test that TARGET_REPO_PATH environment variable takes priority."""
        test_path = tmp_path / "test_repo"
        test_path.mkdir()

        with patch.dict(os.environ, {"TARGET_REPO_PATH": str(test_path)}):
            result = get_project_root()

        assert result == test_path.resolve()

    def test_uses_claude_project_root_when_target_not_set(self, tmp_path: Path) -> None:
        """Test that CLAUDE_PROJECT_ROOT is used when TARGET_REPO_PATH is not set."""
        test_path = tmp_path / "claude_repo"
        test_path.mkdir()

        with patch.dict(os.environ, {"CLAUDE_PROJECT_ROOT": str(test_path)}, clear=True):
            result = get_project_root()

        assert result == test_path.resolve()

    def test_defaults_to_cwd_when_no_env_vars_set(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Test that current working directory is used when no environment variables are set."""
        test_cwd = tmp_path / "current_dir"
        test_cwd.mkdir()

        monkeypatch.chdir(test_cwd)
        # Clear relevant env vars
        monkeypatch.delenv("TARGET_REPO_PATH", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_ROOT", raising=False)
        monkeypatch.delenv("PWD", raising=False)

        result = get_project_root()
        assert result == test_cwd.resolve()

    def test_env_var_priority(self, tmp_path: Path) -> None:
        """Test the priority of environment variables: TARGET > CLAUDE > PWD."""
        target_path = tmp_path / "target_repo"
        claude_path = tmp_path / "claude_repo"
        pwd_path = tmp_path / "pwd_repo"

        target_path.mkdir()
        claude_path.mkdir()
        pwd_path.mkdir()

        # TARGET vs others
        with patch.dict(os.environ, {
            "TARGET_REPO_PATH": str(target_path),
            "CLAUDE_PROJECT_ROOT": str(claude_path),
            "PWD": str(pwd_path)
        }, clear=True):
            assert get_project_root() == target_path.resolve()

        # CLAUDE vs PWD
        with patch.dict(os.environ, {
            "CLAUDE_PROJECT_ROOT": str(claude_path),
            "PWD": str(pwd_path)
        }, clear=True):
            assert get_project_root() == claude_path.resolve()

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlinks require special privileges on Windows"
    )
    def test_handles_symlinks(self, tmp_path: Path) -> None:
        """Test that symlinks are resolved correctly."""
        real_path = tmp_path / "real_repo"
        real_path.mkdir()
        symlink_path = tmp_path / "symlink_repo"
        symlink_path.symlink_to(real_path)

        with patch.dict(os.environ, {"TARGET_REPO_PATH": str(symlink_path)}, clear=True):
            result = get_project_root()

        assert result == real_path.resolve()

    def test_raises_error_when_path_does_not_exist(self) -> None:
        """Test that ValueError is raised when the path does not exist."""
        nonexistent_path = "/path/that/does/not/exist/at/all"

        with patch.dict(os.environ, {"TARGET_REPO_PATH": nonexistent_path}, clear=True):
            with pytest.raises(
                ValueError, match="Target repository path does not exist"
            ):
                get_project_root()

    def test_raises_error_when_path_is_file(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when the path is a file, not a directory."""
        test_file = tmp_path / "test_file.txt"
        test_file.write_text(encoding="utf-8", data="test content")

        with patch.dict(os.environ, {"TARGET_REPO_PATH": str(test_file)}, clear=True):
            with pytest.raises(
                ValueError, match="Target repository path is not a directory"
            ):
                get_project_root()

    def test_resolves_relative_paths(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Test that relative paths are resolved to absolute paths."""
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)

        monkeypatch.chdir(parent)

        with patch.dict(os.environ, {"TARGET_REPO_PATH": "./child"}, clear=True):
            result = get_project_root()

        assert result == child.resolve()
        assert result.is_absolute()
