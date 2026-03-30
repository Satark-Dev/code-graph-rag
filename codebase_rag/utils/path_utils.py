from pathlib import Path

from .. import constants as cs


def should_skip_path(
    path: Path,
    repo_path: Path,
    exclude_paths: frozenset[str] | None = None,
    unignore_paths: frozenset[str] | None = None,
) -> bool:
    # Performance critical: use string operations instead of pathlib to avoid object overhead
    if path.is_file() and path.suffix in cs.IGNORE_SUFFIXES:
        return True

    path_str = path.as_posix()
    repo_path_str = repo_path.as_posix()

    if path_str.startswith(repo_path_str):
        rel_path_str = path_str[len(repo_path_str) :].lstrip("/")
        if not rel_path_str:
            return False
    else:
        try:
            rel_path_str = path.relative_to(repo_path).as_posix()
        except ValueError:
            return False

    parts = rel_path_str.split("/")
    dir_parts = parts[:-1] if path.is_file() else parts
    dir_parts_set = set(dir_parts)

    if exclude_paths:
        if not exclude_paths.isdisjoint(dir_parts_set):
            return True
        if rel_path_str in exclude_paths:
            return True
        if any(rel_path_str.startswith(f"{p}/") for p in exclude_paths):
            return True

    if unignore_paths and any(
        rel_path_str == p or rel_path_str.startswith(f"{p}/") for p in unignore_paths
    ):
        return False

    return not cs.IGNORE_PATTERNS.isdisjoint(dir_parts_set)
