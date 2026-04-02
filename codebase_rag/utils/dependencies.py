from __future__ import annotations

import importlib.util
from collections.abc import Sequence

_dependency_cache: dict[str, bool] = {}


def _check_dependency(module_name: str) -> bool:
    if module_name not in _dependency_cache:
        _dependency_cache[module_name] = (
            importlib.util.find_spec(module_name) is not None
        )
    return _dependency_cache[module_name]


def has_pgvector() -> bool:
    return _check_dependency("pgvector") and _check_dependency("psycopg")


def has_pgvector_client() -> bool:
    # Legacy/optional dependency check for an older vector store implementation.
    return _check_dependency("pgvector_client")


def has_semantic_dependencies() -> bool:
    # Semantic search is backed by PGVector + psycopg. Embeddings are provided via OpenAI.
    return has_pgvector()


def check_dependencies(required_modules: Sequence[str]) -> bool:
    return all(_check_dependency(module) for module in required_modules)


def get_missing_dependencies(required_modules: Sequence[str]) -> list[str]:
    return [module for module in required_modules if not _check_dependency(module)]
