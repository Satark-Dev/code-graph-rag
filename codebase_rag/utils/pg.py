from __future__ import annotations

import threading
from collections.abc import Callable

from ..utils.dependencies import has_pgvector

_init_lock = threading.Lock()
_initialized: set[str] = set()


class PgInitKey:
    PGVECTOR_EMBEDDINGS = "pgvector_embeddings"
    TOOL_CALL_STORE = "tool_call_store"


def effective_org_id_for_routing() -> str | None:
    """Resolved tenant id from request context."""
    from ..request_context import org_id_context

    ctx = org_id_context.get()
    if ctx and str(ctx).strip():
        return str(ctx).strip()
    return None


def require_effective_org_id() -> str:
    """Tenant id required for Postgres (Core DB → region → org shard)."""
    oid = effective_org_id_for_routing()
    if not oid:
        raise RuntimeError(
            "No org_id is set. Pass org_id on the API request (e.g. /api/chat)."
        )
    return oid


def pg_init_key(base: str) -> str:
    """Schema init runs once per effective org_id (each tenant may use a different org DB)."""
    return f"{base}:{require_effective_org_id()}"


if has_pgvector():
    import psycopg

    def pg_connect(*, autocommit: bool = True):
        """
        Postgres for pgvector and tool-call persistence via Core DB (user_org)
        → region → org DB DSN.
        """
        from .org_region_resolver import get_org_region_resolver

        org_id = require_effective_org_id()
        resolver = get_org_region_resolver()
        region = resolver.get_region_for_org_id(org_id)
        dsn = resolver.get_org_db_connection_string(region)
        return psycopg.connect(dsn, autocommit=autocommit)

else:

    def pg_connect(*, autocommit: bool = True):  # type: ignore[no-redef]
        raise RuntimeError("Postgres connection requested but pgvector/psycopg not installed")


def ensure_pg_initialized(key: str, init_fn: Callable[[], None]) -> bool:
    with _init_lock:
        if key in _initialized:
            return False
        init_fn()
        _initialized.add(key)
        return True
