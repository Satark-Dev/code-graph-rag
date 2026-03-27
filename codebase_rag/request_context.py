"""Request-scoped context (org_id, user_id) for org/region Postgres routing."""

from __future__ import annotations

import contextvars
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

org_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "org_id", default=None
)
user_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "user_id", default=None
)


class OrgUserContextBody(BaseModel):
    """Subset of fields used by APIs that require tenant context."""

    org_id: str
    user_id: str | None = None


def bind_org_context(*, org_id: str | None, user_id: str | None = None) -> None:
    """Set ContextVars for the current async task (call at the start of a request handler)."""
    if org_id:
        org_id_context.set(org_id)
    if user_id:
        user_id_context.set(user_id)


def reset_org_context() -> None:
    """Clear org/user context (e.g. after a request)."""
    org_id_context.set(None)
    user_id_context.set(None)


def bind_context_from_body(body: Any, *, require_org: bool = True) -> None:
    """
    Read org_id (and optional user_id) from a Pydantic model or mapping.
    Used when the request body follows OrgUserContextBody-like shape.
    """
    if body is None:
        raise HTTPException(status_code=400, detail="Missing request body.")
    org_id = getattr(body, "org_id", None)
    if isinstance(body, dict):
        org_id = body.get("org_id", org_id)
    if require_org and not org_id:
        raise HTTPException(status_code=400, detail="Missing org_id in body JSON.")
    if org_id:
        org_id_context.set(org_id)
    user_id = getattr(body, "user_id", None)
    if isinstance(body, dict):
        user_id = body.get("user_id", user_id)
    if user_id:
        user_id_context.set(user_id)
