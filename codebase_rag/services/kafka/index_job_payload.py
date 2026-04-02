from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class IndexJobPayload(BaseModel):
    """Payload for index jobs (equivalent to /api/index request)."""

    model_config = ConfigDict(extra="ignore")

    org_id: str
    repo_url: str
    # Branch asset row: public.org_tool_findings.id where type='asset'; git branch = name.
    org_tool_findings_id: str
    clean: bool = True
    exclude: list[str] | None = None
    invocation_id: str

    @field_validator("org_id")
    @classmethod
    def _org_id_non_blank(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("org_id is required")
        return v

    @field_validator("org_tool_findings_id")
    @classmethod
    def _finding_non_blank(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("org_tool_findings_id is required")
        return v

    @field_validator("repo_url")
    @classmethod
    def _repo_url_non_blank(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("repo_url is required")
        return v

    @classmethod
    def model_validate_payload(cls, raw: Any) -> IndexJobPayload:
        """Validate consumer-side dict; logs and re-raises validation errors."""
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            logger.warning("Invalid index job payload: {}", exc)
            raise

