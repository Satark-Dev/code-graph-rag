from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class IndexJobPayload(BaseModel):
    """Payload for index jobs (equivalent to /api/index request)."""

    model_config = ConfigDict(extra="ignore")

    org_id: str
    repo_path: str
    clean: bool = True
    exclude: list[str] | None = None

    @field_validator("org_id")
    @classmethod
    def _org_id_non_blank(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("org_id is required")
        return v

    @field_validator("repo_path")
    @classmethod
    def _repo_non_blank(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("repo_path is required")
        return v

    @classmethod
    def model_validate_payload(cls, raw: Any) -> IndexJobPayload:
        """Validate consumer-side dict; logs and re-raises validation errors."""
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            logger.warning("Invalid index job payload: {}", exc)
            raise

