from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Schema version 1: mirrors /api/chat job shape; partition key should be org_id.


class ChatJobPayloadV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    org_id: str
    org_tool_findings_ids: list[str]
    orchestrator: str | None = None
    cypher: str | None = None
    invocation_id: str = Field(
        ...,
        description="Correlation id for logs.",
    )

    @field_validator("org_id")
    @classmethod
    def _org_id_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("org_id is required")
        return cleaned

    @field_validator("org_tool_findings_ids")
    @classmethod
    def _findings_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [x.strip() for x in value if isinstance(x, str) and x.strip()]
        if not cleaned:
            raise ValueError("org_tool_findings_ids must be non-empty")
        return cleaned

    @model_validator(mode="before")
    @classmethod
    def _legacy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "orgId" in out and "org_id" not in out:
            out["org_id"] = out.pop("orgId")
        if "orgToolFindingsIds" in out and "org_tool_findings_ids" not in out:
            out["org_tool_findings_ids"] = out.pop("orgToolFindingsIds")
        if "invocationId" in out and "invocation_id" not in out:
            out["invocation_id"] = out.pop("invocationId")
        if "schemaVersion" in out and "schema_version" not in out:
            out["schema_version"] = out.pop("schemaVersion")
        return out

    @classmethod
    def model_validate_payload(cls, raw: Any) -> ChatJobPayloadV1:
        """Validate consumer-side dict; logs and re-raises validation errors."""
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            logger.warning("Invalid chat job payload: {}", exc)
            raise
