from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class EvidenceJobPayloadV1(BaseModel):
    """Stage 1: evidence generation. Mirrors the old chat job payload."""

    model_config = ConfigDict(extra="ignore")

    org_id: str
    org_tool_findings_ids: list[str]
    orchestrator: str | None = None
    cypher: str | None = None
    invocation_id: str = Field(..., description="Correlation id for logs.")

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
        return out

    @classmethod
    def model_validate_payload(cls, raw: Any) -> EvidenceJobPayloadV1:
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            logger.warning("Invalid evidence job payload: {}", exc)
            raise


class DownstreamStagePayloadV1(BaseModel):
    """Stage 2/3: scoring or remediation, fed by evidence persisted in tool_call_store."""

    model_config = ConfigDict(extra="ignore")

    org_id: str
    tool_call_id: str
    cache_key: str
    repo_state_hash: str
    target_repo_path: str
    invocation_id: str
    # Unique per evidence Kafka message; used to refcount shared clones under target_repo_path.
    repo_lease_id: str
    org_tool_findings_ids: list[str]
    orchestrator: str | None = None
    cypher: str | None = None

    @field_validator(
        "org_id",
        "tool_call_id",
        "cache_key",
        "repo_state_hash",
        "target_repo_path",
        "invocation_id",
        "repo_lease_id",
    )
    @classmethod
    def _non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field is required")
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
    def _default_repo_lease_id(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "repoLeaseId" in out and "repo_lease_id" not in out:
            out["repo_lease_id"] = out.pop("repoLeaseId")
        rl = out.get("repo_lease_id")
        if not rl or (isinstance(rl, str) and not rl.strip()):
            ck = out.get("cache_key")
            if ck and str(ck).strip():
                out["repo_lease_id"] = str(ck).strip()
        return out

    @classmethod
    def model_validate_payload(cls, raw: Any) -> DownstreamStagePayloadV1:
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            logger.warning("Invalid downstream stage payload: {}", exc)
            raise

