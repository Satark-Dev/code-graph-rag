from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(_StrictModel):
    question: str
    answer: str
    code_reference: str | None = None
    evidence: dict[str, Any] | None = None


class ScoringFinding(_StrictModel):
    analysis: dict[str, Any]


class RemediationFinding(_StrictModel):
    remediation: dict[str, Any]


class TokenUsage(BaseModel):
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class EvidenceToolOutput(BaseModel):
    """LLM output contract for evidence stage (no meta fields)."""

    items: list[EvidenceItem]


class ScoringToolOutput(BaseModel):
    """LLM output contract for scoring stage (no meta fields)."""

    findings: list[ScoringFinding]


class RemediationToolOutput(BaseModel):
    """LLM output contract for remediation stage (no meta fields)."""

    findings: list[RemediationFinding]


class EvidenceStage(BaseModel):
    items: list[EvidenceItem]
    timings_ms: int = Field(..., ge=0)
    token_usage: TokenUsage


class ScoringStage(BaseModel):
    findings: list[ScoringFinding]
    timings_ms: int = Field(..., ge=0)
    token_usage: TokenUsage


class RemediationStage(BaseModel):
    findings: list[RemediationFinding]
    timings_ms: int = Field(..., ge=0)
    token_usage: TokenUsage


class ModelDescriptor(BaseModel):
    provider: str
    model: str


class ChatModels(BaseModel):
    orchestrator: ModelDescriptor
    cypher: ModelDescriptor


class ChatResponsePayload(BaseModel):
    invocation_id: UUID
    evidence: EvidenceStage
    scoring: ScoringStage
    remediation: RemediationStage
    models: ChatModels


class ApiErrorDetail(BaseModel):
    code: str
    message: str
    invocation_id: UUID | None = None


class ApiErrorResponse(BaseModel):
    error: ApiErrorDetail

