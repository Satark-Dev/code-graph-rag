from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AIEventBase(BaseModel):
    invocation_id: str
    org_id: str
    timestamp_ms: int


class AIMessageCreated(AIEventBase):
    event_type: Literal["ai.message.created"] = "ai.message.created"
    content: str
    tool_call_id: str
    actor: str = "assistant"


class AIToolCalled(AIEventBase):
    event_type: Literal["ai.tool.called"] = "ai.tool.called"
    tool_name: str
    tool_call_id: str
    status: Literal["started", "completed", "failed"]
    duration_ms: int | None = None


class AIToolUsage(AIEventBase):
    event_type: Literal["ai.tool.usage"] = "ai.tool.usage"
    tool_name: str
    tool_call_id: str
    model_name: str
    input_tokens: int
    output_tokens: int


class AIInvocationCompleted(AIEventBase):
    event_type: Literal["ai.invocation.completed"] = "ai.invocation.completed"
    status: Literal["success", "error"]
    duration_ms: int
    tool_call_id: str
    error_message: str | None = None
