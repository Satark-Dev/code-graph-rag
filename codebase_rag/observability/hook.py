from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass
from uuid import uuid4

from codebase_rag.config import settings
from codebase_rag.events.observability import (
    AIInvocationCompleted,
    AIMessageCreated,
    AIToolCalled,
    AIToolUsage,
)
from codebase_rag.services.kafka.producer import kafka_service


@dataclass(slots=True)
class HookContext:
    invocation_id: str
    org_id: str
    user_id: str | None
    start_time_ms: int
    total_tokens: int = 0


_hook_context: contextvars.ContextVar[HookContext | None] = contextvars.ContextVar(
    "_hook_context", default=None
)


class KafkaObservabilityHook:
    _instance: KafkaObservabilityHook | None = None

    def __new__(cls) -> KafkaObservabilityHook:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def topic(self) -> str:
        return settings.KAFKA_OBSERVABILITY_TOPIC or "queue.ai.invocation.logs"

    def _get_current_ms(self) -> int:
        return int(time.time() * 1000)

    async def before_chat(
        self,
        org_id: str,
        invocation_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        iid = invocation_id or str(uuid4())
        ctx = HookContext(
            invocation_id=iid,
            org_id=org_id,
            user_id=user_id,
            start_time_ms=self._get_current_ms(),
        )
        _hook_context.set(ctx)
        return iid

    async def log_message(
        self,
        content: str,
        tool_call_id: str,
        actor: str = "assistant",
    ) -> None:
        ctx = _hook_context.get()
        if not ctx:
            return

        event = AIMessageCreated(
            invocation_id=ctx.invocation_id,
            org_id=ctx.org_id,
            timestamp_ms=self._get_current_ms(),
            content=content,
            tool_call_id=tool_call_id,
            actor=actor,
        )
        await kafka_service.send(self.topic, event)

    async def log_tool_start(self, tool_name: str, tool_call_id: str) -> None:
        ctx = _hook_context.get()
        if not ctx:
            return

        event = AIToolCalled(
            invocation_id=ctx.invocation_id,
            org_id=ctx.org_id,
            timestamp_ms=self._get_current_ms(),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            status="started",
        )
        await kafka_service.send(self.topic, event)

    async def log_llm_usage(
        self,
        tool_name: str,
        tool_call_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int | None = None,
    ) -> None:
        ctx = _hook_context.get()
        if not ctx:
            return

        ctx.total_tokens += (input_tokens + output_tokens)

        # Log completion of tool call
        await kafka_service.send(
            self.topic,
            AIToolCalled(
                invocation_id=ctx.invocation_id,
                org_id=ctx.org_id,
                timestamp_ms=self._get_current_ms(),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status="completed",
                duration_ms=duration_ms,
            ),
        )

        # Log usage details
        await kafka_service.send(
            self.topic,
            AIToolUsage(
                invocation_id=ctx.invocation_id,
                org_id=ctx.org_id,
                timestamp_ms=self._get_current_ms(),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    async def after_chat_success(self, tool_call_id: str) -> None:
        ctx = _hook_context.get()
        if not ctx:
            return

        duration = self._get_current_ms() - ctx.start_time_ms
        event = AIInvocationCompleted(
            invocation_id=ctx.invocation_id,
            org_id=ctx.org_id,
            timestamp_ms=self._get_current_ms(),
            status="success",
            duration_ms=duration,
            tool_call_id=tool_call_id,
        )
        await kafka_service.send(self.topic, event)

    async def after_chat_error(self, error: Exception, tool_call_id: str) -> None:
        ctx = _hook_context.get()
        if not ctx:
            return

        duration = self._get_current_ms() - ctx.start_time_ms
        event = AIInvocationCompleted(
            invocation_id=ctx.invocation_id,
            org_id=ctx.org_id,
            timestamp_ms=self._get_current_ms(),
            status="error",
            duration_ms=duration,
            tool_call_id=tool_call_id,
            error_message=str(error),
        )
        await kafka_service.send(self.topic, event)


observability_hook = KafkaObservabilityHook()
