from __future__ import annotations

import contextvars
import json
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from datetime import UTC, datetime
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


@dataclass(frozen=True, slots=True)
class _ModelPricing:
    input_token_cost: Decimal
    output_token_cost: Decimal
    cached_token_cost: Decimal
    currency: str


_PRICING_CACHE_LOCK = threading.Lock()
_PRICING_CACHE: dict[str, _ModelPricing] = {}


def _load_model_pricing_from_core_db(model_slug: str) -> _ModelPricing | None:
    """
    Best-effort pricing lookup from Core DB `public.model_pricing` by `model_slug`.

    Table stores per-token costs; observability expects per-1M-token prices.
    """
    slug = str(model_slug or "").strip()
    if not slug:
        return None

    with _PRICING_CACHE_LOCK:
        cached = _PRICING_CACHE.get(slug)
        if cached is not None:
            return cached

    try:
        import psycopg
    except Exception:
        return None

    try:
        from ..utils.org_region_resolver import get_org_region_resolver

        dsn = get_org_region_resolver().core_db_dsn
        conn = psycopg.connect(dsn, connect_timeout=float(settings.DB_CONNECT_TIMEOUT))
    except Exception:
        return None

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT input_token_cost, output_token_cost, cached_token_cost, currency
                FROM public.model_pricing
                WHERE enabled = true AND model_slug = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (slug,),
            )
            row = cur.fetchone()
        if not row:
            return None

        pricing = _ModelPricing(
            input_token_cost=Decimal(str(row[0])),
            output_token_cost=Decimal(str(row[1])),
            cached_token_cost=Decimal(str(row[2])),
            currency=str(row[3] or "USD"),
        )
        with _PRICING_CACHE_LOCK:
            _PRICING_CACHE[slug] = pricing
        return pricing
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _per_1m(cost_per_token: Decimal) -> float:
    return float(cost_per_token * Decimal("1000000"))


class KafkaObservabilityHook:
    _instance: KafkaObservabilityHook | None = None

    def __new__(cls) -> KafkaObservabilityHook:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def topic(self) -> str:
        return settings.KAFKA_OBSERVABILITY_TOPIC or "queue.ai.invocation.logs"

    async def _send_event(self, event: object) -> None:
        """
        Best-effort send to Kafka. Starts producer if needed.

        This hook is used from API handlers and standalone Kafka consumers; the producer
        might not be started in the latter case.
        """
        try:
            await kafka_service.start()
            # (H) Shape observability events as slug/payload for downstream consumers,
            # and emit additional completed/failed envelopes.
            to_send = event
            extra: list[object] = []

            if isinstance(event, AIToolCalled):
                # Map internal status → external status
                status_map = {
                    "completed": "success",
                    "failed": "error",
                    "started": "started",
                }
                ext_status = status_map.get(event.status, event.status)
                created_at = datetime.fromtimestamp(
                    event.timestamp_ms / 1000.0, tz=UTC
                ).isoformat()

                # Best-effort model/provider inference; non-LLM tools can use generic labels.
                model = getattr(settings.active_orchestrator_config, "model_id", None)
                provider = getattr(settings.active_orchestrator_config, "provider", None)
                description = f"Tool call for {event.tool_name}"

                to_send = {
                    "slug": "ai.tool.called",
                    "payload": {
                        "toolCallId": event.tool_call_id,
                        "invocationId": event.invocation_id,
                        "orgId": event.org_id,
                        "toolName": event.tool_name,
                        "description": description,
                        "model": model,
                        "provider": provider,
                        "status": ext_status,
                        "durationMs": event.duration_ms,
                        "createdAt": created_at,
                        "metadata": {},
                    },
                }

                # Emit dedicated completed/failed envelopes when appropriate.
                if event.status == "completed":
                    extra.append(
                        {
                            "slug": "ai.tool.call.completed",
                            "payload": {
                                "toolCallId": event.tool_call_id,
                                "invocationId": event.invocation_id,
                                "orgId": event.org_id,
                                "toolName": event.tool_name,
                                "completedAt": created_at,
                            },
                        }
                    )
                elif event.status == "failed":
                    extra.append(
                        {
                            "slug": "ai.tool.call.failed",
                            "payload": {
                                "toolCallId": event.tool_call_id,
                                "invocationId": event.invocation_id,
                                "orgId": event.org_id,
                                "toolName": event.tool_name,
                                "completedAt": created_at,
                            },
                        }
                    )

            if isinstance(event, AIToolUsage):
                # Wrap tool usage in a slug/payload envelope with basic pricing metadata.
                model = event.model_name
                pricing = _load_model_pricing_from_core_db(model)
                to_send = {
                    "slug": "ai.tool.usage",
                    "payload": {
                        "orgId": event.org_id,
                        "invocationId": event.invocation_id,
                        "toolCallId": event.tool_call_id,
                        "toolName": event.tool_name,
                        "usageDetails": {
                            "price": {
                                "inputPricePer1MTokens": (
                                    _per_1m(pricing.input_token_cost) if pricing else 0.0
                                ),
                                "cachedInputPricePer1MTokens": (
                                    _per_1m(pricing.cached_token_cost) if pricing else 0.0
                                ),
                                "outputPricePer1MTokens": (
                                    _per_1m(pricing.output_token_cost) if pricing else 0.0
                                ),
                                "currency": (pricing.currency if pricing else "USD"),
                                "usedModel": model,
                            },
                            "usage": {
                                "inputTokenUsage": event.input_tokens,
                                "outputTokenUsage": event.output_tokens,
                                "cacheTokenUsage": 0,
                            },
                        },
                        "metadata": {},
                    },
                }

            if isinstance(event, AIMessageCreated):
                created_at = datetime.fromtimestamp(
                    event.timestamp_ms / 1000.0, tz=UTC
                ).isoformat()
                actor = event.actor
                meta_type = "message"
                if isinstance(actor, str):
                    a = actor.strip().lower()
                    if a in ("system prompt", "user prompt"):
                        meta_type = "prompt"
                to_send = {
                    "slug": "ai.message.created",
                    "payload": {
                        "invocationId": event.invocation_id,
                        "orgId": event.org_id,
                        "toolCallId": event.tool_call_id,
                        "actor": actor,
                        "message": event.content,
                        "metadata": {"type": meta_type},
                        "createdAt": created_at,
                    },
                }

            if isinstance(event, AIInvocationCompleted):
                end_time = datetime.fromtimestamp(
                    event.timestamp_ms / 1000.0, tz=UTC
                ).isoformat()
                model = getattr(settings.active_orchestrator_config, "model_id", None)
                provider = getattr(settings.active_orchestrator_config, "provider", None)
                # (H) compliancesCount is domain-specific; emit 0 when unknown.
                to_send = {
                    "slug": "ai.invocation.completed",
                    "payload": {
                        "invocationId": event.invocation_id,
                        "invocationEndTime": end_time,
                        "durationMs": event.duration_ms,
                        "metadata": {
                            "model": model,
                            "provider": provider,
                            "compliancesCount": 0,
                        },
                    },
                }

            def _invocation_key(obj: object) -> str | None:
                # Internal pydantic events
                if hasattr(obj, "invocation_id"):
                    try:
                        iid = getattr(obj, "invocation_id")
                        if isinstance(iid, str) and iid.strip():
                            return iid.strip()
                    except Exception:
                        pass
                # Slug/payload envelopes
                if isinstance(obj, dict):
                    payload = obj.get("payload")
                    if isinstance(payload, dict):
                        iid = payload.get("invocationId")
                        if isinstance(iid, str) and iid.strip():
                            return iid.strip()
                return None

            key = _invocation_key(event) or _invocation_key(to_send)

            await kafka_service.send(self.topic, to_send, key=key)
            for item in extra:
                await kafka_service.send(self.topic, item, key=key)
        except Exception:
            # Observability must never break the main workload.
            return

    def _get_current_ms(self) -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _format_message_markdown(*, content: str, actor: str) -> str:
        """
        Normalize all ai.message.created payloads to markdown so the frontend can render them
        consistently (prompts, JSON payloads, model outputs).
        """
        def _short(s: str, n: int = 240) -> str:
            s = s.strip()
            return s if len(s) <= n else (s[: n - 1] + "…")

        def _json_to_md(value: object, *, key: str | None = None, depth: int = 0) -> list[str]:
            """
            Render JSON-ish data (dict/list/str/number/bool/None) into readable Markdown.
            Keeps it simple and stable for front-end rendering.
            """
            indent = "  " * depth
            lines: list[str] = []

            def _heading(k: str) -> None:
                if depth == 0:
                    lines.append(f"## {k}")
                else:
                    lines.append(f"{indent}- **{k}**")

            if key is not None:
                _heading(key)
                # For nested values under a heading, increase depth for children.
                depth += 1
                indent = "  " * depth

            if isinstance(value, dict):
                if not value:
                    lines.append(f"{indent}- _empty_")
                    return lines
                for k, v in value.items():
                    # Render code-bearing fields as fenced blocks so Markdown viewers don't
                    # interpret HTML-like strings (e.g. "<iframe ...>") as raw HTML.
                    if isinstance(v, str) and str(k).lower() in (
                        "before",
                        "after",
                        "snippet",
                        "code",
                        "patch",
                    ):
                        lines.append(f"{indent}- **{k}**:")
                        block_indent = indent + "  "
                        lines.append(f"{block_indent}```")
                        lines.extend(f"{block_indent}{ln}" for ln in v.rstrip("\n").splitlines())
                        lines.append(f"{block_indent}```")
                        continue
                    if isinstance(v, (dict, list)):
                        lines.extend(_json_to_md(v, key=str(k), depth=depth))
                    else:
                        if v is None:
                            v_s = "_null_"
                        elif isinstance(v, bool):
                            v_s = "true" if v else "false"
                        else:
                            v_s = str(v)
                        lines.append(f"{indent}- **{k}**: {_short(v_s, 800)}")
                return lines

            if isinstance(value, list):
                if not value:
                    lines.append(f"{indent}- _empty list_")
                    return lines
                for idx, item in enumerate(value, start=1):
                    if isinstance(item, (dict, list)):
                        lines.extend(_json_to_md(item, key=f"{idx}", depth=depth))
                    else:
                        lines.append(f"{indent}- {idx}. {_short(str(item), 800)}")
                return lines

            # Scalars
            if value is None:
                lines.append(f"{indent}- _null_")
            elif isinstance(value, bool):
                lines.append(f"{indent}- {'true' if value else 'false'}")
            else:
                lines.append(f"{indent}- {_short(str(value), 2000)}")
            return lines

        raw = content if isinstance(content, str) else str(content)
        text = raw.strip()

        # Prefer Markdown rendering when content is JSON (LLM outputs are often JSON).
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

        if parsed is not None:
            # Special-case common pipeline shapes for a cleaner log view.
            if isinstance(parsed, dict):
                # Final aggregated response
                if isinstance(parsed.get("invocation_id"), str) and any(
                    k in parsed for k in ("evidence", "scoring", "remediation")
                ):
                    inv = parsed.get("invocation_id")
                    lines = [f"## Invocation `{inv}`", ""]
                    for section in ("evidence", "scoring", "remediation"):
                        part = parsed.get(section)
                        if isinstance(part, dict):
                            md = part.get("markdown")
                            if isinstance(md, str) and md.strip():
                                lines.append(f"## {section.title()}")
                                lines.append(md.strip())
                                lines.append("")
                            else:
                                lines.append(f"## {section.title()}")
                                lines.extend(_json_to_md(part, depth=0))
                                lines.append("")
                    models = parsed.get("models")
                    if isinstance(models, dict) and models:
                        lines.append("## Models")
                        lines.extend(_json_to_md(models, depth=0))
                        lines.append("")
                    return "\n".join(lines).strip()

                # Evidence stage output
                if isinstance(parsed.get("items"), list):
                    lines = ["## Evidence", ""]
                    items = parsed.get("items") or []
                    for idx, item in enumerate(items, start=1):
                        if not isinstance(item, dict):
                            continue
                        q = str(item.get("question") or f"Finding {idx}")
                        a = str(item.get("answer") or "")
                        lines.append(f"### {idx}. {q}")
                        if a.strip():
                            lines.append("")
                            lines.append(a.strip())
                        code_ref = item.get("code_reference")
                        if code_ref:
                            lines.append("")
                            lines.append(f"- **Code reference**: `{code_ref}`")
                        ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else None
                        if ev:
                            f = ev.get("file")
                            lr = ev.get("line_range")
                            if f or lr:
                                lines.append(f"- **Evidence**: `{f}` {lr or ''}".rstrip())
                        lines.append("")
                    return "\n".join(lines).strip()

                # Scoring / Remediation stage outputs (generic but readable)
                if isinstance(parsed.get("findings"), list):
                    lines = ["## Findings", ""]
                    lines.extend(_json_to_md(parsed.get("findings"), depth=0))
                    return "\n".join(lines).strip()

            # Fallback: generic JSON-to-markdown.
            return "\n".join(_json_to_md(parsed, depth=0)).strip()

        fence_lang = "text"
        if actor.lower().startswith("system"):
            fence_lang = "text"
        return f"```{fence_lang}\n{text}\n```" if text else f"```{fence_lang}\n\n```"

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

        md = self._format_message_markdown(content=content, actor=actor)
        event = AIMessageCreated(
            invocation_id=ctx.invocation_id,
            org_id=ctx.org_id,
            timestamp_ms=self._get_current_ms(),
            content=md,
            tool_call_id=tool_call_id,
            actor=actor,
        )
        await self._send_event(event)

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
        await self._send_event(event)

    async def log_tool_failed(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        duration_ms: int | None = None,
    ) -> None:
        ctx = _hook_context.get()
        if not ctx:
            return
        await self._send_event(
            AIToolCalled(
                invocation_id=ctx.invocation_id,
                org_id=ctx.org_id,
                timestamp_ms=self._get_current_ms(),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status="failed",
                duration_ms=duration_ms,
            )
        )

    async def log_tool_completed(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        duration_ms: int | None = None,
    ) -> None:
        ctx = _hook_context.get()
        if not ctx:
            return
        await self._send_event(
            AIToolCalled(
                invocation_id=ctx.invocation_id,
                org_id=ctx.org_id,
                timestamp_ms=self._get_current_ms(),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status="completed",
                duration_ms=duration_ms,
            )
        )

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
        await self._send_event(
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
        await self._send_event(
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

    async def log_tool_usage(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        Emit ai.tool.usage without implying an LLM call.
        Useful for non-LLM pipeline steps (e.g., indexing) so the frontend can render a
        consistent usage block for every tool.
        """
        ctx = _hook_context.get()
        if not ctx:
            return
        await self._send_event(
            AIToolUsage(
                invocation_id=ctx.invocation_id,
                org_id=ctx.org_id,
                timestamp_ms=self._get_current_ms(),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                model_name=model_name,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
            )
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
        await self._send_event(event)

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
        await self._send_event(event)


observability_hook = KafkaObservabilityHook()
