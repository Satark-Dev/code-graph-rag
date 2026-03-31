import asyncio
import hashlib
import json
import time
from uuid import uuid4
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError
from pydantic_ai import DeferredToolRequests, DeferredToolResults

from ..bootstrap import initialize_services_and_agent
from ..chat_schemas import EvidenceToolOutput, RemediationToolOutput, ScoringToolOutput
from ..config import settings
from ..constants import HASH_CACHE_FILENAME
from ..observability.hook import observability_hook
from ..prompts import (
    API_EVIDENCE_PROMPT,
    API_REMEDIATION_PROMPT,
    API_SCORING_PROMPT,
)
from ..services.llm import create_rag_orchestrator
from ..utils.token_utils import count_tokens
from ..utils.tool_call_store import new_run_id, store_tool_call


def _compute_repo_state_hash(target_repo_path: str) -> str:
    hash_file = Path(target_repo_path) / HASH_CACHE_FILENAME
    state_hash = target_repo_path
    if hash_file.is_file():
        try:
            state_hash = hashlib.md5(hash_file.read_bytes()).hexdigest()
        except OSError:
            pass
    return state_hash


def _persist_stage(
    *,
    run_id: str,
    cache_key: str,
    repo_path: str,
    repo_state_hash: str,
    stage: str,
    tool_input: dict[str, Any],
    tool_output: dict[str, Any],
) -> None:
    try:
        store_tool_call(
            run_id=run_id,
            cache_key=cache_key,
            repo_path=repo_path,
            repo_state_hash=repo_state_hash,
            stage=stage,
            tool_input=tool_input,
            tool_output=tool_output,
        )
    except Exception as e:
        logger.warning(f"Failed to persist {stage} stage: {e}")


class ChatStageError(Exception):
    __slots__ = ("code", "message", "run_id")

    def __init__(self, *, code: str, message: str, run_id: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.run_id = run_id


class PipelineStage(str, Enum):
    EVIDENCE = "evidence"
    SCORING = "scoring"
    REMEDIATION = "remediation"


def _validate_evidence_stage(data: dict[str, Any]) -> dict[str, Any]:
    """
    Accept both the new flat evidence schema (items[]) and the legacy schema
    (findings[] -> justification.questions_and_answers[]), normalizing to items[].
    """
    # New schema: already flat
    if isinstance(data.get("items"), list):
        EvidenceToolOutput.model_validate(data)
        return data

    # Legacy schema: findings -> justification.questions_and_answers
    findings = data.get("findings")
    if not isinstance(findings, list):
        # Let pydantic raise a useful error
        EvidenceToolOutput.model_validate(data)  # type: ignore[arg-type]
        return data

    items: list[dict[str, Any]] = []
    for finding in findings:
        justification = (
            finding.get("justification") if isinstance(finding, dict) else None
        )
        qas = (
            justification.get("questions_and_answers")
            if isinstance(justification, dict)
            else None
        )
        if not isinstance(qas, list):
            continue
        for qa in qas:
            if not isinstance(qa, dict):
                continue
            item = {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
            }
            if "code_reference" in qa:
                item["code_reference"] = qa.get("code_reference")
            if "evidence" in qa:
                item["evidence"] = qa.get("evidence")
            items.append(item)

    normalized = {"items": items}
    EvidenceToolOutput.model_validate(normalized)
    return normalized


def _validate_scoring_stage(data: dict[str, Any]) -> None:
    ScoringToolOutput.model_validate(data)


def _validate_remediation_stage(data: dict[str, Any]) -> None:
    RemediationToolOutput.model_validate(data)


_STAGE_VALIDATORS: dict[
    PipelineStage, Callable[[dict[str, Any]], dict[str, Any] | None]
] = {
    PipelineStage.EVIDENCE: _validate_evidence_stage,
    PipelineStage.SCORING: _validate_scoring_stage,
    PipelineStage.REMEDIATION: _validate_remediation_stage,
}


async def _run_with_timeout(coro, *, timeout_s: float, stage: str, run_id: str):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except TimeoutError as e:
        raise ChatStageError(
            code="TIMEOUT",
            message=f"{stage} stage timed out after {timeout_s} seconds",
            run_id=run_id,
        ) from e


def _parse_and_validate_stage(
    *, stage: PipelineStage, raw_text: str, run_id: str
) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except Exception as e:
        raise ChatStageError(
            code="LLM_INVALID_JSON",
            message=f"{stage} stage output was not valid JSON",
            run_id=run_id,
        ) from e

    try:
        validator = _STAGE_VALIDATORS.get(stage)
        if validator is None:
            raise ValueError(f"Unknown stage: {stage}")
        transformed = validator(parsed)
        if transformed is not None:
            parsed = transformed
    except ValidationError as e:
        raise ChatStageError(
            code="LLM_SCHEMA_MISMATCH",
            message=f"{stage} stage output did not match expected schema",
            run_id=run_id,
        ) from e

    return parsed


def _usage_from_result(result: Any) -> tuple[int | None, int | None, int | None]:
    """
    Best-effort extraction of token usage from pydantic_ai results across providers.
    Returns (input, output, total) or (None, None, None) if unavailable.
    """
    def _extract_usage(obj: Any) -> dict[str, Any] | None:
        if isinstance(obj, dict):
            return obj.get("usage") if isinstance(obj.get("usage"), dict) else obj

        # (H) pydantic-ai result objects have a usage() method.
        if hasattr(obj, "usage") and callable(obj.usage):
            try:
                # Try calling it and converting to dict
                usage_obj = obj.usage()
                if hasattr(usage_obj, "model_dump"):
                    return usage_obj.model_dump()
                if isinstance(usage_obj, dict):
                    return usage_obj
            except Exception:
                pass

        # Try finding usage attribute
        for attr in ("usage", "model_response", "response"):
            val = getattr(obj, attr, None)
            if val is not None:
                if isinstance(val, dict):
                    return val
                if hasattr(val, "usage") and not callable(val.usage):
                    return getattr(val, "usage")
                if hasattr(val, "model_dump"):
                    try:
                        d = val.model_dump()
                        if isinstance(d, dict):
                            return d.get("usage") or d
                    except Exception:
                        pass
        return None

    usage = _extract_usage(result)
    if not isinstance(usage, dict):
        return None, None, None

    prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion = usage.get("completion_tokens") or usage.get("output_tokens")
    total = usage.get("total_tokens")

    if not any(isinstance(x, int) for x in (prompt, completion, total)):
        return None, None, None

    in_t = int(prompt) if isinstance(prompt, int) else None
    out_t = int(completion) if isinstance(completion, int) else None
    tot_t = int(total) if isinstance(total, int) else None

    if tot_t is None and in_t is not None and out_t is not None:
        tot_t = in_t + out_t

    return in_t, out_t, tot_t


def _build_usage_dict(
    provider_usage: tuple[int | None, int | None, int | None],
    estimated_in: int,
    estimated_out: int,
) -> dict[str, int]:
    in_u, out_u, tot_u = provider_usage
    input_tokens = in_u if in_u is not None else estimated_in
    output_tokens = out_u if out_u is not None else estimated_out
    total_tokens = (
        tot_u if tot_u is not None else input_tokens + output_tokens
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _get_lang_from_path(file_path: Any) -> str:
    if not isinstance(file_path, str) or "." not in file_path:
        return ""
    ext = file_path.rsplit(".", 1)[-1].lower()
    mapping = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "java": "java",
        "go": "go",
        "rs": "rust",
        "cs": "csharp",
        "cpp": "cpp",
        "c": "c",
        "h": "c",
        "html": "html",
        "css": "css",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        "xml": "xml",
        "sh": "bash",
        "md": "markdown"
    }
    return mapping.get(ext, ext)


def _render_evidence_item(idx: int, item: dict[str, Any], lines: list[str]) -> None:
    question = item.get("question") or f"Finding {idx}"
    answer = item.get("answer") or ""
    code_ref = item.get("code_reference")
    ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else None

    lines.append(f"## {idx}. {question}")
    lines.append("")
    if code_ref:
        lines.append(f"**Code reference**: `{code_ref}`")
        lines.append("")
    if ev:
        ev_file = ev.get("file")
        ev_lines = ev.get("line_range")
        ev_type = ev.get("type")
        if ev_file or ev_lines or ev_type:
            parts: list[str] = []
            if ev_file:
                parts.append(f"file `{ev_file}`")
            if ev_lines:
                parts.append(f"lines {ev_lines}")
            if ev_type:
                parts.append(f"type {ev_type}")
            lines.append("**Evidence**: " + ", ".join(parts))
            lines.append("")
        snippet = ev.get("snippet")
        if snippet:
            lang = _get_lang_from_path(ev_file)
            lines.append(f"```{lang}")
            lines.append(str(snippet))
            lines.append("```")
            lines.append("")
        interp = ev.get("interpretation")
        if interp:
            lines.append(f"**Interpretation**: {interp}")
            lines.append("")

    if answer:
        lines.append("**Answer:**")
        lines.append(answer)
        lines.append("")


def evidence_to_markdown(evidence_json: dict[str, Any]) -> str:
    """
    Render the evidence section as a human-readable Markdown document.
    """
    items = evidence_json.get("items")
    if not isinstance(items, list) or not items:
        return ""

    lines: list[str] = ["# Evidence Pack", ""]

    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            _render_evidence_item(idx, item, lines)

    return "\n".join(lines).strip()


def scoring_to_markdown(scoring_json: dict[str, Any]) -> str:
    """
    Render the scoring section as a human-readable Markdown document.
    """
    findings = scoring_json.get("findings")
    if not isinstance(findings, list) or not findings:
        return ""

    lines: list[str] = ["# Scoring Summary", ""]

    for idx, finding in enumerate(findings, start=1):
        analysis = finding.get("analysis") if isinstance(finding, dict) else None
        if not isinstance(analysis, dict):
            continue

        verdict = analysis.get("verdict")
        score = analysis.get("score")
        explanation = analysis.get("explanation")
        breakdown = (
            analysis.get("scoring_breakdown")
            if isinstance(analysis.get("scoring_breakdown"), dict)
            else {}
        )

        title = verdict or f"Finding {idx}"
        if isinstance(score, (int, float)):
            title = f"{title} (score: {score})"

        lines.append(f"## {idx}. {title}")
        lines.append("")

        if explanation:
            lines.append("**Explanation:**")
            lines.append(explanation)
            lines.append("")

        if breakdown:
            lines.append("**Scoring breakdown:**")
            for k, v in breakdown.items():
                if not isinstance(v, dict):
                    continue
                part_score = v.get("score")
                reason = v.get("reason")
                label = k.replace("_", " ").title()
                if part_score is not None:
                    lines.append(f"- **{label}** ({part_score}): {reason}")
                else:
                    lines.append(f"- **{label}**: {reason}")
            lines.append("")

    return "\n".join(lines).strip()


def _format_remediation_heading(idx: int, rem: dict[str, Any]) -> str:
    title = f"Finding {idx}"
    priority = rem.get("priority")
    effort = rem.get("effort")

    heading_parts = [f"## {title}"]
    if priority or effort:
        meta: list[str] = []
        if priority:
            meta.append(f"**Priority**: {priority}")
        if effort:
            meta.append(f"**Effort**: {effort}")
        heading_parts.append(" — " + " | ".join(meta))
    return "".join(heading_parts)


def _format_primary_fix(primary: dict[str, Any], lines: list[str]) -> None:
    desc = primary.get("description")
    if desc:
        lines.append(f"**Description:** {desc}")
        lines.append("")

    before = primary.get("before")
    after = primary.get("after")
    file_path = primary.get("file")

    if file_path:
        lines.append(f"**File**: `{file_path}`")
        lines.append("")

    if before:
        lang = _get_lang_from_path(file_path)
        lines.append("**Before**:")
        lines.append(f"```{lang}")
        lines.append(str(before))
        lines.append("```")
        lines.append("")

    if after:
        lang = _get_lang_from_path(file_path)
        lines.append("**After**:")
        lines.append(f"```{lang}")
        lines.append(str(after))
        lines.append("```")
        lines.append("")


def _format_secondary_fixes(secondary: Any, lines: list[str]) -> None:
    if not isinstance(secondary, list) or not secondary:
        return
    lines.append("**Secondary fixes:**")
    for sec in secondary:
        if not isinstance(sec, dict):
            continue
        desc = sec.get("description")
        sec_file = sec.get("file")
        lines.append(f"- {desc}" + (f" (`{sec_file}`)" if sec_file else ""))
    lines.append("")


def _format_list_section(title: str, items: Any, lines: list[str], tick: bool = False) -> None:
    if not isinstance(items, list) or not items:
        return
    lines.append(f"**{title}:**")
    for item in items:
        if tick:
            lines.append(f"- `{item}`")
        else:
            lines.append(f"- {item}")
    lines.append("")


def remediation_to_markdown(remediation_json: dict[str, Any]) -> str:
    """
    Render the remediation section as a human-readable Markdown document.
    """
    findings = remediation_json.get("findings")
    if not isinstance(findings, list) or not findings:
        return ""

    lines: list[str] = ["# Remediation Plan", ""]

    for idx, finding in enumerate(findings, start=1):
        rem = finding.get("remediation")
        if not isinstance(rem, dict):
            continue

        lines.append(_format_remediation_heading(idx, rem))
        lines.append("")

        _format_primary_fix(rem.get("primary_fix") or {}, lines)
        _format_secondary_fixes(rem.get("secondary_fixes"), lines)
        _format_list_section("Verification steps", rem.get("verification_steps"), lines)
        _format_list_section("References", rem.get("references"), lines, tick=True)

    return "\n".join(lines).strip()


class ChatOrchestratorService:
    """Service class to encapsulate the RAG agent sequence and business logic."""

    @classmethod
    def _get_cache_key(cls, request_query: dict, target_repo_path: str) -> str:
        query_str = json.dumps(request_query, sort_keys=True)
        state_hash = _compute_repo_state_hash(target_repo_path)
        key_material = f"{query_str}|{state_hash}"
        return hashlib.md5(key_material.encode("utf-8")).hexdigest()


    @classmethod
    async def _log_stage_prompts(
        cls,
        *,
        agent: Any,
        default_system_prompt: str,
        user_payload: str,
        tool_call_id: str,
    ) -> None:
        """Helper to log system and user prompts for a stage."""
        system_prompt = getattr(agent, "system_prompt", default_system_prompt)
        if not isinstance(system_prompt, str):
            system_prompt = default_system_prompt

        await observability_hook.log_message(
            actor="system prompt", content=system_prompt, tool_call_id=tool_call_id
        )
        await observability_hook.log_message(
            actor="user prompt", content=user_payload, tool_call_id=tool_call_id
        )

    @classmethod
    async def _log_assistant_llm_output(
        cls,
        *,
        stage: str,
        raw_text: str,
        tool_call_id: str,
    ) -> None:
        """Emit one ai.message.created per LLM completion so monitors can filter by stage."""
        await observability_hook.log_message(
            actor="assistant",
            content=raw_text,
            tool_call_id=tool_call_id,
        )

    @classmethod
    async def _run_evidence_stage(
        cls,
        *,
        run_id: str,
        tool_call_id: str,
        query_payload: str,
        evidence_agent: Any,
        cache_key: str,
        target_repo_path: str,
        repo_state_hash: str,
    ) -> tuple[dict[str, Any], dict[str, int], int]:
        message_history = []
        deferred_results = None
        evidence_usage_from_provider: tuple[int | None, int | None, int | None] = (None, None, None)
        evidence_in_tokens = count_tokens(API_EVIDENCE_PROMPT) + count_tokens(query_payload)

        # (H) Log system and user prompts for evidence stage
        await cls._log_stage_prompts(
            agent=evidence_agent,
            default_system_prompt=API_EVIDENCE_PROMPT,
            user_payload=query_payload,
            tool_call_id=tool_call_id,
        )

        async def _run_once() -> dict[str, Any]:
            nonlocal deferred_results, message_history, evidence_usage_from_provider
            while True:
                result = await evidence_agent.run(
                    query_payload,
                    message_history=message_history,
                    deferred_tool_results=deferred_results,
                )

                if isinstance(result.output, DeferredToolRequests):
                    deferred_results = DeferredToolResults()
                    for call in result.output.approvals:
                        deferred_results.approvals[call.tool_call_id] = True
                    message_history.extend(result.new_messages())
                    continue

                if not isinstance(result.output, str):
                    raise ChatStageError(
                        code="LLM_INVALID_OUTPUT_TYPE",
                        message=f"Unexpected evidence response format: {type(result.output)}",
                        run_id=run_id,
                    )

                await cls._log_assistant_llm_output(
                    stage="evidence",
                    raw_text=result.output,
                    tool_call_id=tool_call_id,
                )
                evidence_usage_from_provider = _usage_from_result(result)
                return _parse_and_validate_stage(
                    stage=PipelineStage.EVIDENCE,
                    raw_text=result.output,
                    run_id=run_id,
                )

        evidence_timeout = float(settings.CHAT_EVIDENCE_TIMEOUT_SECONDS)
        evidence_attempts = max(1, int(settings.CHAT_SCHEMA_RETRY_ATTEMPTS))

        evidence_json = None
        evidence_ms = 0
        for attempt in range(evidence_attempts):
            t_stage = time.perf_counter()
            try:
                evidence_json = await _run_with_timeout(
                    _run_once(),
                    timeout_s=evidence_timeout,
                    stage="evidence",
                    run_id=run_id,
                )
                evidence_ms = int((time.perf_counter() - t_stage) * 1000)
                break
            except ChatStageError:
                if attempt >= evidence_attempts - 1:
                    raise
                deferred_results = None
                message_history = []

        assert evidence_json is not None
        evidence_items = evidence_json.get("items", [])
        evidence_out_tokens = count_tokens(json.dumps(evidence_json, ensure_ascii=False))
        evidence_usage = _build_usage_dict(
            evidence_usage_from_provider,
            evidence_in_tokens,
            evidence_out_tokens,
        )

        # Emit tool usage with best-effort token counts (provider or fallback).
        await observability_hook.log_llm_usage(
            tool_name="evidence",
            tool_call_id=tool_call_id,
            model_name=settings.active_orchestrator_config.model_id,
            input_tokens=evidence_usage.get("input_tokens", 0),
            output_tokens=evidence_usage.get("output_tokens", 0),
            duration_ms=evidence_ms or None,
        )

        _persist_stage(
            run_id=run_id,
            cache_key=cache_key,
            repo_path=target_repo_path,
            repo_state_hash=repo_state_hash,
            stage="evidence",
            tool_input={"items": evidence_items},
            tool_output={"items": evidence_items},
        )
        return evidence_json, evidence_usage, evidence_ms

    @classmethod
    async def _run_scoring_stage(
        cls,
        *,
        run_id: str,
        tool_call_id: str,
        shared_payload: str,
        timeout: float,
    ) -> tuple[dict[str, Any], tuple[int | None, int | None, int | None]]:
        scoring_agent = create_rag_orchestrator(tools=[], system_prompt=API_SCORING_PROMPT)

        async def _run_once() -> dict[str, Any]:
            # (H) Log system and user prompts for scoring stage
            await cls._log_stage_prompts(
                agent=scoring_agent,
                default_system_prompt=API_SCORING_PROMPT,
                user_payload=shared_payload,
                tool_call_id=tool_call_id,
            )
            result = await scoring_agent.run(shared_payload)
            if not isinstance(result.output, str):
                raise ChatStageError(
                    code="LLM_INVALID_OUTPUT_TYPE",
                    message=f"Unexpected scoring response format: {type(result.output)}",
                    run_id=run_id,
                )
            await cls._log_assistant_llm_output(
                stage="scoring",
                raw_text=result.output,
                tool_call_id=tool_call_id,
            )
            return result, _parse_and_validate_stage(
                stage=PipelineStage.SCORING,
                raw_text=result.output,
                run_id=run_id,
            )

        result, scoring_json = await _run_with_timeout(
            _run_once(),
            timeout_s=timeout,
            stage="scoring",
            run_id=run_id,
        )
        return scoring_json, _usage_from_result(result)

    @classmethod
    async def _run_remediation_stage(
        cls,
        *,
        run_id: str,
        tool_call_id: str,
        shared_payload: str,
        timeout: float,
    ) -> tuple[dict[str, Any], tuple[int | None, int | None, int | None]]:
        remediation_agent = create_rag_orchestrator(
            tools=[], system_prompt=API_REMEDIATION_PROMPT
        )

        async def _run_once() -> dict[str, Any]:
            # (H) Log system and user prompts for remediation stage
            await cls._log_stage_prompts(
                agent=remediation_agent,
                default_system_prompt=API_REMEDIATION_PROMPT,
                user_payload=shared_payload,
                tool_call_id=tool_call_id,
            )
            result = await remediation_agent.run(shared_payload)
            if not isinstance(result.output, str):
                raise ChatStageError(
                    code="LLM_INVALID_OUTPUT_TYPE",
                    message=f"Unexpected remediation response format: {type(result.output)}",
                    run_id=run_id,
                )
            await cls._log_assistant_llm_output(
                stage="remediation",
                raw_text=result.output,
                tool_call_id=tool_call_id,
            )
            return result, _parse_and_validate_stage(
                stage=PipelineStage.REMEDIATION,
                raw_text=result.output,
                run_id=run_id,
            )

        result, remediation_json = await _run_with_timeout(
            _run_once(),
            timeout_s=timeout,
            stage="remediation",
            run_id=run_id,
        )
        return remediation_json, _usage_from_result(result)

    @classmethod
    def _build_final_response(
        cls,
        *,
        invocation_id: str,
        evidence_json: dict[str, Any],
        evidence_ms: int,
        evidence_usage: dict[str, int],
        scoring_json: dict[str, Any],
        scoring_ms: int,
        scoring_usage: dict[str, int],
        remediation_json: dict[str, Any],
        remediation_ms: int,
        remediation_usage: dict[str, int],
    ) -> dict[str, Any]:
        models = {
            "orchestrator": {
                "provider": settings.active_orchestrator_config.provider,
                "model": settings.active_orchestrator_config.model_id,
            },
            "cypher": {
                "provider": settings.active_cypher_config.provider,
                "model": settings.active_cypher_config.model_id,
            },
        }

        return {
            "invocation_id": invocation_id,
            "evidence": {
                **evidence_json,
                "timings_ms": evidence_ms,
                "token_usage": evidence_usage,
                "markdown": evidence_to_markdown(evidence_json),
            },
            "scoring": {
                **scoring_json,
                "timings_ms": scoring_ms,
                "token_usage": scoring_usage,
                "markdown": scoring_to_markdown(scoring_json),
            },
            "remediation": {
                **remediation_json,
                "timings_ms": remediation_ms,
                "token_usage": remediation_usage,
                "markdown": remediation_to_markdown(remediation_json),
            },
            "models": models,
        }

    @classmethod
    def _prepare_context(
        cls, request_query: dict, target_repo_path: str, ingestor: Any, *, run_id: str
    ) -> tuple[str, str, str, str, Any]:
        """Prepare execution context, identifiers, and legacy payload normalization."""
        cache_key = cls._get_cache_key(request_query, target_repo_path)
        repo_state_hash = _compute_repo_state_hash(target_repo_path)

        evidence_agent, _ = initialize_services_and_agent(
            target_repo_path, ingestor, system_prompt=API_EVIDENCE_PROMPT
        )

        if isinstance(request_query, dict) and isinstance(request_query.get("findings"), list):
            findings_payload = request_query
        else:
            findings_payload = {"findings": [request_query]}

        query_payload = json.dumps(findings_payload, ensure_ascii=False)
        return run_id, cache_key, repo_state_hash, query_payload, evidence_agent

    @classmethod
    async def _run_parallel_stages(
        cls,
        *,
        run_id: str,
        scoring_tool_id: str,
        remediation_tool_id: str,
        evidence_json: dict[str, Any],
        cache_key: str,
        target_repo_path: str,
        repo_state_hash: str,
    ) -> dict[str, Any]:
        """Execute scoring and remediation in parallel with retry logic."""
        evidence_items = evidence_json.get("items", [])
        shared_input = {"findings": evidence_items}
        shared_payload = json.dumps(shared_input, ensure_ascii=False)

        scoring_timeout = float(settings.CHAT_SCORING_TIMEOUT_SECONDS)
        remediation_timeout = float(settings.CHAT_REMEDIATION_TIMEOUT_SECONDS)
        stage_attempts = max(1, int(settings.CHAT_SCHEMA_RETRY_ATTEMPTS))

        async def _timed(stage: str, coro):
            t_start = time.perf_counter()
            res = await coro
            return res, int((time.perf_counter() - t_start) * 1000)

        scoring_json = None
        remediation_json = None
        scoring_ms = 0
        remediation_ms = 0
        scoring_usage_provider = (None, None, None)
        remediation_usage_provider = (None, None, None)

        for attempt in range(stage_attempts):
            try:
                (
                    (scoring_json, scoring_usage_provider),
                    scoring_ms,
                ), (
                    (remediation_json, remediation_usage_provider),
                    remediation_ms,
                ) = await asyncio.gather(
                    _timed("scoring", cls._run_scoring_stage(run_id=run_id, tool_call_id=scoring_tool_id, shared_payload=shared_payload, timeout=scoring_timeout)),
                    _timed("remediation", cls._run_remediation_stage(run_id=run_id, tool_call_id=remediation_tool_id, shared_payload=shared_payload, timeout=remediation_timeout)),
                )
                break
            except ChatStageError:
                if attempt >= stage_attempts - 1:
                    raise

        # Persistence
        _persist_stage(
            run_id=run_id,
            cache_key=cache_key,
            repo_path=target_repo_path,
            repo_state_hash=repo_state_hash,
            stage="scoring",
            tool_input=shared_input,
            tool_output=scoring_json,
        )
        _persist_stage(
            run_id=run_id,
            cache_key=cache_key,
            repo_path=target_repo_path,
            repo_state_hash=repo_state_hash,
            stage="remediation",
            tool_input=shared_input,
            tool_output=remediation_json,
        )

        # Emit tool usage with best-effort token counts (provider or fallback).
        scoring_usage = _build_usage_dict(
            scoring_usage_provider,
            count_tokens(API_SCORING_PROMPT) + count_tokens(shared_payload),
            count_tokens(json.dumps(scoring_json, ensure_ascii=False)),
        )
        remediation_usage = _build_usage_dict(
            remediation_usage_provider,
            count_tokens(API_REMEDIATION_PROMPT) + count_tokens(shared_payload),
            count_tokens(json.dumps(remediation_json, ensure_ascii=False)),
        )

        await observability_hook.log_llm_usage(
            tool_name="scoring",
            tool_call_id=scoring_tool_id,
            model_name=settings.active_orchestrator_config.model_id,
            input_tokens=scoring_usage.get("input_tokens", 0),
            output_tokens=scoring_usage.get("output_tokens", 0),
            duration_ms=scoring_ms or None,
        )
        await observability_hook.log_llm_usage(
            tool_name="remediation",
            tool_call_id=remediation_tool_id,
            model_name=settings.active_orchestrator_config.model_id,
            input_tokens=remediation_usage.get("input_tokens", 0),
            output_tokens=remediation_usage.get("output_tokens", 0),
            duration_ms=remediation_ms or None,
        )

        return {
            "scoring_json": scoring_json,
            "scoring_ms": scoring_ms,
            "scoring_usage_provider": scoring_usage_provider,
            "remediation_json": remediation_json,
            "remediation_ms": remediation_ms,
            "remediation_usage_provider": remediation_usage_provider,
            "shared_payload": shared_payload,
        }

    @classmethod
    def _finalize_orchestration_response(
        cls,
        *,
        run_id: str,
        evidence_json: dict[str, Any],
        evidence_ms: int,
        evidence_usage: dict[str, Any],
        scoring_json: dict[str, Any],
        scoring_ms: int,
        scoring_usage_provider: tuple[int | None, int | None, int | None],
        remediation_json: dict[str, Any],
        remediation_ms: int,
        remediation_usage_provider: tuple[int | None, int | None, int | None],
        shared_payload: str,
    ) -> dict[str, Any]:
        """Aggregate all stage data and compute final usage/response."""
        scoring_usage = _build_usage_dict(
            scoring_usage_provider,
            count_tokens(API_SCORING_PROMPT) + count_tokens(shared_payload),
            count_tokens(json.dumps(scoring_json, ensure_ascii=False)),
        )
        remediation_usage = _build_usage_dict(
            remediation_usage_provider,
            count_tokens(API_REMEDIATION_PROMPT) + count_tokens(shared_payload),
            count_tokens(json.dumps(remediation_json, ensure_ascii=False)),
        )

        return cls._build_final_response(
            invocation_id=run_id,
            evidence_json=evidence_json,
            evidence_ms=evidence_ms,
            evidence_usage=evidence_usage,
            scoring_json=scoring_json,
            scoring_ms=scoring_ms,
            scoring_usage=scoring_usage,
            remediation_json=remediation_json,
            remediation_ms=remediation_ms,
            remediation_usage=remediation_usage,
        )

    @classmethod
    async def process_query(
        cls, request_query: dict, ingestor: Any, target_repo_path: str, org_id: str, invocation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Executes the logic to extract an evidence pack, score it, and propose remediation.
        """
        run_id = await observability_hook.before_chat(org_id=org_id, invocation_id=invocation_id)
        try:
            # 1. Setup Context & Evidence
            run_id, cache_key, repo_state_hash, query_payload, evidence_agent = cls._prepare_context(
                request_query, target_repo_path, ingestor, run_id=run_id
            )

            # Distinct UUIDs per stage for observability (ai.message.created, tool usage, etc.)
            evidence_tool_id = str(uuid4())
            scoring_tool_id = str(uuid4())
            remediation_tool_id = str(uuid4())

            evidence_json, evidence_usage, evidence_ms = await cls._run_evidence_stage(
                run_id=run_id,
                tool_call_id=evidence_tool_id,
                query_payload=query_payload,
                evidence_agent=evidence_agent,
                cache_key=cache_key,
                target_repo_path=target_repo_path,
                repo_state_hash=repo_state_hash,
            )

            # 2. Parallel Stages (Scoring & Remediation)
            parallel_results = await cls._run_parallel_stages(
                run_id=run_id,
                scoring_tool_id=scoring_tool_id,
                remediation_tool_id=remediation_tool_id,
                evidence_json=evidence_json,
                cache_key=cache_key,
                target_repo_path=target_repo_path,
                repo_state_hash=repo_state_hash,
            )

            # 3. Finalize
            final_resp = cls._finalize_orchestration_response(
                run_id=run_id,
                evidence_json=evidence_json,
                evidence_ms=evidence_ms,
                evidence_usage=evidence_usage,
                **parallel_results,
            )

            await observability_hook.log_message(
                actor="assistant",
                content=json.dumps(final_resp, default=str),
                tool_call_id=run_id,
            )
            await observability_hook.after_chat_success(tool_call_id=run_id)
            return final_resp
        except Exception as e:
            await observability_hook.after_chat_error(e, tool_call_id=run_id)
            raise
