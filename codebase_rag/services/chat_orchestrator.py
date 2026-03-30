import asyncio
import hashlib
import json
import time
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
    # Common patterns: result.usage (dict or object), result.model_response.usage (dict)
    candidates: list[Any] = []
    for attr in ("usage", "model_response", "response", "raw_response"):
        if hasattr(result, attr):
            candidates.append(getattr(result, attr))
    candidates.append(result)

    def _as_dict(v: Any) -> dict[str, Any] | None:
        if isinstance(v, dict):
            return v
        if hasattr(v, "usage") and isinstance(getattr(v, "usage"), dict):
            return getattr(v, "usage")
        if hasattr(v, "model_dump"):
            try:
                dumped = v.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                return None
        return None

    for c in candidates:
        d = _as_dict(c)
        if not d:
            continue

        usage = d.get("usage") if isinstance(d.get("usage"), dict) else d
        if not isinstance(usage, dict):
            continue

        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        total = usage.get("total_tokens")
        if isinstance(prompt, int) or isinstance(completion, int) or isinstance(total, int):
            in_t = int(prompt) if isinstance(prompt, int) else None
            out_t = int(completion) if isinstance(completion, int) else None
            tot_t = int(total) if isinstance(total, int) else None
            if tot_t is None and in_t is not None and out_t is not None:
                tot_t = in_t + out_t
            return in_t, out_t, tot_t

    return None, None, None


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


def evidence_to_markdown(evidence_json: dict[str, Any]) -> str:
    """
    Render the evidence section as a human-readable Markdown document.
    """
    items = evidence_json.get("items")
    if not isinstance(items, list) or not items:
        return ""

    lines: list[str] = ["# Evidence Pack", ""]

    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
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
    async def process_query(
        cls, request_query: dict, ingestor: Any, target_repo_path: str
    ) -> dict[str, Any]:
        """
        Executes the logic to extract an evidence pack, score it, and propose remediation.

        Args:
            request_query: The incoming query data payload (e.g. findings).
            ingestor: The Memgraph database ingestor instance.
            target_repo_path: Absolute path to the repository being analyzed.

        Returns:
            Dictionary containing 'evidence', 'scoring', 'remediation', or an 'error'.
        """
        cache_key = cls._get_cache_key(request_query, target_repo_path)

        repo_state_hash = _compute_repo_state_hash(target_repo_path)

        run_id = new_run_id()

        evidence_agent, _ = initialize_services_and_agent(
            target_repo_path, ingestor, system_prompt=API_EVIDENCE_PROMPT
        )

        message_history = []
        deferred_results = None

        if isinstance(request_query, dict) and isinstance(
            request_query.get("findings"), list
        ):
            findings_payload = request_query
        else:
            findings_payload = {"findings": [request_query]}

        query_payload = json.dumps(findings_payload, ensure_ascii=False)
        evidence_in_tokens = count_tokens(API_EVIDENCE_PROMPT) + count_tokens(query_payload)
        evidence_usage_from_provider: tuple[int | None, int | None, int | None] = (
            None,
            None,
            None,
        )

        async def _run_evidence_once() -> dict[str, Any]:
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

                evidence_usage_from_provider = _usage_from_result(result)
                return _parse_and_validate_stage(
                    stage=PipelineStage.EVIDENCE,
                    raw_text=result.output,
                    run_id=run_id,
                )

        evidence_timeout = float(settings.CHAT_EVIDENCE_TIMEOUT_SECONDS)
        evidence_attempts = max(1, int(settings.CHAT_SCHEMA_RETRY_ATTEMPTS))

        evidence_json: dict[str, Any] | None = None
        evidence_ms = 0
        for attempt in range(evidence_attempts):
            t_stage = time.perf_counter()
            try:
                evidence_json = await _run_with_timeout(
                    _run_evidence_once(),
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
        # Flattened evidence: items[] instead of nested findings/justification/Q&A.
        evidence_input = {"items": evidence_items}
        evidence_output = {"items": evidence_items}
        _persist_stage(
            run_id=run_id,
            cache_key=cache_key,
            repo_path=target_repo_path,
            repo_state_hash=repo_state_hash,
            stage="evidence",
            tool_input=evidence_input,
            tool_output=evidence_output,
        )

        shared_input = {"findings": evidence_items}
        shared_payload = json.dumps(shared_input, ensure_ascii=False)
        scoring_in_tokens = count_tokens(API_SCORING_PROMPT) + count_tokens(shared_payload)
        remediation_in_tokens = count_tokens(API_REMEDIATION_PROMPT) + count_tokens(shared_payload)
        scoring_usage_from_provider: tuple[int | None, int | None, int | None] = (
            None,
            None,
            None,
        )
        remediation_usage_from_provider: tuple[int | None, int | None, int | None] = (
            None,
            None,
            None,
        )

        scoring_agent = create_rag_orchestrator(tools=[], system_prompt=API_SCORING_PROMPT)
        remediation_agent = create_rag_orchestrator(
            tools=[], system_prompt=API_REMEDIATION_PROMPT
        )

        async def _run_scoring_once() -> dict[str, Any]:
            nonlocal scoring_usage_from_provider
            result = await scoring_agent.run(shared_payload)
            if not isinstance(result.output, str):
                raise ChatStageError(
                    code="LLM_INVALID_OUTPUT_TYPE",
                    message=f"Unexpected scoring response format: {type(result.output)}",
                    run_id=run_id,
                )
            scoring_usage_from_provider = _usage_from_result(result)
            return _parse_and_validate_stage(
                stage=PipelineStage.SCORING,
                raw_text=result.output,
                run_id=run_id,
            )

        async def _run_remediation_once() -> dict[str, Any]:
            nonlocal remediation_usage_from_provider
            result = await remediation_agent.run(shared_payload)
            if not isinstance(result.output, str):
                raise ChatStageError(
                    code="LLM_INVALID_OUTPUT_TYPE",
                    message=f"Unexpected remediation response format: {type(result.output)}",
                    run_id=run_id,
                )
            remediation_usage_from_provider = _usage_from_result(result)
            return _parse_and_validate_stage(
                stage=PipelineStage.REMEDIATION,
                raw_text=result.output,
                run_id=run_id,
            )

        scoring_timeout = float(settings.CHAT_SCORING_TIMEOUT_SECONDS)
        remediation_timeout = float(settings.CHAT_REMEDIATION_TIMEOUT_SECONDS)
        stage_attempts = max(1, int(settings.CHAT_SCHEMA_RETRY_ATTEMPTS))

        async def _timed(stage: str, coro):
            t_stage = time.perf_counter()
            result = await coro
            return result, int((time.perf_counter() - t_stage) * 1000)

        scoring_json: dict[str, Any] | None = None
        remediation_json: dict[str, Any] | None = None
        scoring_ms = 0
        remediation_ms = 0
        for attempt in range(stage_attempts):
            try:
                (scoring_json, scoring_ms), (remediation_json, remediation_ms) = await asyncio.gather(
                    _timed(
                        "scoring",
                        _run_with_timeout(
                            _run_scoring_once(),
                            timeout_s=scoring_timeout,
                            stage="scoring",
                            run_id=run_id,
                        ),
                    ),
                    _timed(
                        "remediation",
                        _run_with_timeout(
                            _run_remediation_once(),
                            timeout_s=remediation_timeout,
                            stage="remediation",
                            run_id=run_id,
                        ),
                    ),
                )
                break
            except ChatStageError:
                if attempt >= stage_attempts - 1:
                    raise

        assert scoring_json is not None
        assert remediation_json is not None

        scoring_out_tokens = count_tokens(json.dumps(scoring_json, ensure_ascii=False))
        remediation_out_tokens = count_tokens(json.dumps(remediation_json, ensure_ascii=False))

        scoring_usage = _build_usage_dict(
            scoring_usage_from_provider,
            scoring_in_tokens,
            scoring_out_tokens,
        )
        remediation_usage = _build_usage_dict(
            remediation_usage_from_provider,
            remediation_in_tokens,
            remediation_out_tokens,
        )

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
            "run_id": run_id,
            "evidence": {
                **evidence_output,
                "timings_ms": evidence_ms,
                "token_usage": evidence_usage,
                "markdown": evidence_to_markdown(evidence_output),
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
