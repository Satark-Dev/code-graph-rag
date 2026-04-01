from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

from ...config import settings
from ...main import update_model_settings
from ...observability.hook import observability_hook
from ...request_context import org_id_context
from ...services.chat_orchestrator import ChatOrchestratorService
from ...utils.org_tool_finding_store import persist_org_tool_finding_scores
from ...utils.tool_call_store import fetch_latest_stage_output
from ...utils.tool_call_store import store_tool_call
from ...utils.token_utils import count_tokens
from ...prompts import API_SCORING_PROMPT
from .stage_job_payloads import DownstreamStagePayloadV1
def _validate_models_for_payload(payload: DownstreamStagePayloadV1) -> None:
    if payload.orchestrator or payload.cypher:
        update_model_settings(payload.orchestrator, payload.cypher)
    settings.active_orchestrator_config.validate_api_key("orchestrator")
    settings.active_cypher_config.validate_api_key("cypher")


async def process_scoring_job_message(*, payload: DownstreamStagePayloadV1, ingestor: Any) -> bool:
    invocation = payload.invocation_id
    ctx_token = org_id_context.set(payload.org_id)
    t0 = time.perf_counter()
    try:
        await observability_hook.before_chat(org_id=payload.org_id, invocation_id=invocation)
        await observability_hook.log_tool_start(tool_name="scoring", tool_call_id=payload.tool_call_id)
        _validate_models_for_payload(payload)

        evidence_out = fetch_latest_stage_output(run_id=payload.invocation_id, stage="evidence")
        if not evidence_out or not isinstance(evidence_out, dict):
            logger.warning(
                "Kafka scoring job {} org_id={}: missing evidence stage output for invocation_id={} (retry)",
                invocation,
                payload.org_id,
                payload.invocation_id,
            )
            return False

        evidence_items = evidence_out.get("items", [])
        shared_input = {"findings": evidence_items}
        shared_payload = json.dumps(shared_input, ensure_ascii=False)

        scoring_json, _usage_provider = await ChatOrchestratorService._run_scoring_stage(
            run_id=payload.invocation_id,
            tool_call_id=payload.tool_call_id,
            shared_payload=shared_payload,
            timeout=float(settings.CHAT_SCORING_TIMEOUT_SECONDS),
        )
        scoring_ms = int((time.perf_counter() - t0) * 1000)

        # Persist scoring stage output for final aggregation.
        store_tool_call(
            run_id=payload.invocation_id,
            cache_key=payload.cache_key,
            repo_path=payload.target_repo_path,
            repo_state_hash=payload.repo_state_hash,
            stage="scoring",
            tool_input=shared_input,
            tool_output=scoring_json,
        )

        # Persist per-finding scores back to org_tool_findings.
        findings = scoring_json.get("findings") or []
        scored: list[tuple[str, float, str | None]] = []
        for idx, finding_id in enumerate(payload.org_tool_findings_ids):
            if idx >= len(findings):
                break
            raw_finding = findings[idx]
            analysis = raw_finding.get("analysis") if isinstance(raw_finding, dict) else None
            if not isinstance(analysis, dict):
                continue
            score = analysis.get("score")
            explanation = analysis.get("explanation")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            explanation_text = explanation if isinstance(explanation, str) else None
            scored.append((finding_id, score_value, explanation_text))

        if scored:
            updated = persist_org_tool_finding_scores(
                org_id=payload.org_id,
                scored_findings=tuple(scored),
            )
            logger.info(
                "Kafka scoring job {} org_id={}: persisted scores for {} finding(s)",
                invocation,
                payload.org_id,
                updated,
            )
        else:
            logger.warning(
                "Kafka scoring job {} org_id={}: no valid scoring entries to persist",
                invocation,
                payload.org_id,
            )

        # Emit usage using our deterministic token counter.
        input_tokens = count_tokens(API_SCORING_PROMPT) + count_tokens(shared_payload)
        output_tokens = count_tokens(json.dumps(scoring_json, ensure_ascii=False))
        await observability_hook.log_llm_usage(
            tool_name="scoring",
            tool_call_id=payload.tool_call_id,
            model_name=settings.active_orchestrator_config.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=scoring_ms,
        )
        return True
    except Exception:
        # Persist failure status so repo cleanup does not trigger.
        try:
            store_tool_call(
                run_id=payload.invocation_id,
                cache_key=payload.cache_key,
                repo_path=payload.target_repo_path,
                repo_state_hash=payload.repo_state_hash,
                stage="scoring",
                stage_status="error",
                error_message="scoring_failed",
                tool_input={},
                tool_output={},
            )
        except Exception:  # noqa: BLE001
            pass
        logger.exception(
            "Kafka scoring job {} org_id={}: unexpected error (retry)",
            invocation,
            payload.org_id,
        )
        await observability_hook.log_tool_failed(
            tool_name="scoring",
            tool_call_id=payload.tool_call_id,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
        return False
    finally:
        org_id_context.reset(ctx_token)

