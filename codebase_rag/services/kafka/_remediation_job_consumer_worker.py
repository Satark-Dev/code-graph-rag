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
from ...prompts import API_REMEDIATION_PROMPT
from ...utils.token_utils import count_tokens
from ...utils.tool_call_store import fetch_latest_stage_output, store_tool_call
from .stage_job_payloads import DownstreamStagePayloadV1
from ._pipeline_cleanup import release_repo_lease_if_unused


def _validate_models_for_payload(payload: DownstreamStagePayloadV1) -> None:
    if payload.orchestrator or payload.cypher:
        update_model_settings(payload.orchestrator, payload.cypher)
    settings.active_orchestrator_config.validate_api_key("orchestrator")
    settings.active_cypher_config.validate_api_key("cypher")


async def process_remediation_job_message(*, payload: DownstreamStagePayloadV1, ingestor: Any) -> bool:
    invocation = payload.invocation_id
    ctx_token = org_id_context.set(payload.org_id)
    t0 = time.perf_counter()
    try:
        await observability_hook.before_chat(org_id=payload.org_id, invocation_id=invocation)
        await observability_hook.log_tool_start(tool_name="remediation", tool_call_id=payload.tool_call_id)
        _validate_models_for_payload(payload)

        evidence_out = fetch_latest_stage_output(run_id=payload.invocation_id, stage="evidence")
        if not evidence_out or not isinstance(evidence_out, dict):
            logger.warning(
                "Kafka remediation job {} org_id={}: missing evidence stage output for invocation_id={} (retry)",
                invocation,
                payload.org_id,
                payload.invocation_id,
            )
            return False

        evidence_items = evidence_out.get("items", [])
        shared_input = {"findings": evidence_items}
        shared_payload = json.dumps(shared_input, ensure_ascii=False)

        remediation_json, _usage_provider = await ChatOrchestratorService._run_remediation_stage(
            run_id=payload.invocation_id,
            tool_call_id=payload.tool_call_id,
            shared_payload=shared_payload,
            timeout=float(settings.CHAT_REMEDIATION_TIMEOUT_SECONDS),
        )
        remediation_ms = int((time.perf_counter() - t0) * 1000)

        store_tool_call(
            run_id=payload.invocation_id,
            cache_key=payload.cache_key,
            repo_path=payload.target_repo_path,
            repo_state_hash=payload.repo_state_hash,
            stage="remediation",
            tool_input=shared_input,
            tool_output=remediation_json,
        )

        input_tokens = count_tokens(API_REMEDIATION_PROMPT) + count_tokens(shared_payload)
        output_tokens = count_tokens(json.dumps(remediation_json, ensure_ascii=False))
        await observability_hook.log_llm_usage(
            tool_name="remediation",
            tool_call_id=payload.tool_call_id,
            model_name=settings.active_orchestrator_config.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=remediation_ms,
        )
        # Mark the end of the full Kafka chat pipeline invocation (evidence + scoring + remediation).
        await observability_hook.after_chat_success(tool_call_id=payload.invocation_id)
        await release_repo_lease_if_unused(
            repo_path=payload.target_repo_path,
            repo_lease_id=payload.repo_lease_id,
        )
        logger.info(
            "Kafka remediation job {} org_id={} completed successfully",
            invocation,
            payload.org_id,
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
                stage="remediation",
                stage_status="error",
                error_message="remediation_failed",
                tool_input={},
                tool_output={},
            )
        except Exception:  # noqa: BLE001
            pass
        logger.exception(
            "Kafka remediation job {} org_id={}: unexpected error (retry)",
            invocation,
            payload.org_id,
        )
        await observability_hook.log_tool_failed(
            tool_name="remediation",
            tool_call_id=payload.tool_call_id,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
        await observability_hook.after_chat_error(
            RuntimeError("remediation_failed"),
            tool_call_id=payload.invocation_id,
        )
        return False
    finally:
        org_id_context.reset(ctx_token)

