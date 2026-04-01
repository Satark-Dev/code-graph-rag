from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from loguru import logger

from ...config import settings
from ...main import update_model_settings
from ...observability.hook import observability_hook
from ...request_context import org_id_context
from ...services.chat_orchestrator import ChatOrchestratorService
from ...services.index_guard import RepoIndexQueryError, RepoNotIndexedError, assert_repo_indexed
from ...utils.org_tool_finding_store import get_branch_name_for_finding
from ...utils.tool_call_store import fetch_latest_stage_output
from .producer import kafka_service
from .repo_manager import RepoManager
from .stage_job_payloads import DownstreamStagePayloadV1, EvidenceJobPayloadV1


def _validate_models_for_payload(payload: EvidenceJobPayloadV1) -> None:
    if payload.orchestrator or payload.cypher:
        update_model_settings(payload.orchestrator, payload.cypher)
    settings.active_orchestrator_config.validate_api_key("orchestrator")
    settings.active_cypher_config.validate_api_key("cypher")


async def process_evidence_job_message(
    *,
    payload: EvidenceJobPayloadV1,
    ingestor: Any,
    repo_manager: RepoManager,
    key: str | None,
) -> bool:
    """
    Returns True if the offset may be committed (success or poison skip).
    Returns False to leave the offset uncommitted (retry).
    """
    invocation = payload.invocation_id
    t0 = time.perf_counter()
    evidence_tool_id: str | None = None

    ctx_token = org_id_context.set(payload.org_id)
    try:
        await observability_hook.before_chat(org_id=payload.org_id, invocation_id=invocation)
        # Wrapper tool so the UI can show a dedicated "scoring agent" invocation
        # containing evidence + scoring + remediation.
        await observability_hook.log_tool_start(tool_name="scoring_agent", tool_call_id=invocation)

        branch_name = get_branch_name_for_finding(payload.org_tool_findings_ids[0], payload.org_id)
        if not branch_name or not str(branch_name).strip():
            logger.error(
                "Kafka evidence job {}: could not resolve branch name from finding {} (org_id={})",
                invocation,
                payload.org_tool_findings_ids[0],
                payload.org_id,
            )
            return True
        branch_name = str(branch_name).strip()
        for finding_id in payload.org_tool_findings_ids[1:]:
            b = get_branch_name_for_finding(finding_id, payload.org_id)
            if not b or str(b).strip() != branch_name:
                logger.error(
                    "Kafka evidence job {}: mixed branch context: finding {} branch={!r} expected={!r}",
                    invocation,
                    finding_id,
                    b,
                    branch_name,
                )
                return True

        target_repo_path = repo_manager.require_existing_local_clone(
            org_id=payload.org_id, branch=branch_name
        )
        repo_lease_id = str(uuid4())
        await repo_manager.mark_in_use(repo_path=target_repo_path, lease_id=repo_lease_id)
        await assert_repo_indexed(ingestor=ingestor, target_repo_path=target_repo_path)
        _validate_models_for_payload(payload)

        # Prepare evidence context and run only evidence stage.
        request_query = {"findings": []}  # placeholder; ChatBatchService fetches finding payloads in old flow
        # In Kafka flow we want the same behavior as process_chat_for_findings_ids, so build the
        # exact query payload from fetched findings via ChatBatchService helpers would require DB.
        # Instead, reuse the existing per-finding fetch path by delegating to ChatBatchService is not possible here.
        # For now, keep the existing behavior by fetching findings via the findings client path:
        from ...services.findings_client import fetch_org_tool_finding

        findings_payloads = [
            await fetch_org_tool_finding(org_id=payload.org_id, org_tool_findings_id=fid)
            for fid in payload.org_tool_findings_ids
        ]
        request_query = {"findings": findings_payloads}

        # Use invocation_id as the stable run identifier across stages.
        run_id = invocation
        run_id, cache_key, repo_state_hash, query_payload, evidence_agent = ChatOrchestratorService._prepare_context(
            request_query, target_repo_path, ingestor, run_id=run_id
        )
        evidence_tool_id = str(uuid4())
        await observability_hook.log_tool_start(tool_name="evidence", tool_call_id=evidence_tool_id)
        evidence_json, evidence_usage, evidence_ms = await ChatOrchestratorService._run_evidence_stage(
            run_id=run_id,
            tool_call_id=evidence_tool_id,
            query_payload=query_payload,
            evidence_agent=evidence_agent,
            cache_key=cache_key,
            target_repo_path=target_repo_path,
            repo_state_hash=repo_state_hash,
        )

        # Safety: evidence stage must have persisted its output for downstream consumers.
        persisted = fetch_latest_stage_output(run_id=run_id, stage="evidence")
        if persisted is None:
            logger.error(
                "Kafka evidence job {}: evidence output was not persisted; cannot fan out scoring/remediation (invocation_id={})",
                invocation,
                run_id,
            )
            return False

        # Fan out to scoring + remediation topics.
        scoring_payload = DownstreamStagePayloadV1(
            org_id=payload.org_id,
            tool_call_id=str(uuid4()),
            cache_key=cache_key,
            repo_state_hash=repo_state_hash,
            target_repo_path=target_repo_path,
            invocation_id=invocation,
            repo_lease_id=repo_lease_id,
            org_tool_findings_ids=payload.org_tool_findings_ids,
            orchestrator=payload.orchestrator,
            cypher=payload.cypher,
        )
        remediation_payload = DownstreamStagePayloadV1(
            org_id=payload.org_id,
            tool_call_id=str(uuid4()),
            cache_key=cache_key,
            repo_state_hash=repo_state_hash,
            target_repo_path=target_repo_path,
            invocation_id=invocation,
            repo_lease_id=repo_lease_id,
            org_tool_findings_ids=payload.org_tool_findings_ids,
            orchestrator=payload.orchestrator,
            cypher=payload.cypher,
        )

        # Ensure producer is running (in serve lifespan it is; in standalone worker it's optional).
        await kafka_service.start()
        await kafka_service.send(
            settings.KAFKA_SCORING_JOBS_TOPIC,
            value=scoring_payload.model_dump(mode="json"),
            key=invocation,
        )
        await kafka_service.send(
            settings.KAFKA_REMEDIATION_JOBS_TOPIC,
            value=remediation_payload.model_dump(mode="json"),
            key=invocation,
        )

        logger.info(
            "Kafka evidence job completed invocation_id={} org_id={} findings={} evidence_tool_call_id={}",
            invocation,
            payload.org_id,
            len(payload.org_tool_findings_ids),
            evidence_tool_id,
        )
        return True
    except RepoNotIndexedError as e:
        logger.warning(
            "Kafka evidence job {} org_id={}: repo not indexed (retry): {}",
            invocation,
            payload.org_id,
            e,
        )
        if evidence_tool_id is not None:
            await observability_hook.log_tool_failed(
                tool_name="evidence",
                tool_call_id=evidence_tool_id,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        return False
    except RepoIndexQueryError as e:
        logger.warning(
            "Kafka evidence job {} org_id={}: index query error (retry): {}",
            invocation,
            payload.org_id,
            e,
        )
        if evidence_tool_id is not None:
            await observability_hook.log_tool_failed(
                tool_name="evidence",
                tool_call_id=evidence_tool_id,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        return False
    except ValueError as e:
        logger.warning(
            "Kafka evidence job {} org_id={}: validation/local repo missing (skip): {}",
            invocation,
            payload.org_id,
            e,
        )
        if evidence_tool_id is not None:
            await observability_hook.log_tool_failed(
                tool_name="evidence",
                tool_call_id=evidence_tool_id,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        return True
    except Exception:
        logger.exception(
            "Kafka evidence job {} org_id={}: unexpected error (retry)",
            invocation,
            payload.org_id,
        )
        if evidence_tool_id is not None:
            await observability_hook.log_tool_failed(
                tool_name="evidence",
                tool_call_id=evidence_tool_id,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        return False
    finally:
        org_id_context.reset(ctx_token)

