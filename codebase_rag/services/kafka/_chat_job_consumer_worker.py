from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

from loguru import logger

from ...config import settings
from ...main import update_model_settings
from ...request_context import org_id_context
from ...services.chat_batch_service import process_chat_for_findings_ids
from ...services.chat_orchestrator import ChatStageError
from ...services.index_guard import (
    RepoIndexQueryError,
    RepoNotIndexedError,
    assert_repo_indexed,
)
from ...utils.org_tool_finding_store import persist_org_tool_finding_scores
from .chat_job_payload import ChatJobPayloadV1


def _resolve_repo_path(repo_path: str | None) -> str:
    resolved = repo_path or settings.TARGET_REPO_PATH
    if not resolved or not str(resolved).strip():
        raise ValueError("repo_path is required in payload or TARGET_REPO_PATH")
    return os.path.abspath(resolved)


def _validate_models_for_payload(payload: ChatJobPayloadV1) -> None:
    if payload.orchestrator or payload.cypher:
        update_model_settings(payload.orchestrator, payload.cypher)
    settings.active_orchestrator_config.validate_api_key("orchestrator")
    settings.active_cypher_config.validate_api_key("cypher")


def _persist_scores_safe(org_id: str, scored_findings: list[Any], invocation: str) -> None:
    try:
        persist_org_tool_finding_scores(
            org_id=org_id,
            scored_findings=scored_findings,
        )
    except Exception as e:
        logger.warning(
            "Kafka chat job {}: persist scores failed: {}",
            invocation,
            e,
        )


def _build_markdown_preview(response_data: dict[str, Any]) -> str:
    evidence_md = (
        response_data.get("evidence", {}).get("markdown") or ""
        if isinstance(response_data.get("evidence"), dict)
        else ""
    )
    scoring_md = (
        response_data.get("scoring", {}).get("markdown") or ""
        if isinstance(response_data.get("scoring"), dict)
        else ""
    )
    remediation_md = (
        response_data.get("remediation", {}).get("markdown") or ""
        if isinstance(response_data.get("remediation"), dict)
        else ""
    )

    if any([evidence_md.strip(), scoring_md.strip(), remediation_md.strip()]):
        combined_md_parts = []
        if evidence_md.strip():
            combined_md_parts.append(evidence_md.strip())
        if scoring_md.strip():
            combined_md_parts.append(scoring_md.strip())
        if remediation_md.strip():
            combined_md_parts.append(remediation_md.strip())
        return "\n\n".join(combined_md_parts)

    try:
        return json.dumps(
            response_data,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        return str(response_data)


async def _execute_chat_job_pipeline(
    payload: ChatJobPayloadV1,
    ingestor: Any,
    target_repo_path: str,
    invocation: str,
) -> None:
    await assert_repo_indexed(ingestor=ingestor, target_repo_path=target_repo_path)
    response_data, scored_findings = await process_chat_for_findings_ids(
        org_id=payload.org_id,
        org_tool_findings_ids=payload.org_tool_findings_ids,
        ingestor=ingestor,
        target_repo_path=target_repo_path,
    )
    response_data["run_id"] = str(uuid4())

    _persist_scores_safe(payload.org_id, scored_findings, invocation)

    response_preview = _build_markdown_preview(response_data)

    logger.info(
        "Kafka chat job completed invocation_id={} org_id={} findings={} response_md=\n{}",
        invocation,
        payload.org_id,
        len(payload.org_tool_findings_ids),
        response_preview,
    )


async def process_chat_job_message(
    *,
    payload: ChatJobPayloadV1,
    ingestor: Any,
    key: str | None,
) -> bool:
    """
    Returns True if the message offset may be committed (success or poison skip).
    Returns False to leave the offset uncommitted (retry after rebalance/restart).
    """
    invocation = payload.invocation_id or key or str(uuid4())[:8]
    target_repo_path = _resolve_repo_path(payload.repo_path)

    try:
        _validate_models_for_payload(payload)
    except ValueError as e:
        logger.warning(
            "Kafka chat job {} org_id={}: model validation failed (skip): {}",
            invocation,
            payload.org_id,
            e,
        )
        return True

    ctx_token = org_id_context.set(payload.org_id)
    try:
        await _execute_chat_job_pipeline(
            payload=payload,
            ingestor=ingestor,
            target_repo_path=target_repo_path,
            invocation=invocation,
        )
        return True
    except RepoNotIndexedError as e:
        logger.warning(
            "Kafka chat job {} org_id={}: repo not indexed (retry): {}",
            invocation,
            payload.org_id,
            e,
        )
        return False
    except RepoIndexQueryError as e:
        logger.warning(
            "Kafka chat job {} org_id={}: index query error (retry): {}",
            invocation,
            payload.org_id,
            e,
        )
        return False
    except ChatStageError as e:
        logger.warning(
            "Kafka chat job {} org_id={}: pipeline error (retry): {} {}",
            invocation,
            payload.org_id,
            e.code,
            e.message,
        )
        return False
    except ValueError as e:
        logger.warning(
            "Kafka chat job {} org_id={}: validation error (skip): {}",
            invocation,
            payload.org_id,
            e,
        )
        return True
    except Exception:
        logger.exception(
            "Kafka chat job {} org_id={}: unexpected error (retry)",
            invocation,
            payload.org_id,
        )
        return False
    finally:
        org_id_context.reset(ctx_token)
