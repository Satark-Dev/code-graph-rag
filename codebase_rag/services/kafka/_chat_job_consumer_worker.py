from __future__ import annotations

import json
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
from ...utils.org_tool_finding_store import (
    get_chat_finding_metadata,
    persist_org_tool_finding_scores,
)
from .chat_job_payload import ChatJobPayloadV1
from .repo_manager import RepoManager


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


def _get_markdown_from_section(response_data: dict[str, Any], section: str) -> str:
    data = response_data.get(section)
    return data.get("markdown") or "" if isinstance(data, dict) else ""


def _build_markdown_preview(response_data: dict[str, Any]) -> str:
    sections = ["evidence", "scoring", "remediation"]
    markdown_parts = [
        md for section in sections if (md := _get_markdown_from_section(response_data, section).strip())
    ]

    if markdown_parts:
        return "\n\n".join(markdown_parts)

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
        invocation_id=invocation,
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
    repo_manager: RepoManager,
    key: str | None,
) -> bool:
    """
    Returns True if the message offset may be committed (success or poison skip).
    Returns False to leave the offset uncommitted (retry after rebalance/restart).
    """
    invocation = payload.invocation_id
    target_repo_path = None

    try:
        # Resolve repo URL and branch name from the first finding
        repo_url, branch_name = get_chat_finding_metadata(
            payload.org_tool_findings_ids[0],
            payload.org_id,
        )

        if not repo_url:
            logger.error(
                "Kafka chat job {}: could not resolve repo_url from finding {}",
                invocation,
                payload.org_tool_findings_ids[0],
            )
            return True  # Skip poison pill

        # Batch Validation: Ensure all findings belong to the same repo/branch
        for finding_id in payload.org_tool_findings_ids[1:]:
            curr_url, curr_branch = get_chat_finding_metadata(
                finding_id,
                payload.org_id,
            )
            if curr_url != repo_url or curr_branch != branch_name:
                logger.error(
                    "Kafka chat job {}: mixed repository context detected. "
                    "{} ({}) != {} ({})",
                    invocation,
                    repo_url,
                    branch_name,
                    curr_url,
                    curr_branch,
                )
                return True  # Skip poison pill

        # Ensuring repository exists in the deterministic process folder
        target_repo_path = await repo_manager.ensure_cloned(
            repo_url=repo_url,
            org_id=payload.org_id,
            branch=branch_name,
        )

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
