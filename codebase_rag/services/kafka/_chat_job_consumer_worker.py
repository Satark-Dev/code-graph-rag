from __future__ import annotations

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


async def process_chat_job_message(
    *,
    payload: ChatJobPayloadV1,
    ingestor: Any,
    key: str | None,
) -> bool:
    """
    Returns True if the message offset may be committed (success or poison skip).
    Returns False to leave the offset uncommitted (retry after rebalance/restart).

    Unlike repomind scoring, this worker does not use pg_try_advisory_lock: duplicate
    delivery is mitigated by org-scoped processing and DB upserts where applicable;
    add a DB lock + invocation_id idempotency row if you run many consumers per org.
    """
    invocation = payload.invocation_id or key or str(uuid4())[:8]
    target_repo_path = _resolve_repo_path(payload.repo_path)

    try:
        if payload.orchestrator or payload.cypher:
            update_model_settings(payload.orchestrator, payload.cypher)
        settings.active_orchestrator_config.validate_api_key("orchestrator")
        settings.active_cypher_config.validate_api_key("cypher")
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
        await assert_repo_indexed(
            ingestor=ingestor, target_repo_path=target_repo_path
        )
        response_data, scored_findings = await process_chat_for_findings_ids(
            org_id=payload.org_id,
            org_tool_findings_ids=payload.org_tool_findings_ids,
            ingestor=ingestor,
            target_repo_path=target_repo_path,
        )
        response_data["run_id"] = str(uuid4())

        try:
            persist_org_tool_finding_scores(
                org_id=payload.org_id,
                scored_findings=scored_findings,
            )
        except Exception as e:
            logger.warning(
                "Kafka chat job {}: persist scores failed: {}",
                invocation,
                e,
            )

        logger.info(
            "Kafka chat job completed invocation_id={} org_id={} findings={}",
            invocation,
            payload.org_id,
            len(payload.org_tool_findings_ids),
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
