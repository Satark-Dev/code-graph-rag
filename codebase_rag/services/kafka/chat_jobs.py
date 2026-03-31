from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from loguru import logger

from ...config import settings
from .stage_job_payloads import EvidenceJobPayloadV1
from .producer import kafka_service


def get_chat_jobs_topic() -> str:
    # Stage-separated pipeline starts at evidence topic.
    return settings.KAFKA_EVIDENCE_JOBS_TOPIC


def get_chat_job_key(org_id: str) -> str:
    """Stable partition key: all jobs for one org serialize per partition when keyed."""
    return org_id.strip()


async def enqueue_chat_job(
    *,
    app: FastAPI,
    org_id: str,
    org_tool_findings_ids: list[str],
    invocation_id: str | None = None,
    repo_path: str | None = None,
    orchestrator: str | None = None,
    cypher: str | None = None,
    kafka_key: str | None = None,
) -> str:
    """Produce one chat job (same work as POST /api/chat). Returns invocation_id."""
    inv = invocation_id or uuid4().hex
    payload = EvidenceJobPayloadV1(
        org_id=org_id,
        org_tool_findings_ids=org_tool_findings_ids,
        invocation_id=inv,
        orchestrator=orchestrator,
        cypher=cypher,
    )
    topic = get_chat_jobs_topic()
    key = kafka_key or get_chat_job_key(org_id)

    await kafka_service.start()
    await kafka_service.send(topic, value=payload.model_dump(mode="json"), key=key)

    logger.debug(
        "Enqueued chat job topic={} org_id={} invocation_id={}",
        topic,
        org_id,
        inv,
    )
    return inv
