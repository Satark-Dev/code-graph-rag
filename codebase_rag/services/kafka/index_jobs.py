from __future__ import annotations

from loguru import logger

from ...config import settings
from .index_job_payload import IndexJobPayload
from .producer import kafka_service


def get_index_jobs_topic() -> str:
    return settings.KAFKA_INDEX_JOBS_TOPIC


def get_index_job_key(
    org_id: str | None = None,
    repo_path: str | None = None,
    *,
    invocation_id: str | None = None,
) -> str:
    """
    Kafka partition key for index jobs.

    Preferred: invocation_id (keeps the full pipeline on one partition).
    Fallback: org_id::repo_path for older call sites.
    """
    if invocation_id and str(invocation_id).strip():
        return str(invocation_id).strip()
    if org_id is None or repo_path is None:
        raise ValueError("Either invocation_id or (org_id, repo_path) is required")
    return f"{org_id.strip()}::{repo_path.strip()}"


async def enqueue_index_job(
    *,
    org_id: str,
    repo_path: str,
    clean: bool,
    exclude: list[str] | None,
) -> None:
    """Produce one index job; returns immediately (work done by Kafka consumer)."""
    payload = IndexJobPayload(
        org_id=org_id,
        repo_path=repo_path,
        clean=clean,
        exclude=exclude,
    )
    topic = get_index_jobs_topic()
    key = get_index_job_key(invocation_id=payload.invocation_id)

    await kafka_service.start()
    await kafka_service.send(topic, value=payload.model_dump(mode="json"), key=key)

    logger.info(
        "Enqueued index job topic={} org_id={} repo_path={} clean={}",
        topic,
        org_id,
        repo_path,
        clean,
    )
