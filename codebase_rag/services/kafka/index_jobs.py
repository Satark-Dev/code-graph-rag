from __future__ import annotations

from fastapi import FastAPI
from loguru import logger

from ...config import settings
from .index_job_consumer_controller import ensure_index_job_consumer_started
from .index_job_payload import IndexJobPayload
from .producer import kafka_service


def get_index_jobs_topic() -> str:
    return settings.KAFKA_INDEX_JOBS_TOPIC


async def enqueue_index_job(
    *,
    app: FastAPI,
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
    # Use invocation_id as the Kafka message key so all stages for the same
    # pipeline run stay on the same partition.
    key = payload.invocation_id

    await kafka_service.start()
    await kafka_service.send(topic, value=payload.model_dump(mode="json"), key=key)

    await ensure_index_job_consumer_started(app)

    logger.info(
        "Enqueued index job topic={} org_id={} repo_path={} clean={}",
        topic,
        org_id,
        repo_path,
        clean,
    )

