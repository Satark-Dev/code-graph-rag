from __future__ import annotations

import asyncio

from loguru import logger

from codebase_rag.services.kafka import (
    get_index_job_key,
    get_index_jobs_topic,
    kafka_service,
)
from codebase_rag.services.kafka.index_job_payload import IndexJobPayload


async def send_index_example() -> None:
    """Send a single example index job to Kafka for end-to-end testing."""
    org_id = "3b393436-119f-45ca-8d53-842d7ec96771"
    repo_path = "/Users/infynnosolutions/Desktop/Code/JavaGoat"

    payload = IndexJobPayload(
        org_id=org_id,
        repo_path=repo_path,
        clean=True,
        exclude=None,
    )

    topic = get_index_jobs_topic()
    key = get_index_job_key(org_id, repo_path)

    logger.info(
        "Sending index job topic={} key={} payload={}",
        topic,
        key,
        payload.model_dump(mode="json"),
    )

    await kafka_service.start()
    try:
        await kafka_service.send(
            topic,
            value=payload.model_dump(mode="json"),
            key=key,
        )
    finally:
        await kafka_service.stop()


if __name__ == "__main__":
    asyncio.run(send_index_example())

