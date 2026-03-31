import asyncio
import uuid

from loguru import logger

from codebase_rag.services.kafka import (
    get_index_jobs_topic,
    kafka_service,
)
from codebase_rag.services.kafka.index_job_payload import IndexJobPayload


async def send_index_example() -> None:
    """Send a single example index job to Kafka for end-to-end testing."""
    org_id = "3b393436-119f-45ca-8d53-842d7ec96771"
    repo_url = "https://github.com/Sheru-Sarabhai-ni-Company/JavaGoat.git"
    org_finding_id = "4395a501-c99c-43ac-b86c-0354eb36c6fd"
    invocation_id = str(uuid.uuid4())

    payload = IndexJobPayload(
        org_id=org_id,
        repo_url=repo_url,
        org_tool_finding_id=org_finding_id,
        clean=True,
        exclude=None,
        invocation_id=invocation_id,
    )

    topic = get_index_jobs_topic()
    key = invocation_id

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
