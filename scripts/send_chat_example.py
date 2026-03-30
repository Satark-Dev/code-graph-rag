from __future__ import annotations

import asyncio
import uuid

from loguru import logger

from codebase_rag.services.kafka import get_chat_jobs_topic, kafka_service
from codebase_rag.services.kafka.chat_job_payload import ChatJobPayloadV1


async def send_chat_example() -> None:
    """Send a single example chat job to Kafka for end-to-end testing."""
    org_id = "3b393436-119f-45ca-8d53-842d7ec96771"
    findings = [
        "4395a501-c99c-43ac-b86c-0354eb36c6fd"
    ]
    invocation_id = str(uuid.uuid4())

    payload = ChatJobPayloadV1(
        org_id=org_id,
        org_tool_findings_ids=findings,
        invocation_id=invocation_id,
    )

    topic = get_chat_jobs_topic()
    key = org_id  # stable partitioning key by org

    logger.info(
        "Sending chat job topic={} key={} payload={}",
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
    asyncio.run(send_chat_example())
