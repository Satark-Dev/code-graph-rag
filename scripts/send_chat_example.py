from __future__ import annotations

import asyncio
import uuid

from loguru import logger

from codebase_rag.services.kafka import get_chat_jobs_topic, kafka_service
from codebase_rag.services.kafka.stage_job_payloads import EvidenceJobPayloadV1


async def send_chat_example() -> None:
    """Send a single example chat job to Kafka for end-to-end testing.

    This only publishes to the chat topic; it does not enqueue an index job or clone repos.
    If you use ``cgr consumers`` with embedded Kafka, run with ``--no-kafka-index-consumer`` (or
    ``CGR_EMBEDDED_KAFKA_INDEX_CONSUMERS=0``) when you only want the chat worker and have
    already indexed via a separate index worker or an earlier run.
    """
    org_id = "3b393436-119f-45ca-8d53-842d7ec96771"
    findings = [
        "4395a501-c99c-43ac-b86c-0354eb36c6fd"
    ]
    invocation_id = str(uuid.uuid4())

    payload = EvidenceJobPayloadV1(
        org_id=org_id,
        org_tool_findings_ids=findings,
        invocation_id=invocation_id,
    )

    topic = get_chat_jobs_topic()
    key = invocation_id

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
