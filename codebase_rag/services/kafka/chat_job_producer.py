from __future__ import annotations

import contextlib
from typing import Any

from ...config import settings
from .chat_job_payload import ChatJobPayloadV1
from .producer import kafka_service


async def send_chat_job(
    payload: ChatJobPayloadV1,
    *,
    producer: Any | None = None,
) -> None:
    """
    Send one chat job. If producer is None, uses KafkaService singleton (preferred).
    """
    data = payload.model_dump(mode="json", exclude_none=True)
    key = payload.org_id.encode("utf-8")
    topic = settings.KAFKA_CHAT_JOBS_TOPIC

    if producer is not None:
        import json

        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        await producer.send_and_wait(topic, value=raw, key=key)
        return

    await kafka_service.start()
    await kafka_service.send(topic, value=data, key=key)


async def build_chat_job_producer() -> Any:
    """Legacy: dedicated producer instance. Prefer kafka_service from producer.py."""
    from aiokafka import AIOKafkaProducer

    hosts = settings.kafka_bootstrap_servers_list()
    if not hosts:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required for producer")
    p = AIOKafkaProducer(bootstrap_servers=hosts)
    await p.start()
    return p


async def stop_chat_job_producer(producer: Any) -> None:
    with contextlib.suppress(Exception):
        await producer.stop()
