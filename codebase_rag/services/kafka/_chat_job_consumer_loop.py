from __future__ import annotations

import asyncio
import json
from typing import Any

from aiokafka import AIOKafkaConsumer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from loguru import logger

from ...config import settings
from ._base_consumer_loop import ConsumerLoopConfig, run_consumer_loop
from ._chat_job_consumer_worker import process_chat_job_message
from .chat_job_payload import ChatJobPayloadV1


def build_chat_job_consumer() -> AIOKafkaConsumer:
    hosts = settings.kafka_bootstrap_servers_list()
    if not hosts:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required when chat consumer runs")
    return AIOKafkaConsumer(
        settings.KAFKA_CHAT_JOBS_TOPIC,
        bootstrap_servers=hosts,
        group_id=settings.KAFKA_CHAT_CONSUMER_GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset=settings.KAFKA_CHAT_AUTO_OFFSET_RESET,
        session_timeout_ms=settings.KAFKA_CHAT_SESSION_TIMEOUT_MS,
        fetch_max_wait_ms=settings.KAFKA_CHAT_FETCH_MAX_WAIT_MS,
    )


async def _ensure_topic() -> None:
    """Ensure the chat jobs topic exists; create it if missing."""
    hosts = settings.kafka_bootstrap_servers_list()
    if not hosts:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required when ensuring chat topic")
    admin = AIOKafkaAdminClient(bootstrap_servers=hosts)
    await admin.start()
    try:
        await admin.create_topics(
            [
                NewTopic(
                    name=settings.KAFKA_CHAT_JOBS_TOPIC,
                    num_partitions=settings.KAFKA_CHAT_TOPIC_NUM_PARTITIONS,
                    replication_factor=settings.KAFKA_CHAT_TOPIC_REPLICATION_FACTOR,
                )
            ],
            timeout_ms=10_000,
        )
        logger.info("Kafka chat topic created: {}", settings.KAFKA_CHAT_JOBS_TOPIC)
    except TopicAlreadyExistsError:
        # Race: topic was created by another process
        return
    except Exception as e:
        logger.warning("Kafka chat topic create failed: {}", e)
    finally:
        await admin.close()


async def _commit_offset(
    consumer: AIOKafkaConsumer, tp: TopicPartition, offset: int
) -> None:
    await consumer.commit({tp: OffsetAndMetadata(offset + 1, "")})


async def _process_one_raw(
    consumer: AIOKafkaConsumer,
    msg: Any,
    ingestor: Any,
) -> None:
    tp = TopicPartition(msg.topic, msg.partition)
    key_s = msg.key.decode("utf-8") if msg.key else None
    raw = msg.value
    if raw is None:
        logger.warning(
            "Kafka chat job null value topic={} partition={} offset={}",
            tp.topic,
            tp.partition,
            msg.offset,
        )
        await _commit_offset(consumer, tp, msg.offset)
        return
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception as e:
        logger.warning(
            "Kafka chat job invalid JSON topic={} partition={} offset={}: {}",
            tp.topic,
            tp.partition,
            msg.offset,
            e,
        )
        await _commit_offset(consumer, tp, msg.offset)
        return
    try:
        payload = ChatJobPayloadV1.model_validate_payload(body)
    except Exception as e:
        logger.warning(
            "Kafka chat job invalid payload topic={} partition={} offset={}: {}",
            tp.topic,
            tp.partition,
            msg.offset,
            e,
        )
        await _commit_offset(consumer, tp, msg.offset)
        return

    ok = await process_chat_job_message(
        payload=payload, ingestor=ingestor, key=key_s
    )
    if ok:
        await _commit_offset(consumer, tp, msg.offset)


async def run_chat_job_consumer_loop(
    *,
    ingestor: Any,
    stop_event: asyncio.Event,
) -> None:
    """
    At-least-once delivery with manual commit after successful handling.

    Notes:
    - Kafka topics are expected to be pre-created (no auto-create in code).
    - Per-partition FIFO (Kafka fetch order per partition) with cross-partition
      parallelism capped by KAFKA_CHAT_MAX_CONCURRENCY.
    """
    cfg = ConsumerLoopConfig(
        reconnect_backoff_initial=settings.KAFKA_CHAT_RECONNECT_BACKOFF_INITIAL,
        reconnect_max_seconds=settings.KAFKA_CHAT_RECONNECT_MAX_SECONDS,
        fetch_max_wait_ms=settings.KAFKA_CHAT_FETCH_MAX_WAIT_MS,
        max_concurrency=settings.KAFKA_CHAT_MAX_CONCURRENCY,
        missing_topic_log_msg=(
            f"Kafka chat topic missing: {settings.KAFKA_CHAT_JOBS_TOPIC}; attempting auto-create"
        ),
        start_failed_log_msg="Kafka consumer start failed: {}",
        read_error_log_msg="Kafka read error: {}",
        reconnect_log_msg="Kafka chat consumer reconnecting after reader exit",
    )

    await run_consumer_loop(
        build_consumer=build_chat_job_consumer,
        ensure_topic=_ensure_topic,
        process_message=_process_one_raw,
        ingestor=ingestor,
        stop_event=stop_event,
        cfg=cfg,
    )
