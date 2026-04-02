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
from ._index_job_consumer_worker import process_index_job_message
from .index_job_payload import IndexJobPayload


def build_index_job_consumer() -> AIOKafkaConsumer:
    hosts = settings.kafka_bootstrap_servers_list()
    if not hosts:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required when index consumer runs")
    return AIOKafkaConsumer(
        settings.KAFKA_INDEX_JOBS_TOPIC,
        bootstrap_servers=hosts,
        group_id=settings.KAFKA_INDEX_CONSUMER_GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset=settings.KAFKA_INDEX_AUTO_OFFSET_RESET,
        session_timeout_ms=settings.KAFKA_INDEX_SESSION_TIMEOUT_MS,
        fetch_max_wait_ms=settings.KAFKA_INDEX_FETCH_MAX_WAIT_MS,
    )


async def _ensure_index_topic() -> None:
    """Ensure the index jobs topic exists; create it if missing."""
    hosts = settings.kafka_bootstrap_servers_list()
    if not hosts:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required when ensuring index topic")
    admin = AIOKafkaAdminClient(bootstrap_servers=hosts)
    await admin.start()
    try:
        await admin.create_topics(
            [
                NewTopic(
                    name=settings.KAFKA_INDEX_JOBS_TOPIC,
                    num_partitions=settings.KAFKA_INDEX_TOPIC_NUM_PARTITIONS,
                    replication_factor=settings.KAFKA_INDEX_TOPIC_REPLICATION_FACTOR,
                )
            ],
            timeout_ms=10_000,
        )
        logger.info("Kafka index topic created: {}", settings.KAFKA_INDEX_JOBS_TOPIC)
    except TopicAlreadyExistsError:
        # Race: topic was created by another process
        return
    except Exception as e:
        logger.warning("Kafka index topic create failed: {}", e)
    finally:
        await admin.close()


async def _commit_offset(
    consumer: AIOKafkaConsumer, tp: TopicPartition, offset: int
) -> None:
    await consumer.commit({tp: OffsetAndMetadata(offset + 1, "")})


async def _process_one_raw(
    consumer: AIOKafkaConsumer,
    msg: Any,
    context: dict[str, Any],
) -> None:
    tp = TopicPartition(msg.topic, msg.partition)
    key_s = msg.key.decode("utf-8") if msg.key else None
    raw = msg.value
    if raw is None:
        logger.warning(
            "Kafka index job null value topic={} partition={} offset={}",
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
            "Kafka index job invalid JSON topic={} partition={} offset={}: {}",
            tp.topic,
            tp.partition,
            msg.offset,
            e,
        )
        await _commit_offset(consumer, tp, msg.offset)
        return
    try:
        payload = IndexJobPayload.model_validate_payload(body)
    except Exception as e:
        logger.warning(
            "Kafka index job invalid payload topic={} partition={} offset={}: {}",
            tp.topic,
            tp.partition,
            msg.offset,
            e,
        )
        await _commit_offset(consumer, tp, msg.offset)
        return

    ok = await process_index_job_message(
        payload=payload,
        ingestor=context["ingestor"],
        repo_manager=context["repo_manager"],
        key=key_s,
    )
    if ok:
        await _commit_offset(consumer, tp, msg.offset)


async def run_index_job_consumer_loop(
    *,
    context: dict[str, Any],
    stop_event: asyncio.Event,
) -> None:
    """
    At-least-once delivery with manual commit after successful handling.

    Notes:
    - Kafka index topics are expected to be pre-created (no auto-create in code).
    - Per-partition FIFO with cross-partition parallelism capped by
      KAFKA_INDEX_MAX_CONCURRENCY.
    """
    cfg = ConsumerLoopConfig(
        reconnect_backoff_initial=settings.KAFKA_INDEX_RECONNECT_BACKOFF_INITIAL,
        reconnect_max_seconds=settings.KAFKA_INDEX_RECONNECT_MAX_SECONDS,
        fetch_max_wait_ms=settings.KAFKA_INDEX_FETCH_MAX_WAIT_MS,
        max_concurrency=settings.KAFKA_INDEX_MAX_CONCURRENCY,
        missing_topic_log_msg=(
            f"Kafka index topic missing: {settings.KAFKA_INDEX_JOBS_TOPIC}; attempting auto-create"
        ),
        start_failed_log_msg="Kafka index consumer start failed: {}",
        read_error_log_msg="Kafka index read error: {}",
        reconnect_log_msg="Kafka index consumer reconnecting after reader exit",
    )

    await run_consumer_loop(
        build_consumer=build_index_job_consumer,
        ensure_topic=_ensure_index_topic,
        process_message=_process_one_raw,
        ingestor=context,
        stop_event=stop_event,
        cfg=cfg,
    )
