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
from ._evidence_job_consumer_worker import process_evidence_job_message
from .stage_job_payloads import EvidenceJobPayloadV1


def build_evidence_job_consumer() -> AIOKafkaConsumer:
    hosts = settings.kafka_bootstrap_servers_list()
    if not hosts:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required when evidence consumer runs")
    return AIOKafkaConsumer(
        settings.KAFKA_EVIDENCE_JOBS_TOPIC,
        bootstrap_servers=hosts,
        group_id=settings.KAFKA_EVIDENCE_CONSUMER_GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset=settings.KAFKA_EVIDENCE_AUTO_OFFSET_RESET,
        session_timeout_ms=settings.KAFKA_EVIDENCE_SESSION_TIMEOUT_MS,
        fetch_max_wait_ms=settings.KAFKA_EVIDENCE_FETCH_MAX_WAIT_MS,
    )


async def _ensure_topic() -> None:
    hosts = settings.kafka_bootstrap_servers_list()
    if not hosts:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is required when ensuring evidence topic")
    admin = AIOKafkaAdminClient(bootstrap_servers=hosts)
    await admin.start()
    try:
        await admin.create_topics(
            [
                NewTopic(
                    name=settings.KAFKA_EVIDENCE_JOBS_TOPIC,
                    num_partitions=settings.KAFKA_EVIDENCE_TOPIC_NUM_PARTITIONS,
                    replication_factor=settings.KAFKA_EVIDENCE_TOPIC_REPLICATION_FACTOR,
                )
            ],
            timeout_ms=10_000,
        )
        logger.info("Kafka evidence topic created: {}", settings.KAFKA_EVIDENCE_JOBS_TOPIC)
    except TopicAlreadyExistsError:
        return
    except Exception as e:  # noqa: BLE001
        logger.warning("Kafka evidence topic create failed: {}", e)
    finally:
        await admin.close()


async def _commit_offset(consumer: AIOKafkaConsumer, tp: TopicPartition, offset: int) -> None:
    await consumer.commit({tp: OffsetAndMetadata(offset + 1, "")})


async def _process_one_raw(consumer: AIOKafkaConsumer, msg: Any, context: dict[str, Any]) -> None:
    tp = TopicPartition(msg.topic, msg.partition)
    key_s = msg.key.decode("utf-8") if msg.key else None
    raw = msg.value
    if raw is None:
        logger.warning(
            "Kafka evidence job null value topic={} partition={} offset={}",
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
            "Kafka evidence job invalid JSON topic={} partition={} offset={}: {}",
            tp.topic,
            tp.partition,
            msg.offset,
            e,
        )
        await _commit_offset(consumer, tp, msg.offset)
        return
    try:
        payload = EvidenceJobPayloadV1.model_validate_payload(body)
    except Exception as e:
        logger.warning(
            "Kafka evidence job invalid payload topic={} partition={} offset={}: {}",
            tp.topic,
            tp.partition,
            msg.offset,
            e,
        )
        await _commit_offset(consumer, tp, msg.offset)
        return

    ok = await process_evidence_job_message(
        payload=payload,
        ingestor=context["ingestor"],
        repo_manager=context["repo_manager"],
        key=key_s,
    )
    if ok:
        await _commit_offset(consumer, tp, msg.offset)


async def run_evidence_job_consumer_loop(*, context: dict[str, Any], stop_event: asyncio.Event) -> None:
    cfg = ConsumerLoopConfig(
        reconnect_backoff_initial=settings.KAFKA_EVIDENCE_RECONNECT_BACKOFF_INITIAL,
        reconnect_max_seconds=settings.KAFKA_EVIDENCE_RECONNECT_MAX_SECONDS,
        fetch_max_wait_ms=settings.KAFKA_EVIDENCE_FETCH_MAX_WAIT_MS,
        max_concurrency=settings.KAFKA_EVIDENCE_MAX_CONCURRENCY,
        missing_topic_log_msg=(
            f"Kafka evidence topic missing: {settings.KAFKA_EVIDENCE_JOBS_TOPIC}; attempting auto-create"
        ),
        start_failed_log_msg="Kafka consumer start failed: {}",
        read_error_log_msg="Kafka read error: {}",
        reconnect_log_msg="Kafka evidence consumer reconnecting after reader exit",
    )
    await run_consumer_loop(
        build_consumer=build_evidence_job_consumer,
        ensure_topic=_ensure_topic,
        process_message=_process_one_raw,
        ingestor=context,
        stop_event=stop_event,
        cfg=cfg,
    )

