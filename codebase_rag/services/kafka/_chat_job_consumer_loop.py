from __future__ import annotations

import asyncio
import contextlib
import json
import random
from typing import Any

from aiokafka import AIOKafkaConsumer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaError, TopicAlreadyExistsError, UnknownTopicOrPartitionError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from loguru import logger

from ...config import settings
from ._chat_job_consumer_worker import process_chat_job_message
from .chat_job_payload import ChatJobPayloadV1

_POISON = object()


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
    *,
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
    backoff = settings.KAFKA_CHAT_RECONNECT_BACKOFF_INITIAL
    max_backoff = settings.KAFKA_CHAT_RECONNECT_MAX_SECONDS

    while not stop_event.is_set():
        consumer = build_chat_job_consumer()
        try:
            await consumer.start()
        except UnknownTopicOrPartitionError:
            logger.warning(
                "Kafka chat topic missing: {}; attempting auto-create",
                settings.KAFKA_CHAT_JOBS_TOPIC,
            )
            with contextlib.suppress(Exception):
                await consumer.stop()
            await _ensure_topic()
            if stop_event.is_set():
                return
            await asyncio.sleep(min(backoff, max_backoff))
            backoff = min(backoff * 2, max_backoff)
            continue
        except KafkaError as e:
            logger.warning("Kafka consumer start failed: {}", e)
            with contextlib.suppress(Exception):
                await consumer.stop()
            if stop_event.is_set():
                return
            await asyncio.sleep(min(backoff, max_backoff) + random.uniform(0, 0.25))
            backoff = min(backoff * 2, max_backoff)
            continue

        backoff = settings.KAFKA_CHAT_RECONNECT_BACKOFF_INITIAL
        queues: dict[TopicPartition, asyncio.Queue[Any]] = {}
        workers: dict[TopicPartition, asyncio.Task[None]] = {}
        sem = asyncio.Semaphore(settings.KAFKA_CHAT_MAX_CONCURRENCY)

        async def _partition_worker(tp: TopicPartition, q: asyncio.Queue[Any]) -> None:
            while True:
                item = await q.get()
                if item is _POISON:
                    break
                try:
                    async with sem:
                        await _process_one_raw(consumer, item, ingestor=ingestor)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Kafka partition worker error topic={} partition={}",
                        tp.topic,
                        tp.partition,
                    )

        def _ensure_queue(tp: TopicPartition) -> asyncio.Queue[Any]:
            if tp not in queues:
                q: asyncio.Queue[Any] = asyncio.Queue()
                queues[tp] = q
                workers[tp] = asyncio.create_task(_partition_worker(tp, q))
            return queues[tp]

        async def _reader() -> None:
            poll = max(settings.KAFKA_CHAT_FETCH_MAX_WAIT_MS / 1000.0, 0.05)
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(consumer.getone(), timeout=poll)
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except KafkaError as e:
                    logger.warning("Kafka read error: {}", e)
                    return
                if msg is None:
                    continue
                tp = TopicPartition(msg.topic, msg.partition)
                await _ensure_queue(tp).put(msg)

        read_task = asyncio.create_task(_reader())
        try:
            await read_task
        finally:
            read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await read_task
            for q in queues.values():
                await q.put(_POISON)
            await asyncio.gather(*workers.values(), return_exceptions=True)
            with contextlib.suppress(Exception):
                await consumer.stop()

        if stop_event.is_set():
            return

        logger.info("Kafka chat consumer reconnecting after reader exit")
        await asyncio.sleep(min(backoff, max_backoff))

