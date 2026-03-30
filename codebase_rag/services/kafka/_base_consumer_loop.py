from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError, UnknownTopicOrPartitionError
from aiokafka.structs import TopicPartition
from loguru import logger


BuildConsumerFn = Callable[[], AIOKafkaConsumer]
EnsureTopicFn = Callable[[], Awaitable[None]]
ProcessMessageFn = Callable[[AIOKafkaConsumer, Any, Any], Awaitable[None]]


@dataclass
class ConsumerLoopConfig:
    reconnect_backoff_initial: float
    reconnect_max_seconds: float
    fetch_max_wait_ms: int
    max_concurrency: int
    missing_topic_log_msg: str
    start_failed_log_msg: str
    read_error_log_msg: str
    reconnect_log_msg: str


_POISON = object()


async def run_consumer_loop(
    *,
    build_consumer: BuildConsumerFn,
    ensure_topic: EnsureTopicFn,
    process_message: ProcessMessageFn,
    ingestor: Any,
    stop_event: asyncio.Event,
    cfg: ConsumerLoopConfig,
) -> None:
    """
    Shared Kafka consumer loop with:
    - reconnect/backoff
    - per-partition queues and workers
    - bounded cross-partition concurrency
    """
    backoff = cfg.reconnect_backoff_initial
    max_backoff = cfg.reconnect_max_seconds

    while not stop_event.is_set():
        consumer = build_consumer()
        try:
            await consumer.start()
        except UnknownTopicOrPartitionError:
            logger.warning(cfg.missing_topic_log_msg)
            with contextlib.suppress(Exception):
                await consumer.stop()
            await ensure_topic()
            if stop_event.is_set():
                return
            await asyncio.sleep(min(backoff, max_backoff))
            backoff = min(backoff * 2, max_backoff)
            continue
        except KafkaError as e:
            logger.warning(cfg.start_failed_log_msg, e)
            with contextlib.suppress(Exception):
                await consumer.stop()
            if stop_event.is_set():
                return
            await asyncio.sleep(min(backoff, max_backoff) + random.uniform(0, 0.25))
            backoff = min(backoff * 2, max_backoff)
            continue

        backoff = cfg.reconnect_backoff_initial
        queues: dict[TopicPartition, asyncio.Queue[Any]] = {}
        workers: dict[TopicPartition, asyncio.Task[None]] = {}
        sem = asyncio.Semaphore(cfg.max_concurrency)

        async def _partition_worker(tp: TopicPartition, q: asyncio.Queue[Any]) -> None:
            while True:
                item = await q.get()
                if item is _POISON:
                    break
                try:
                    async with sem:
                        await process_message(consumer, item, ingestor)
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
            poll = max(cfg.fetch_max_wait_ms / 1000.0, 0.05)
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(consumer.getone(), timeout=poll)
                except TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except KafkaError as e:
                    logger.warning(cfg.read_error_log_msg, e)
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

        logger.info(cfg.reconnect_log_msg)
        await asyncio.sleep(min(backoff, max_backoff))

