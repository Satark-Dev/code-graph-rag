from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from loguru import logger

from ...config import settings
from .index_job_consumer import run_index_job_consumer

_start_lock = asyncio.Lock()
_index_task: asyncio.Task[Any] | None = None
_index_stop_event: asyncio.Event | None = None


def _task_running(task: asyncio.Task[Any] | None) -> bool:
    return task is not None and not task.done()


async def start_index_job_consumer_background(
    *,
    ingestor: Any,
) -> None:
    """
    Idempotent: one index consumer task per process.
    """
    if not settings.kafka_bootstrap_servers_list():
        logger.error("KAFKA_BOOTSTRAP_SERVERS is empty; cannot start index consumer")
        return

    async with _start_lock:
        global _index_task, _index_stop_event
        if _task_running(_index_task):
            return

        _index_stop_event = asyncio.Event()
        _index_task = asyncio.create_task(
            run_index_job_consumer(
                ingestor=ingestor,
                stop_event=_index_stop_event,
                start_kafka_service_on_start=False,
                stop_kafka_service_on_exit=False,
            ),
            name="cgr-kafka-index-jobs",
        )
        logger.info(
            "Kafka index job consumer started topic={} group={}",
            settings.KAFKA_INDEX_JOBS_TOPIC,
            settings.KAFKA_INDEX_CONSUMER_GROUP_ID,
        )


async def stop_index_job_consumer_background() -> None:
    global _index_task, _index_stop_event
    if _index_stop_event is not None:
        _index_stop_event.set()
    if _index_task is None:
        return
    grace = settings.KAFKA_INDEX_SHUTDOWN_GRACE_SECONDS
    try:
        await asyncio.wait_for(_index_task, timeout=grace)
    except TimeoutError:
        logger.warning(
            "Kafka index consumer did not finish within {}s; cancelling",
            grace,
        )
        _index_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _index_task
    except asyncio.CancelledError:
        raise
    finally:
        async with _start_lock:
            _index_task = None
            _index_stop_event = None
