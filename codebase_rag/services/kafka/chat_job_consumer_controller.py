from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import FastAPI
from loguru import logger

from ...config import settings
from .chat_job_consumer import run_chat_job_consumer

_start_lock = asyncio.Lock()


def _task_running(task: asyncio.Task[Any] | None) -> bool:
    return task is not None and not task.done()


async def start_chat_job_consumer_background(
    app: FastAPI,
    *,
    ingestor: Any,
) -> None:
    """
    Idempotent: one consumer task per process. Uses a module lock (repomind-style)
    to prevent double-start under concurrent enqueue + lifespan.
    """
    if not settings.kafka_bootstrap_servers_list():
        logger.error(
            "KAFKA_BOOTSTRAP_SERVERS is empty; cannot start chat consumer"
        )
        return

    async with _start_lock:
        existing = getattr(app.state, "chat_job_consumer_task", None)
        if _task_running(existing):
            return

        stop_event = asyncio.Event()
        app.state.chat_job_consumer_stop_event = stop_event
        app.state.chat_job_consumer_task = asyncio.create_task(
            run_chat_job_consumer(
                ingestor=ingestor,
                stop_event=stop_event,
                start_kafka_service_on_start=False,
                stop_kafka_service_on_exit=False,
            ),
            name="cgr-kafka-chat-jobs",
        )
        logger.info(
            "Kafka chat job consumer started topic={} group={}",
            settings.KAFKA_CHAT_JOBS_TOPIC,
            settings.KAFKA_CHAT_CONSUMER_GROUP_ID,
        )


async def ensure_chat_job_consumer_started(app: FastAPI) -> None:
    """On-demand start after enqueue when Kafka is configured."""
    ingestor = getattr(app.state, "ingestor", None)
    if ingestor is None:
        logger.warning("ensure_chat_job_consumer_started: no ingestor on app.state")
        return
    await start_chat_job_consumer_background(app, ingestor=ingestor)


async def stop_chat_job_consumer_background(app: FastAPI) -> None:
    stop_event = getattr(app.state, "chat_job_consumer_stop_event", None)
    task = getattr(app.state, "chat_job_consumer_task", None)
    if stop_event is not None:
        stop_event.set()
    if task is None:
        return
    grace = settings.KAFKA_CHAT_SHUTDOWN_GRACE_SECONDS
    try:
        await asyncio.wait_for(task, timeout=grace)
    except TimeoutError:
        logger.warning(
            "Kafka chat consumer did not finish within {}s; cancelling",
            grace,
        )
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    except asyncio.CancelledError:
        raise
    finally:
        async with _start_lock:
            app.state.chat_job_consumer_task = None
            app.state.chat_job_consumer_stop_event = None
