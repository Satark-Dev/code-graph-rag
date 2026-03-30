from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from ...config import settings
from ._chat_job_consumer_loop import run_chat_job_consumer_loop


async def run_chat_job_consumer(
    *,
    ingestor: Any,
    stop_event: asyncio.Event,
    start_kafka_service_on_start: bool = False,
    stop_kafka_service_on_exit: bool = False,
) -> None:
    """Top-level runner; optional producer lifecycle for standalone workers."""
    if start_kafka_service_on_start:
        from .producer import kafka_service

        await kafka_service.start()

    if not settings.kafka_bootstrap_servers_list():
        logger.warning("KAFKA_BOOTSTRAP_SERVERS not set; chat consumer exiting")
        if stop_kafka_service_on_exit:
            from .producer import kafka_service

            await kafka_service.stop()
        return

    from .repo_manager import RepoManager

    repo_manager = RepoManager()
    context = {"ingestor": ingestor, "repo_manager": repo_manager}

    try:
        await run_chat_job_consumer_loop(context=context, stop_event=stop_event)
    finally:
        repo_manager.cleanup_all()

        if stop_kafka_service_on_exit:
            from .producer import kafka_service

            await kafka_service.stop()
