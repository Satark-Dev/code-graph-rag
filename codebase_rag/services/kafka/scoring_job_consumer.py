from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from ...config import settings
from ._scoring_job_consumer_loop import run_scoring_job_consumer_loop


async def run_scoring_job_consumer(
    *,
    ingestor: Any,
    stop_event: asyncio.Event,
    start_kafka_service_on_start: bool = False,
    stop_kafka_service_on_exit: bool = False,
) -> None:
    if start_kafka_service_on_start:
        from .producer import kafka_service

        await kafka_service.start()

    if not settings.kafka_bootstrap_servers_list():
        logger.warning("KAFKA_BOOTSTRAP_SERVERS not set; scoring consumer exiting")
        if stop_kafka_service_on_exit:
            from .producer import kafka_service

            await kafka_service.stop()
        return

    context = {"ingestor": ingestor}
    try:
        await run_scoring_job_consumer_loop(context=context, stop_event=stop_event)
    finally:
        if stop_kafka_service_on_exit:
            from .producer import kafka_service

            await kafka_service.stop()

