#!/usr/bin/env python3
"""
Dedicated worker: Kafka scoring jobs consumer (stage 2 of chat pipeline).

Usage (from repo root, with .env loaded):

  uv run python scripts/consume_scoring_jobs.py
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from pathlib import Path

# Repo root on sys.path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger  # noqa: E402

from codebase_rag.config import settings  # noqa: E402
from codebase_rag.main import connect_memgraph  # noqa: E402
from codebase_rag.services.kafka.scoring_job_consumer import (  # noqa: E402
    run_scoring_job_consumer,
)


async def _main() -> None:
    if not settings.kafka_bootstrap_servers_list():
        logger.error("KAFKA_BOOTSTRAP_SERVERS is required.")
        sys.exit(1)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

    ingestor = connect_memgraph(settings.resolve_batch_size(None))
    ingestor.__enter__()
    try:
        await run_scoring_job_consumer(
            ingestor=ingestor,
            stop_event=stop,
            start_kafka_service_on_start=True,
            stop_kafka_service_on_exit=True,
        )
    finally:
        ingestor.__exit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(_main())

