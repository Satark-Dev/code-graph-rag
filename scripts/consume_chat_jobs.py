#!/usr/bin/env python3
"""
Dedicated worker: Kafka chat jobs consumer (same pipeline as POST /api/chat).

Usage (from repo root, with .env loaded):

  uv run python scripts/consume_chat_jobs.py
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

from codebase_rag.config import settings
from codebase_rag.bootstrap import connect_memgraph
from codebase_rag.services.kafka.chat_job_consumer import run_chat_job_consumer


async def _main() -> None:
    if not settings.kafka_bootstrap_servers_list():
        print("KAFKA_BOOTSTRAP_SERVERS is required.", file=sys.stderr)
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
        await run_chat_job_consumer(
            ingestor=ingestor,
            stop_event=stop,
            start_kafka_service_on_start=True,
            stop_kafka_service_on_exit=True,
        )
    finally:
        ingestor.__exit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(_main())
