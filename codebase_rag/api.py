import asyncio
import os

from loguru import logger

from .bootstrap import connect_memgraph, prewarm_semantic_model, warm_core_db
from .config import kafka_consumer_reload_guard_allows_start, settings
from .context import app_context
from .services.semantic_reranker import aclose_deepinfra_client

# Configure confirmation prompts in API context based on CLI flag
app_context.session.confirm_edits = not (os.environ.get("CGR_NO_CONFIRM") == "1")


def _embedded_kafka_base_enabled() -> bool:
    """Shared gate: bootstrap, reload guard, master opt-out CGR_EMBEDDED_KAFKA_CONSUMERS."""
    raw = os.environ.get("CGR_EMBEDDED_KAFKA_CONSUMERS", "1").strip().lower()
    if raw in ("0", "false", "no"):
        return False
    if not settings.kafka_bootstrap_servers_list():
        return False
    if not kafka_consumer_reload_guard_allows_start():
        logger.info("Embedded Kafka consumers skipped (reload / subprocess guard).")
        return False
    return True


def _env_flag_enabled(name: str, *, default: str = "1") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw not in ("0", "false", "no")


def _embedded_index_consumer_enabled() -> bool:
    """Index jobs consumer (clone + GraphUpdater); opt-out via CGR_EMBEDDED_KAFKA_INDEX_CONSUMERS."""
    if not _embedded_kafka_base_enabled():
        return False
    return _env_flag_enabled("CGR_EMBEDDED_KAFKA_INDEX_CONSUMERS")


def _embedded_chat_consumer_enabled() -> bool:
    """Chat jobs consumer (local repo only); opt-out via CGR_EMBEDDED_KAFKA_CHAT_CONSUMERS."""
    if not _embedded_kafka_base_enabled():
        return False
    return _env_flag_enabled("CGR_EMBEDDED_KAFKA_CHAT_CONSUMERS")


async def _run_index_consumer(stop_event: asyncio.Event) -> None:
    from .services.kafka.index_job_consumer import run_index_job_consumer

    async with connect_memgraph(settings.resolve_batch_size(None)) as ingestor:
        await run_index_job_consumer(
            ingestor=ingestor,
            stop_event=stop_event,
            start_kafka_service_on_start=False,
            stop_kafka_service_on_exit=False,
        )


async def _run_chat_consumers(stop_event: asyncio.Event) -> None:
    from .services.kafka.evidence_job_consumer import run_evidence_job_consumer
    from .services.kafka.remediation_job_consumer import run_remediation_job_consumer
    from .services.kafka.scoring_job_consumer import run_scoring_job_consumer

    async with connect_memgraph(settings.resolve_batch_size(None)) as ingestor:
        await asyncio.gather(
            run_evidence_job_consumer(
                ingestor=ingestor,
                stop_event=stop_event,
                start_kafka_service_on_start=False,
                stop_kafka_service_on_exit=False,
            ),
            run_scoring_job_consumer(
                ingestor=ingestor,
                stop_event=stop_event,
                start_kafka_service_on_start=False,
                stop_kafka_service_on_exit=False,
            ),
            run_remediation_job_consumer(
                ingestor=ingestor,
                stop_event=stop_event,
                start_kafka_service_on_start=False,
                stop_kafka_service_on_exit=False,
            ),
        )


async def _start_kafka_producer() -> bool:
    if not kafka_consumer_reload_guard_allows_start():
        logger.info("Kafka producer start skipped by reload guard.")
        return False

    try:
        from .services.kafka.producer import kafka_service

        await kafka_service.start()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Kafka producer failed to start: {}", e)
        return False


async def _stop_kafka_producer() -> None:
    try:
        from .services.kafka.producer import kafka_service

        await kafka_service.stop()
    except Exception as e:  # noqa: BLE001
        logger.warning("Kafka producer stop failed: {}", e)


async def _shutdown_clients(kafka_started: bool) -> None:
    if kafka_started:
        await _stop_kafka_producer()

    try:
        await aclose_deepinfra_client()
    except Exception as e:  # noqa: BLE001
        logger.warning("DeepInfra client shutdown failed: {}", e)


def _start_embedded_consumers() -> tuple[
    asyncio.Event | None, list[asyncio.Task[None]]
]:
    start_index = _embedded_index_consumer_enabled()
    start_chat = _embedded_chat_consumer_enabled()
    if not (start_index or start_chat):
        logger.info("Embedded Kafka consumers disabled.")
        return None, []

    stop_event = asyncio.Event()
    tasks: list[asyncio.Task[None]] = []
    if start_index:
        tasks.append(
            asyncio.create_task(
                _run_index_consumer(stop_event),
                name="cgr-kafka-index-consumer",
            )
        )
    if start_chat:
        tasks.append(
            asyncio.create_task(
                _run_chat_consumers(stop_event),
                name="cgr-kafka-chat-consumers",
            )
        )

    logger.info("Starting embedded Kafka consumer(s)...")
    return stop_event, tasks


async def run_kafka_consumers_forever() -> None:
    """
    Kafka-only runner for the API process.

    Starts Kafka producer plus optional embedded index/chat consumer loops.
    Runs until cancelled (Ctrl+C / SIGTERM).
    """
    stop_event = asyncio.Event()
    await run_kafka_consumers_until(stop_event)


async def run_kafka_consumers_until(stop_event: asyncio.Event) -> None:
    """
    Kafka-only runner that stops when `stop_event` is set.

    This is the preferred entrypoint for CLI shutdown: it allows a graceful stop
    (no task cancellation), so Kafka consumers can close cleanly.
    """
    prewarm_semantic_model()
    warm_core_db()
    kafka_started = await _start_kafka_producer()

    consumer_stop, consumer_tasks = _start_embedded_consumers()

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        # Allow direct use without external signal wiring.
        pass
    finally:
        if consumer_stop is not None and consumer_tasks:
            consumer_stop.set()
            results = await asyncio.gather(*consumer_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    logger.warning("Embedded Kafka consumer task ended with: {}", res)
        await _shutdown_clients(kafka_started)


def main() -> None:
    asyncio.run(run_kafka_consumers_forever())


if __name__ == "__main__":
    main()
