import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from .bootstrap import connect_memgraph, prewarm_semantic_model, warm_core_db
from .config import kafka_consumer_reload_guard_allows_start, settings
from .context import app_context
from .services.semantic_reranker import aclose_deepinfra_client

# Configure confirmation prompts in API context based on CLI flag
app_context.session.confirm_edits = not (os.environ.get("CGR_NO_CONFIRM") == "1")


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with connect_memgraph(settings.resolve_batch_size(None)) as ingestor:
        _app.state.ingestor = ingestor

        prewarm_semantic_model()
        warm_core_db()
        kafka_started = await _start_kafka_producer()

        try:
            yield
        finally:
            await _shutdown_clients(kafka_started)


app = FastAPI(
    title="Code Graph RAG API",
    description="API process (lifespan only; no public HTTP endpoints, work goes via Kafka).",
    version="1.0.0",
    lifespan=lifespan,
)
