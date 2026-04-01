import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator

from .bootstrap import connect_memgraph, prewarm_semantic_model, warm_core_db
from .config import kafka_consumer_reload_guard_allows_start, settings
from .context import app_context
from .services.semantic_reranker import aclose_deepinfra_client


class ChatRequest(BaseModel):
    """
    Backwards-compatible request model (used by tests and older callers).
    Note: the API process does not expose chat HTTP endpoints; Kafka drives the pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    org_id: str
    org_tool_findings_ids: list[str]

    @field_validator("org_id")
    @classmethod
    def _org_id_non_blank(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("org_id is required")
        return v

    @field_validator("org_tool_findings_ids")
    @classmethod
    def _findings_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [x.strip() for x in value if isinstance(x, str) and x.strip()]
        if not cleaned:
            raise ValueError("org_tool_findings_ids must be non-empty")
        return cleaned


class IndexRequest(BaseModel):
    """
    Backwards-compatible request model (used by tests and older callers).
    """

    model_config = ConfigDict(extra="forbid")

    org_id: str
    repo_path: str
    clean: bool = True
    exclude: list[str] | None = None

    @field_validator("org_id")
    @classmethod
    def _org_id_non_blank(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("org_id is required")
        return v

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


async def _embedded_index_consumer(stop_event: asyncio.Event) -> None:
    from .services.kafka.index_job_consumer import run_index_job_consumer

    async with connect_memgraph(settings.resolve_batch_size(None)) as ingestor:
        await run_index_job_consumer(
            ingestor=ingestor,
            stop_event=stop_event,
            start_kafka_service_on_start=False,
            stop_kafka_service_on_exit=False,
        )


async def _embedded_chat_consumer(stop_event: asyncio.Event) -> None:
    from .services.kafka.evidence_job_consumer import run_evidence_job_consumer
    from .services.kafka.scoring_job_consumer import run_scoring_job_consumer
    from .services.kafka.remediation_job_consumer import run_remediation_job_consumer

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with connect_memgraph(settings.resolve_batch_size(None)) as ingestor:
        _app.state.ingestor = ingestor

        prewarm_semantic_model()
        warm_core_db()
        kafka_started = await _start_kafka_producer()

        consumer_stop: asyncio.Event | None = None
        consumer_tasks: list[asyncio.Task[Any]] = []
        start_index = _embedded_index_consumer_enabled()
        start_chat = _embedded_chat_consumer_enabled()
        if start_index or start_chat:
            consumer_stop = asyncio.Event()
            parts: list[str] = []
            if start_index:
                parts.append("index")
                consumer_tasks.append(
                    asyncio.create_task(_embedded_index_consumer(consumer_stop))
                )
            if start_chat:
                parts.append("chat")
                consumer_tasks.append(
                    asyncio.create_task(_embedded_chat_consumer(consumer_stop))
                )
            logger.info(
                "Starting embedded Kafka consumer(s) ({}); disable all with "
                "CGR_EMBEDDED_KAFKA_CONSUMERS=0 or --no-kafka-consumers; disable only index with "
                "CGR_EMBEDDED_KAFKA_INDEX_CONSUMERS=0 or --no-kafka-index-consumer.",
                "+".join(parts),
            )

        try:
            yield
        finally:
            if consumer_stop is not None and consumer_tasks:
                consumer_stop.set()
                results = await asyncio.gather(*consumer_tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, BaseException):
                        logger.warning("Embedded Kafka consumer task ended with: {}", res)
            await _shutdown_clients(kafka_started)


app = FastAPI(
    title="Code Graph RAG API",
    description="API process (lifespan only; no public HTTP endpoints, work goes via Kafka).",
    version="1.0.0",
    lifespan=lifespan,
)
