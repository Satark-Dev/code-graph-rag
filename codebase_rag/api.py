import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from . import logs as ls
from .config import kafka_consumer_reload_guard_allows_start, settings
from .main import app_context, connect_memgraph
from .services.semantic_reranker import aclose_deepinfra_client
from .utils.dependencies import has_semantic_dependencies

# Configure confirmation prompts in API context based on CLI flag
app_context.session.confirm_edits = not (os.environ.get("CGR_NO_CONFIRM") == "1")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ingestor = None
    try:
        ingestor = connect_memgraph(settings.resolve_batch_size(None))
        ingestor.__enter__()
        _app.state.ingestor = ingestor
    except Exception:
        if ingestor is not None:
            ingestor.__exit__(None, None, None)
        raise

    if settings.SEMANTIC_SEARCH_ENABLED and has_semantic_dependencies():
        try:
            from .embedder import prewarm_embeddings

            logger.info(ls.SEMANTIC_PREWARM_START)
            prewarm_embeddings()
            logger.info(ls.SEMANTIC_PREWARM_COMPLETE)
        except Exception as e:
            logger.warning(ls.SEMANTIC_PREWARM_FAILED.format(error=e))

    try:
        from .utils.org_region_resolver import get_org_region_resolver

        get_org_region_resolver().warm_core_connection()
    except Exception as e:
        logger.warning("Core DB warm failed: {}", e)

    try:
        from .services.kafka.producer import kafka_service

        await kafka_service.start()
    except Exception as e:
        logger.warning("Kafka producer failed to start: {}", e)

    yield

    try:
        from .services.kafka.producer import kafka_service

        await kafka_service.stop()
    except Exception as e:
        logger.warning("Kafka producer stop failed: {}", e)

    ingestor = getattr(_app.state, "ingestor", None)
    if ingestor is not None:
        ingestor.__exit__(None, None, None)

    try:
        await aclose_deepinfra_client()
    except Exception as e:
        logger.warning("DeepInfra client shutdown failed: {}", e)


app = FastAPI(
    title="Code Graph RAG API",
    description="API process (lifespan only; no public HTTP endpoints, work goes via Kafka).",
    version="1.0.0",
    lifespan=lifespan,
)
