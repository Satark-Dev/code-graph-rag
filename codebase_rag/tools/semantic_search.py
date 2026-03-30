from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager

from loguru import logger
from pydantic_ai import Tool

from .. import constants as cs
from .. import exceptions as ex
from .. import logs as ls
from ..cypher_queries import (
    CYPHER_GET_FUNCTION_SOURCE_LOCATION,
    build_nodes_by_ids_query,
)
from ..services import QueryProtocol
from ..types_defs import SemanticSearchResult
from ..utils.dependencies import has_semantic_dependencies
from . import tool_descriptions as td


@contextmanager
def _maybe_owned_ingestor(ingestor: QueryProtocol | None):
    """
    Yield an ingestor, creating and managing its lifetime if not provided.
    """
    if ingestor is not None:
        yield ingestor
        return

    from ..config import settings
    from ..services.graph_service import MemgraphIngestor

    owned = MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=cs.SEMANTIC_BATCH_SIZE,
    )
    with owned:
        yield owned


class SemanticSearchService:
    """Core semantic code search implementation (search + rerank)."""

    def __init__(self, ingestor: QueryProtocol | None = None) -> None:
        self._ingestor = ingestor

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        if not has_semantic_dependencies():
            logger.warning(ex.SEMANTIC_EXTRA)
            return []

        try:
            from ..config import settings
            from ..embedder import embed_code
            from ..services.semantic_reranker import rerank_semantic_results
            from ..vector_store import search_embeddings

            query_embedding = embed_code(query)

            candidate_k = top_k
            if settings.SEMANTIC_RERANK_ENABLED:
                candidate_k = max(int(top_k), int(settings.SEMANTIC_RERANK_CANDIDATES))

            search_results = search_embeddings(query_embedding, top_k=candidate_k)

            if not search_results:
                logger.info(ls.SEMANTIC_NO_MATCH.format(query=query))
                return []

            node_ids = [node_id for node_id, _ in search_results]

            with _maybe_owned_ingestor(self._ingestor) as active_ingestor:
                cypher_query = build_nodes_by_ids_query(node_ids)
                params = {str(i): node_id for i, node_id in enumerate(node_ids)}
                results = active_ingestor.fetch_all(cypher_query, params)

            results_map = {res["node_id"]: res for res in results}

            formatted_results: list[SemanticSearchResult] = []
            for node_id, score in search_results:
                if node_id in results_map:
                    result = results_map[node_id]
                    result_type = result["type"]
                    qualified_name = str(result["qualified_name"])
                    name = str(result["name"])
                    type_str = (
                        result_type[0]
                        if isinstance(result_type, list) and result_type
                        else cs.SEMANTIC_TYPE_UNKNOWN
                    )
                    formatted_results.append(
                        SemanticSearchResult(
                            node_id=node_id,
                            qualified_name=qualified_name,
                            name=name,
                            type=type_str,
                            score=round(score, 3),
                        )
                    )

            if not formatted_results:
                logger.info(ls.SEMANTIC_NO_MATCH.format(query=query))
                return []

            if settings.SEMANTIC_RERANK_ENABLED:
                ordered_ids = await rerank_semantic_results(
                    query=query,
                    candidates=formatted_results,
                    top_k=top_k,
                )
                if ordered_ids:
                    by_id = {int(r["node_id"]): r for r in formatted_results}
                    reranked: list[SemanticSearchResult] = []
                    for nid in ordered_ids:
                        if nid in by_id:
                            reranked.append(by_id[nid])
                    if len(reranked) < top_k:
                        seen = {int(r["node_id"]) for r in reranked}
                        for r in formatted_results:
                            rid = int(r["node_id"])
                            if rid not in seen:
                                reranked.append(r)
                            if len(reranked) >= top_k:
                                break
                    formatted_results = reranked

            logger.info(
                ls.SEMANTIC_FOUND.format(
                    count=len(formatted_results),
                    query=query,
                )
            )
            return formatted_results[:top_k]

        except Exception as e:  # noqa: BLE001
            logger.error(ls.SEMANTIC_FAILED.format(query=query, error=e))
            return []


async def semantic_code_search_async(
    query: str,
    top_k: int = 5,
    ingestor: QueryProtocol | None = None,
) -> list[SemanticSearchResult]:
    """Async‑first API that delegates to SemanticSearchService."""
    service = SemanticSearchService(ingestor=ingestor)
    return await service.search(query, top_k=top_k)


def semantic_code_search(
    query: str,
    top_k: int = 5,
    ingestor: QueryProtocol | None = None,
) -> list[SemanticSearchResult]:
    """
    Sync wrapper for semantic search (and optional rerank).

    This function **must not** be called from within an active asyncio event loop.
    Prefer `semantic_code_search_async` in async contexts.
    """
    try:
        asyncio.get_running_loop()
        # If we get here, we're inside an event loop – using this sync API would hang.
        raise RuntimeError(ex.SEMANTIC_SYNC_IN_EVENT_LOOP)
    except RuntimeError:
        # No running loop: safe to drive the async API synchronously.
        return asyncio.run(
            semantic_code_search_async(
                query,
                top_k,
                ingestor=ingestor,
            )
        )


def get_function_source_code(
    node_id: int, ingestor: QueryProtocol | None = None
) -> str | None:
    try:
        from ..utils.source_extraction import (
            extract_source_lines,
            validate_source_location,
        )

        with _maybe_owned_ingestor(ingestor) as active_ingestor:
            results = active_ingestor.fetch_all(
                CYPHER_GET_FUNCTION_SOURCE_LOCATION, {"node_id": node_id}
            )

            if not results:
                logger.warning(ls.SEMANTIC_NODE_NOT_FOUND.format(id=node_id))
                return None

            result = results[0]
            file_path = result.get("path")
            start_line = result.get("start_line")
            end_line = result.get("end_line")

            is_valid, file_path_obj = validate_source_location(file_path, start_line, end_line)
            if not is_valid or file_path_obj is None:
                logger.warning(ls.SEMANTIC_INVALID_LOCATION.format(id=node_id))
                return None

            return extract_source_lines(file_path_obj, start_line, end_line)

    except Exception as e:
        logger.error(ls.SEMANTIC_SOURCE_FAILED.format(id=node_id, error=e))
        return None


def create_semantic_search_tool(ingestor: QueryProtocol | None = None) -> Tool:
    """
    Create the agent-facing semantic search tool.

    The underlying search/rerank/formatting is handled by SemanticSearchService;
    this wrapper is responsible only for lightweight caching/throttling and
    string formatting.
    """
    last_empty_query: str | None = None
    last_empty_ts = 0.0
    last_any_query: str | None = None
    last_any_ts = 0.0
    last_any_response: str | None = None

    service = SemanticSearchService(ingestor=ingestor)

    async def semantic_search_functions(query: str, top_k: int = 5) -> str:
        from ..config import settings

        logger.info(ls.SEMANTIC_TOOL_SEARCH.format(query=query))

        nonlocal last_empty_query, last_empty_ts
        nonlocal last_any_query, last_any_ts
        nonlocal last_any_response
        now = time.monotonic()

        # Return cached positive result if the same query was asked recently.
        if (
            last_any_query == query
            and (now - last_any_ts) < settings.SEMANTIC_SEARCH_REPEAT_COOLDOWN_SECONDS
        ):
            if last_any_response is not None:
                return last_any_response
            return cs.MSG_SEMANTIC_NO_RESULTS.format(query=query)

        # Throttle repeated empty-result queries.
        if (
            last_empty_query == query
            and (now - last_empty_ts) < settings.SEMANTIC_SEARCH_EMPTY_COOLDOWN_SECONDS
        ):
            return cs.MSG_SEMANTIC_NO_RESULTS.format(query=query)

        results = await service.search(query=query, top_k=top_k)
        last_any_query = query
        last_any_ts = now

        if not results:
            last_empty_query = query
            last_empty_ts = now
            last_any_response = cs.MSG_SEMANTIC_NO_RESULTS.format(query=query)
            return last_any_response

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result['qualified_name']} (type: {result['type']}, score: {result['score']})"
            )

        response = cs.MSG_SEMANTIC_RESULT_HEADER.format(count=len(results), query=query)
        response += "\n".join(formatted_results)
        response += cs.MSG_SEMANTIC_RESULT_FOOTER
        last_any_response = response

        return response

    return Tool(
        semantic_search_functions,
        name=td.AgenticToolName.SEMANTIC_SEARCH,
        description=td.SEMANTIC_SEARCH,
    )


def create_get_function_source_tool(ingestor: QueryProtocol | None = None) -> Tool:
    async def get_function_source_by_id(node_id: int) -> str:
        logger.info(ls.SEMANTIC_TOOL_SOURCE.format(id=node_id))

        source_code = get_function_source_code(node_id, ingestor=ingestor)

        if source_code is None:
            return cs.MSG_SEMANTIC_SOURCE_UNAVAILABLE.format(id=node_id)

        return cs.MSG_SEMANTIC_SOURCE_FORMAT.format(id=node_id, code=source_code)

    return Tool(
        get_function_source_by_id,
        name=td.AgenticToolName.GET_FUNCTION_SOURCE,
        description=td.GET_FUNCTION_SOURCE,
    )
