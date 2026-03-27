from __future__ import annotations

import asyncio

from loguru import logger
from pydantic_ai import Tool
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .. import exceptions as ex
from .. import logs as ls
from ..config import settings
from ..constants import (
    QUERY_NOT_AVAILABLE,
    QUERY_RESULTS_PANEL_TITLE,
    QUERY_SUMMARY_DB_ERROR,
    QUERY_SUMMARY_SUCCESS,
    QUERY_SUMMARY_TRANSLATION_FAILED,
    QUERY_SUMMARY_TRUNCATED,
)
from ..schemas import QueryGraphData
from ..services import QueryProtocol
from ..services.llm import CypherGenerator
from ..utils.cypher_sanitizer import CypherSanitizer
from ..utils.token_utils import truncate_results_by_tokens
from . import tool_descriptions as td


def create_query_tool(
    ingestor: QueryProtocol,
    cypher_gen: CypherGenerator,
    console: Console | None = None,
) -> Tool:
    if console is None:
        console = Console(width=None, stderr=True, force_terminal=True)

    async def query_codebase_knowledge_graph(
        natural_language_query: str,
    ) -> QueryGraphData:
        logger.info(ls.TOOL_QUERY_RECEIVED.format(query=natural_language_query))
        cypher_query = QUERY_NOT_AVAILABLE

        def _sanitize(q: str) -> str:
            q = CypherSanitizer.strip_comments(q)
            if CypherSanitizer.contains_property_exists(q):
                q = CypherSanitizer.replace_property_exists(q)
            if CypherSanitizer.contains_match_after_optional(q):
                q = CypherSanitizer.rewrite_match_after_optional(q)
            q = CypherSanitizer.rewrite_tautologies(q)
            q = CypherSanitizer.ensure_return_distinct(q)
            return CypherSanitizer.first_statement(q)

        async def _generate_once(prompt: str) -> str:
            q = await cypher_gen.generate(prompt)
            q = CypherSanitizer.strip_comments(q)
            if CypherSanitizer.contains_union_syntax(q):
                raise ex.LLMGenerationError(
                    "Generated Cypher uses union syntax (e.g., :A|B or [:A|B]). "
                    "Memgraph does not accept this syntax."
                )
            if CypherSanitizer.contains_bare_node_match(q):
                raise ex.LLMGenerationError("Generated Cypher contains standalone MATCH (n).")
            return q

        async def _generate_with_retries(nl: str) -> str:
            """
            Prefer constrained prompts to reduce invalid Memgraph Cypher.
            Falls back to STRICT MODE only if constraints still fail.
            """
            try:
                return await _generate_once(CypherSanitizer.append_memgraph_constraints(nl))
            except ex.LLMGenerationError:
                return await _generate_once(CypherSanitizer.append_memgraph_strict_constraints(nl))

        try:
            cypher_query = await _generate_with_retries(natural_language_query)

            cypher_query = _sanitize(cypher_query)

            results = await asyncio.to_thread(ingestor.fetch_all, cypher_query)

            total_count = len(results)
            if total_count > settings.QUERY_RESULT_ROW_CAP:
                results = results[: settings.QUERY_RESULT_ROW_CAP]

            results, tokens_used, was_truncated = truncate_results_by_tokens(
                results,
                max_tokens=settings.QUERY_RESULT_MAX_TOKENS,
                original_total=total_count,
            )

            if results:
                table = Table(
                    show_header=True,
                    header_style="bold magenta",
                )
                headers = results[0].keys()
                for header in headers:
                    table.add_column(header)

                for row in results:
                    renderable_values = []
                    for value in row.values():
                        if value is None:
                            renderable_values.append("")
                        elif isinstance(value, bool):
                            renderable_values.append("✓" if value else "✗")
                        elif isinstance(value, int | float):
                            renderable_values.append(str(value))
                        else:
                            renderable_values.append(str(value))
                    table.add_row(*renderable_values)

                console.print(
                    Panel(
                        table,
                        title=QUERY_RESULTS_PANEL_TITLE,
                        expand=False,
                    )
                )

            if was_truncated or total_count > len(results):
                summary = QUERY_SUMMARY_TRUNCATED.format(
                    kept=len(results),
                    total=total_count,
                    tokens=tokens_used,
                    max_tokens=settings.QUERY_RESULT_MAX_TOKENS,
                )
            else:
                summary = QUERY_SUMMARY_SUCCESS.format(count=len(results))
            return QueryGraphData(
                query_used=cypher_query, results=results, summary=summary
            )
        except ex.LLMGenerationError as e:
            return QueryGraphData(
                query_used=QUERY_NOT_AVAILABLE,
                results=[],
                summary=QUERY_SUMMARY_TRANSLATION_FAILED.format(error=e),
            )
        except Exception as e:
            logger.exception(ls.TOOL_QUERY_ERROR.format(error=e))
            return QueryGraphData(
                query_used=cypher_query,
                results=[],
                summary=QUERY_SUMMARY_DB_ERROR.format(error=e),
            )

    return Tool(
        function=query_codebase_knowledge_graph,
        name=td.AgenticToolName.QUERY_GRAPH,
        description=td.CODEBASE_QUERY,
    )
