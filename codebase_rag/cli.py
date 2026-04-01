import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as get_version
from pathlib import Path

import typer
from loguru import logger
from rich.panel import Panel
from rich.table import Table

from . import cli_help as ch
from . import constants as cs
from . import logs as ls
from .bootstrap import connect_memgraph, update_model_settings
from .config import load_cgrignore_patterns, settings
from .context import app_context
from .graph_export import export_graph_to_file, prompt_for_unignored_directories
from .graph_updater import GraphUpdater
from .main import main_async, main_optimize_async, main_single_query
from .parser_loader import load_parsers
from .services.protobuf_service import ProtobufFileIngestor
from .tools.health_checker import HealthChecker
from .tools.language import cli as language_click_app
from .types_defs import ResultRow
from .ui import style

app = typer.Typer(
    name=cs.PACKAGE_NAME,
    help=ch.APP_DESCRIPTION,
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        app_context.console.print(
            cs.CLI_MSG_VERSION.format(
                package=cs.PACKAGE_NAME, version=get_version(cs.PACKAGE_NAME)
            ),
            highlight=False,
        )
        raise typer.Exit()


def validate_models_early() -> None:
    try:
        orchestrator_config = settings.active_orchestrator_config
        orchestrator_config.validate_api_key(cs.ModelRole.ORCHESTRATOR)

        cypher_config = settings.active_cypher_config
        cypher_config.validate_api_key(cs.ModelRole.CYPHER)
    except ValueError as e:
        app_context.console.print(style(str(e), cs.Color.RED))
        raise typer.Exit(1) from e


def _update_and_validate_models(orchestrator: str | None, cypher: str | None) -> None:
    try:
        update_model_settings(orchestrator, cypher)
    except ValueError as e:
        app_context.console.print(style(str(e), cs.Color.RED))
        raise typer.Exit(1) from e

    validate_models_early()


@app.callback()
def _global_options(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help=ch.HELP_VERSION,
        callback=_version_callback,
        is_eager=True,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output (progress messages, banners, informational logs).",
        is_eager=True,
    ),
) -> None:
    settings.QUIET = quiet
    if quiet:
        logger.remove()
        logger.add(lambda msg: app_context.console.print(msg, end=""), level="ERROR")


def _info(msg: str) -> None:
    if not settings.QUIET:
        app_context.console.print(msg)


def _error_and_exit(
    msg: str,
    *,
    code: int = 1,
    hint: str | None = None,
) -> None:
    app_context.console.print(style(msg, cs.Color.RED))
    if hint:
        _info(style(hint, cs.Color.YELLOW))
    raise typer.Exit(code)


def _resolve_exclusions(
    *,
    repo_path: Path,
    exclude: list[str] | None,
    interactive_setup: bool,
) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    cgrignore = load_cgrignore_patterns(repo_path)
    cli_excludes = frozenset(exclude) if exclude else frozenset()
    exclude_paths = cli_excludes | cgrignore.exclude or None

    if interactive_setup:
        unignore_paths = prompt_for_unignored_directories(repo_path, exclude)
    else:
        _info(style(cs.CLI_MSG_AUTO_EXCLUDE, cs.Color.YELLOW))
        unignore_paths = cgrignore.unignore or None

    return exclude_paths, unignore_paths


def _run_clean_only(*, repo_path: Path, batch_size: int) -> None:
    with connect_memgraph(batch_size) as ingestor:
        _info(style(cs.CLI_MSG_CLEANING_DB, cs.Color.YELLOW))
        ingestor.clean_database()

    _delete_hash_cache(repo_path)
    _info(style(cs.CLI_MSG_CLEAN_DONE, cs.Color.GREEN))


def _run_update_graph(
    *,
    repo_path: Path,
    batch_size: int,
    exclude: list[str] | None,
    interactive_setup: bool,
    project_name: str | None,
    clean: bool,
    output: str | None,
) -> None:
    _info(style(cs.CLI_MSG_UPDATING_GRAPH.format(path=repo_path), cs.Color.GREEN))

    exclude_paths, unignore_paths = _resolve_exclusions(
        repo_path=repo_path,
        exclude=exclude,
        interactive_setup=interactive_setup,
    )

    with connect_memgraph(batch_size) as ingestor:
        if clean:
            _info(style(cs.CLI_MSG_CLEANING_DB, cs.Color.YELLOW))
            ingestor.clean_database()
            _delete_hash_cache(repo_path)

        ingestor.ensure_constraints()

        parsers, queries = load_parsers()

        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=repo_path,
            parsers=parsers,
            queries=queries,
            unignore_paths=unignore_paths,
            exclude_paths=exclude_paths,
            project_name=project_name,
        )
        updater.run()

        if output:
            _info(style(cs.CLI_MSG_EXPORTING_TO.format(path=output), cs.Color.CYAN))
            if not export_graph_to_file(ingestor, output):
                raise typer.Exit(1)

    _info(style(cs.CLI_MSG_GRAPH_UPDATED, cs.Color.GREEN))


def _run_chat(
    *,
    target_repo_path: str,
    batch_size: int,
    ask_agent: str | None,
) -> None:
    try:
        if ask_agent:
            main_single_query(target_repo_path, batch_size, ask_agent)
        else:
            asyncio.run(main_async(target_repo_path, batch_size))
    except KeyboardInterrupt:
        app_context.console.print(style(cs.CLI_MSG_APP_TERMINATED, cs.Color.RED))
    except ValueError as e:
        app_context.console.print(style(cs.CLI_ERR_STARTUP.format(error=e), cs.Color.RED))

def _delete_hash_cache(repo_path: Path) -> None:
    cache_path = repo_path / cs.HASH_CACHE_FILENAME
    if cache_path.exists():
        _info(
            style(
                cs.CLI_MSG_CLEANING_HASH_CACHE.format(path=cache_path),
                cs.Color.YELLOW,
            )
        )
        cache_path.unlink(missing_ok=True)


@app.command(help=ch.CMD_START)
def start(
    repo_path: str | None = typer.Option(
        None, "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    update_graph: bool = typer.Option(
        False,
        "--update-graph",
        help=ch.HELP_UPDATE_GRAPH,
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help=ch.HELP_CLEAN_DB,
    ),
    output: str | None = typer.Option(
        None,
        "-o",
        "--output",
        help=ch.HELP_OUTPUT_GRAPH,
    ),
    orchestrator: str | None = typer.Option(
        None,
        "--orchestrator",
        help=ch.HELP_ORCHESTRATOR,
    ),
    cypher: str | None = typer.Option(
        None,
        "--cypher",
        help=ch.HELP_CYPHER_MODEL,
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help=ch.HELP_NO_CONFIRM,
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help=ch.HELP_BATCH_SIZE,
    ),
    project_name: str | None = typer.Option(
        None,
        "--project-name",
        help=ch.HELP_PROJECT_NAME,
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help=ch.HELP_EXCLUDE_PATTERNS,
    ),
    interactive_setup: bool = typer.Option(
        False,
        "--interactive-setup",
        help=ch.HELP_INTERACTIVE_SETUP,
    ),
    ask_agent: str | None = typer.Option(
        None,
        "-a",
        "--ask-agent",
        help=ch.HELP_ASK_AGENT,
    ),
) -> None:
    app_context.session.confirm_edits = not no_confirm

    target_repo_path = repo_path or "."

    if output and not update_graph:
        _error_and_exit(cs.CLI_ERR_OUTPUT_REQUIRES_UPDATE)

    effective_batch_size = settings.resolve_batch_size(batch_size)

    if clean and not update_graph:
        _run_clean_only(repo_path=Path(target_repo_path), batch_size=effective_batch_size)
        return

    _update_and_validate_models(orchestrator, cypher)

    if update_graph:
        _run_update_graph(
            repo_path=Path(target_repo_path),
            batch_size=effective_batch_size,
            exclude=exclude,
            interactive_setup=interactive_setup,
            project_name=project_name,
            clean=clean,
            output=output,
        )
        return

    _run_chat(
        target_repo_path=target_repo_path,
        batch_size=effective_batch_size,
        ask_agent=ask_agent,
    )


@app.command(help=ch.CMD_INDEX)
def index(
    repo_path: str | None = typer.Option(
        None, "--repo-path", help=ch.HELP_REPO_PATH_INDEX
    ),
    output_proto_dir: str = typer.Option(
        ...,
        "-o",
        "--output-proto-dir",
        help=ch.HELP_OUTPUT_PROTO_DIR,
    ),
    split_index: bool = typer.Option(
        False,
        "--split-index",
        help=ch.HELP_SPLIT_INDEX,
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help=ch.HELP_EXCLUDE_PATTERNS,
    ),
    interactive_setup: bool = typer.Option(
        False,
        "--interactive-setup",
        help=ch.HELP_INTERACTIVE_SETUP,
    ),
) -> None:
    target_repo_path = repo_path or "."
    repo_to_index = Path(target_repo_path)
    _info(style(cs.CLI_MSG_INDEXING_AT.format(path=repo_to_index), cs.Color.GREEN))

    _info(style(cs.CLI_MSG_OUTPUT_TO.format(path=output_proto_dir), cs.Color.CYAN))

    exclude_paths, unignore_paths = _resolve_exclusions(
        repo_path=repo_to_index,
        exclude=exclude,
        interactive_setup=interactive_setup,
    )

    try:
        ingestor = ProtobufFileIngestor(
            output_path=output_proto_dir, split_index=split_index
        )
        parsers, queries = load_parsers()
        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=repo_to_index,
            parsers=parsers,
            queries=queries,
            unignore_paths=unignore_paths,
            exclude_paths=exclude_paths,
        )

        updater.run()
        _info(style(cs.CLI_MSG_INDEXING_DONE, cs.Color.GREEN))

    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_INDEXING.format(error=e), cs.Color.RED)
        )
        logger.exception(ls.INDEXING_FAILED)
        raise typer.Exit(1) from e


@app.command(help=ch.CMD_EXPORT)
def export(
    output: str = typer.Option(..., "-o", "--output", help=ch.HELP_OUTPUT_PATH),
    format_json: bool = typer.Option(
        True, "--json/--no-json", help=ch.HELP_FORMAT_JSON
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help=ch.HELP_BATCH_SIZE,
    ),
) -> None:
    if not format_json:
        app_context.console.print(style(cs.CLI_ERR_ONLY_JSON, cs.Color.RED))
        raise typer.Exit(1)

    _info(style(cs.CLI_MSG_CONNECTING_MEMGRAPH, cs.Color.CYAN))

    effective_batch_size = settings.resolve_batch_size(batch_size)

    try:
        with connect_memgraph(effective_batch_size) as ingestor:
            _info(style(cs.CLI_MSG_EXPORTING_DATA, cs.Color.CYAN))

            if not export_graph_to_file(ingestor, output):
                raise typer.Exit(1)

    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_EXPORT_FAILED.format(error=e), cs.Color.RED)
        )
        logger.exception(ls.EXPORT_ERROR.format(error=e))
        raise typer.Exit(1) from e


@app.command(help=ch.CMD_OPTIMIZE)
def optimize(
    language: str = typer.Argument(
        ...,
        help=ch.HELP_LANGUAGE_ARG,
    ),
    repo_path: str | None = typer.Option(
        None, "--repo-path", help=ch.HELP_REPO_PATH_OPTIMIZE
    ),
    reference_document: str | None = typer.Option(
        None,
        "--reference-document",
        help=ch.HELP_REFERENCE_DOC,
    ),
    orchestrator: str | None = typer.Option(
        None,
        "--orchestrator",
        help=ch.HELP_ORCHESTRATOR,
    ),
    cypher: str | None = typer.Option(
        None,
        "--cypher",
        help=ch.HELP_CYPHER_MODEL,
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help=ch.HELP_NO_CONFIRM,
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help=ch.HELP_BATCH_SIZE,
    ),
) -> None:
    app_context.session.confirm_edits = not no_confirm

    target_repo_path = repo_path or "."

    _update_and_validate_models(orchestrator, cypher)

    try:
        asyncio.run(
            main_optimize_async(
                language,
                target_repo_path,
                reference_document,
                orchestrator,
                cypher,
                batch_size,
            )
        )
    except KeyboardInterrupt:
        app_context.console.print(style(cs.CLI_MSG_APP_TERMINATED, cs.Color.RED))
    except ValueError as e:
        app_context.console.print(
            style(cs.CLI_ERR_STARTUP.format(error=e), cs.Color.RED)
        )


@app.command(name=ch.CLICommandName.MCP_SERVER, help=ch.CMD_MCP_SERVER)
def mcp_server(
    transport: cs.MCPTransport = typer.Option(
        cs.MCPTransport.STDIO, help=ch.HELP_MCP_TRANSPORT
    ),
    host: str = typer.Option(None, help=ch.HELP_MCP_HTTP_HOST),
    port: int = typer.Option(None, help=ch.HELP_MCP_HTTP_PORT),
) -> None:
    try:
        if transport == cs.MCPTransport.HTTP:
            from codebase_rag.mcp import serve_http

            resolved_host = host or settings.MCP_HTTP_HOST
            resolved_port = port or settings.MCP_HTTP_PORT
            asyncio.run(serve_http(host=resolved_host, port=resolved_port))
        else:
            from codebase_rag.mcp import serve_stdio

            asyncio.run(serve_stdio())
    except KeyboardInterrupt:
        app_context.console.print(style(cs.CLI_MSG_APP_TERMINATED, cs.Color.RED))
    except ValueError as e:
        _error_and_exit(
            cs.CLI_ERR_CONFIG.format(error=e),
            hint=cs.CLI_MSG_HINT_TARGET_REPO,
        )
    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_MCP_SERVER.format(error=e), cs.Color.RED)
        )


@app.command(name=ch.CLICommandName.GRAPH_LOADER, help=ch.CMD_GRAPH_LOADER)
def graph_loader_command(
    graph_file: str = typer.Argument(..., help=ch.HELP_GRAPH_FILE),
) -> None:
    from .graph_loader import load_graph

    try:
        graph = load_graph(graph_file)
        summary = graph.summary()

        app_context.console.print(style(cs.CLI_MSG_GRAPH_SUMMARY, cs.Color.GREEN))
        app_context.console.print(f"  Total nodes: {summary['total_nodes']}")
        app_context.console.print(
            f"  Total relationships: {summary['total_relationships']}"
        )
        app_context.console.print(
            f"  Node types: {list(summary['node_labels'].keys())}"
        )
        app_context.console.print(
            f"  Relationship types: {list(summary['relationship_types'].keys())}"
        )
        app_context.console.print(
            f"  Exported at: {summary['metadata']['exported_at']}"
        )

    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_LOAD_GRAPH.format(error=e), cs.Color.RED)
        )
        raise typer.Exit(1) from e


@app.command(
    name=ch.CLICommandName.LANGUAGE,
    help=ch.CMD_LANGUAGE,
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def language_command(ctx: typer.Context) -> None:
    try:
        language_click_app.main(
            args=list(ctx.args),
            prog_name=f"{cs.PACKAGE_NAME} {ch.CLICommandName.LANGUAGE}",
            standalone_mode=False,
        )
    except SystemExit as e:
        raise typer.Exit(e.code) from e


@app.command(name=ch.CLICommandName.DOCTOR, help=ch.CMD_DOCTOR)
def doctor() -> None:
    checker = HealthChecker()
    results = checker.run_all_checks()

    passed, total = checker.get_summary()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", no_wrap=False)

    for result in results:
        status = "✓" if result.passed else "✗"
        status_color = cs.Color.GREEN if result.passed else cs.Color.RED
        status_text = style(status, status_color, cs.StyleModifier.NONE)

        check_name = f"{status_text} {result.name}"
        table.add_row(check_name)

    panel = Panel(
        table,
        title="Health Check",
        border_style="dim",
        padding=(1, 2),
    )

    app_context.console.print(panel)

    app_context.console.print()
    summary_text = f"{passed}/{total} checks passed"
    if passed == total:
        app_context.console.print(style(summary_text, cs.Color.GREEN))
    else:
        app_context.console.print(style(summary_text, cs.Color.YELLOW))

    failed_checks = [r for r in results if not r.passed and r.error]
    if failed_checks:
        app_context.console.print()
        app_context.console.print(style("Failed checks details:", cs.Color.YELLOW))
        for result in failed_checks:
            error_msg = f"  {result.name}: {result.error}"
            app_context.console.print(
                style(error_msg, cs.Color.YELLOW, cs.StyleModifier.NONE)
            )

    if passed < total:
        raise typer.Exit(1)


def _build_stats_table(
    title: str,
    col_label: str,
    rows: list[ResultRow],
    get_label: Callable[[ResultRow], str],
    total_label: str,
) -> Table:
    table = Table(
        title=style(title, cs.Color.GREEN),
        show_header=True,
        header_style=f"{cs.StyleModifier.BOLD} {cs.Color.MAGENTA}",
    )
    table.add_column(col_label, style=cs.Color.CYAN)
    table.add_column(cs.CLI_STATS_COL_COUNT, style=cs.Color.YELLOW, justify="right")
    total = 0
    for row in rows:
        raw_count = row.get("count", 0)
        count = int(raw_count) if isinstance(raw_count, (int, float)) else 0
        total += count
        table.add_row(get_label(row), f"{count:,}")
    table.add_section()
    table.add_row(
        style(total_label, cs.Color.GREEN),
        style(f"{total:,}", cs.Color.GREEN),
    )
    return table


@app.command(name=ch.CLICommandName.STATS, help=ch.CMD_STATS)
def stats() -> None:
    from .cypher_queries import (
        CYPHER_STATS_NODE_COUNTS,
        CYPHER_STATS_RELATIONSHIP_COUNTS,
    )

    app_context.console.print(style(cs.CLI_MSG_CONNECTING_STATS, cs.Color.CYAN))

    try:
        with connect_memgraph(batch_size=1) as ingestor:
            node_results = ingestor.fetch_all(CYPHER_STATS_NODE_COUNTS)
            rel_results = ingestor.fetch_all(CYPHER_STATS_RELATIONSHIP_COUNTS)

            app_context.console.print(
                _build_stats_table(
                    cs.CLI_STATS_NODE_TITLE,
                    cs.CLI_STATS_COL_NODE_TYPE,
                    node_results,
                    lambda r: ":".join(r.get("labels", [])) or cs.CLI_STATS_UNKNOWN,
                    cs.CLI_STATS_TOTAL_NODES,
                )
            )
            app_context.console.print()
            app_context.console.print(
                _build_stats_table(
                    cs.CLI_STATS_REL_TITLE,
                    cs.CLI_STATS_COL_REL_TYPE,
                    rel_results,
                    lambda r: str(r.get("type", cs.CLI_STATS_UNKNOWN)),
                    cs.CLI_STATS_TOTAL_RELS,
                )
            )

    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_STATS_FAILED.format(error=e), cs.Color.RED)
        )
        logger.exception(ls.STATS_ERROR.format(error=e))
        raise typer.Exit(1) from e


@app.command(name=ch.CLICommandName.CONSUMERS, help=ch.CMD_CONSUMERS)
def consumers(
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help=ch.HELP_NO_CONFIRM,
    ),
    no_kafka_consumers: bool = typer.Option(
        False,
        "--no-kafka-consumers",
        help=ch.HELP_NO_EMBEDDED_KAFKA_CONSUMERS,
    ),
    no_kafka_index_consumer: bool = typer.Option(
        False,
        "--no-kafka-index-consumer",
        help=ch.HELP_NO_EMBEDDED_KAFKA_INDEX_CONSUMER,
    ),
    no_kafka_chat_consumer: bool = typer.Option(
        False,
        "--no-kafka-chat-consumer",
        help=ch.HELP_NO_EMBEDDED_KAFKA_CHAT_CONSUMER,
    ),
) -> None:
    """
    Run embedded Kafka consumers without starting an HTTP server.
    """
    import os
    import signal

    @dataclass(frozen=True)
    class _ConsumersEnvConfig:
        no_confirm: bool
        no_kafka_consumers: bool
        no_kafka_index_consumer: bool
        no_kafka_chat_consumer: bool

        def apply(self) -> None:
            if self.no_confirm:
                os.environ["CGR_NO_CONFIRM"] = "1"
            if self.no_kafka_consumers:
                os.environ["CGR_EMBEDDED_KAFKA_CONSUMERS"] = "0"
            elif self.no_kafka_index_consumer:
                os.environ["CGR_EMBEDDED_KAFKA_INDEX_CONSUMERS"] = "0"
            if not self.no_kafka_consumers and self.no_kafka_chat_consumer:
                os.environ["CGR_EMBEDDED_KAFKA_CHAT_CONSUMERS"] = "0"

    _ConsumersEnvConfig(
        no_confirm=no_confirm,
        no_kafka_consumers=no_kafka_consumers,
        no_kafka_index_consumer=no_kafka_index_consumer,
        no_kafka_chat_consumer=no_kafka_chat_consumer,
    ).apply()

    # Run Kafka-only consumer runner and block.
    from .api import run_kafka_consumers_until

    app_context.console.print(
        style("Starting embedded Kafka consumers (no HTTP server)...", cs.Color.GREEN)
    )

    stop = False

    def _handle(_sig: int, _frame: object | None = None) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    try:
        import anyio
    except ImportError:
        app_context.console.print(
            style(
                "anyio is not installed. Please install it using your project dependencies.",
                cs.Color.RED,
            )
        )
        raise typer.Exit(1)

    async def _run() -> None:
        nonlocal stop
        import asyncio

        stop_event = asyncio.Event()
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_kafka_consumers_until, stop_event)
            while not stop:
                await anyio.sleep(0.2)
            stop_event.set()

    anyio.run(_run)


if __name__ == "__main__":
    app()
