from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from loguru import logger

from . import constants as cs
from . import exceptions as ex
from . import logs as ls
from .config import ModelConfig, settings
from .context import app_context
from .services import QueryProtocol
from .services.graph_service import MemgraphIngestor
from .services.llm import CypherGenerator, create_rag_orchestrator
from .tools.code_retrieval import CodeRetriever, create_code_retrieval_tool
from .tools.codebase_query import create_query_tool
from .tools.directory_lister import DirectoryLister, create_directory_lister_tool
from .tools.document_analyzer import DocumentAnalyzer, create_document_analyzer_tool
from .tools.file_editor import FileEditor, create_file_editor_tool
from .tools.file_reader import FileReader, create_file_reader_tool
from .tools.file_writer import FileWriter, create_file_writer_tool
from .tools.semantic_search import (
    create_get_function_source_tool,
    create_semantic_search_tool,
)
from .tools.shell_command import ShellCommander, create_shell_command_tool
from .types_defs import ConfirmationToolNames
from .utils.cache import EvictingCache
from .utils.dependencies import has_semantic_dependencies
from .utils.error_handling import optional_section

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model


class _ServiceBundle:
    __slots__ = (
        "code_retriever",
        "file_reader",
        "file_writer",
        "file_editor",
        "shell_commander",
        "directory_lister",
        "document_analyzer",
    )

    def __init__(
        self,
        *,
        code_retriever: CodeRetriever,
        file_reader: FileReader,
        file_writer: FileWriter,
        file_editor: FileEditor,
        shell_commander: ShellCommander,
        directory_lister: DirectoryLister,
        document_analyzer: DocumentAnalyzer,
    ) -> None:
        self.code_retriever = code_retriever
        self.file_reader = file_reader
        self.file_writer = file_writer
        self.file_editor = file_editor
        self.shell_commander = shell_commander
        self.directory_lister = directory_lister
        self.document_analyzer = document_analyzer


_SERVICE_CACHE_LOCK = Lock()
_SERVICE_CACHE = EvictingCache[tuple[str, int], _ServiceBundle](
    max_entries=settings.SERVICE_CACHE_MAX_ENTRIES,
    max_size=10**18,  # effectively disabled; LRU by entry count
    size_func=lambda _v: 1,
)


def setup_common_initialization(repo_path: str) -> Path:
    logger.remove()
    logger.add(sys.stdout, format=cs.LOG_FORMAT)

    project_root = Path(repo_path).resolve()
    tmp_dir = project_root / cs.TMP_DIR
    if tmp_dir.exists():
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir)
        else:
            tmp_dir.unlink()
    tmp_dir.mkdir()

    return project_root


def prewarm_semantic_model() -> None:
    """Shared semantic-search prewarm used by both CLI and API."""
    if not settings.SEMANTIC_SEARCH_ENABLED:
        return
    if not has_semantic_dependencies():
        return
    try:
        from .embedder import prewarm_embeddings

        logger.info(ls.SEMANTIC_PREWARM_START)
        prewarm_embeddings()
        logger.info(ls.SEMANTIC_PREWARM_COMPLETE)
    except Exception as e:  # noqa: BLE001
        logger.warning(ls.SEMANTIC_PREWARM_FAILED.format(error=e))


def warm_core_db() -> None:
    """Warm the core database connection (shared between CLI and API)."""
    from .utils.org_region_resolver import get_org_region_resolver

    with optional_section("Core DB warm"):
        get_org_region_resolver().warm_core_connection()


def connect_memgraph(batch_size: int) -> MemgraphIngestor:
    return MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=batch_size,
        username=settings.MEMGRAPH_USERNAME,
        password=settings.MEMGRAPH_PASSWORD,
    )


def _validate_provider_config(role: cs.ModelRole, config: ModelConfig) -> None:
    from .providers.base import get_provider_from_config

    try:
        provider = get_provider_from_config(config)
        provider.validate_config()
    except Exception as e:  # noqa: BLE001
        raise ValueError(ex.CONFIG.format(role=role.value.title(), error=e)) from e


def _create_model_from_string(
    model_string: str, current_override_config: ModelConfig | None = None
) -> tuple[Model, str, ModelConfig]:
    from .providers.base import get_provider_from_config

    base_config = current_override_config or settings.active_orchestrator_config

    if cs.CHAR_COLON not in model_string:
        raise ValueError(ex.MODEL_FORMAT_INVALID)
    provider_name, model_id = (
        p.strip() for p in settings.parse_model_string(model_string)
    )
    if not model_id:
        raise ValueError(ex.MODEL_ID_EMPTY)
    if not provider_name:
        raise ValueError(ex.PROVIDER_EMPTY)

    if provider_name == base_config.provider:
        config = replace(base_config, model_id=model_id)
    elif provider_name == cs.Provider.OLLAMA:
        config = ModelConfig(
            provider=provider_name,
            model_id=model_id,
            endpoint=settings.ollama_endpoint,
            api_key=cs.DEFAULT_API_KEY,
        )
    else:
        config = ModelConfig(provider=provider_name, model_id=model_id)

    canonical_string = f"{provider_name}{cs.CHAR_COLON}{model_id}"
    provider = get_provider_from_config(config)
    return provider.create_model(model_id), canonical_string, config


def update_model_settings(
    orchestrator: str | None,
    cypher: str | None,
) -> None:
    if orchestrator:
        _update_single_model_setting(cs.ModelRole.ORCHESTRATOR, orchestrator)
    if cypher:
        _update_single_model_setting(cs.ModelRole.CYPHER, cypher)


def _update_single_model_setting(role: cs.ModelRole, model_string: str) -> None:
    provider, model = settings.parse_model_string(model_string)

    match role:
        case cs.ModelRole.ORCHESTRATOR:
            current_config = settings.active_orchestrator_config
            set_method = settings.set_orchestrator
        case cs.ModelRole.CYPHER:
            current_config = settings.active_cypher_config
            set_method = settings.set_cypher

    kwargs = current_config.to_update_kwargs()

    if provider == cs.Provider.OLLAMA and not kwargs[cs.FIELD_ENDPOINT]:
        kwargs[cs.FIELD_ENDPOINT] = settings.ollama_endpoint
        kwargs[cs.FIELD_API_KEY] = cs.DEFAULT_API_KEY

    set_method(provider, model, **kwargs)


def initialize_services_and_agent(
    repo_path: str,
    ingestor: QueryProtocol,
    system_prompt: str | None = None,
) -> tuple[Agent[None, str], ConfirmationToolNames]:
    _validate_provider_config(
        cs.ModelRole.ORCHESTRATOR, settings.active_orchestrator_config
    )
    _validate_provider_config(cs.ModelRole.CYPHER, settings.active_cypher_config)

    cypher_generator = CypherGenerator()

    repo_key = str(Path(repo_path).resolve())
    cache_key = (repo_key, id(ingestor))
    with _SERVICE_CACHE_LOCK:
        bundle = _SERVICE_CACHE.get(cache_key)
        if bundle is None:
            bundle = _ServiceBundle(
                code_retriever=CodeRetriever(project_root=repo_key, ingestor=ingestor),
                file_reader=FileReader(project_root=repo_key),
                file_writer=FileWriter(project_root=repo_key),
                file_editor=FileEditor(project_root=repo_key),
                shell_commander=ShellCommander(
                    project_root=repo_key, timeout=settings.SHELL_COMMAND_TIMEOUT
                ),
                directory_lister=DirectoryLister(project_root=repo_key),
                document_analyzer=DocumentAnalyzer(project_root=repo_key),
            )
            _SERVICE_CACHE.put(cache_key, bundle)

    code_retriever = bundle.code_retriever
    file_reader = bundle.file_reader
    file_writer = bundle.file_writer
    file_editor = bundle.file_editor
    shell_commander = bundle.shell_commander
    directory_lister = bundle.directory_lister
    document_analyzer = bundle.document_analyzer

    query_tool = create_query_tool(ingestor, cypher_generator, app_context.console)
    code_tool = create_code_retrieval_tool(code_retriever)
    file_reader_tool = create_file_reader_tool(file_reader)
    file_writer_tool = create_file_writer_tool(file_writer)
    file_editor_tool = create_file_editor_tool(file_editor)
    shell_command_tool = create_shell_command_tool(shell_commander)
    directory_lister_tool = create_directory_lister_tool(directory_lister)
    document_analyzer_tool = create_document_analyzer_tool(document_analyzer)
    semantic_search_tool = create_semantic_search_tool(ingestor=ingestor)
    function_source_tool = create_get_function_source_tool(ingestor=ingestor)

    confirmation_tool_names = ConfirmationToolNames(
        replace_code=file_editor_tool.name,
        create_file=file_writer_tool.name,
        shell_command=shell_command_tool.name,
    )

    rag_agent = create_rag_orchestrator(
        tools=[
            query_tool,
            code_tool,
            file_reader_tool,
            file_writer_tool,
            file_editor_tool,
            shell_command_tool,
            directory_lister_tool,
            document_analyzer_tool,
            semantic_search_tool,
            function_source_tool,
        ],
        system_prompt=system_prompt,
    )
    return rag_agent, confirmation_tool_names

