from __future__ import annotations

import os

# AppConfig (strict mode) requires every key from .env.example in os.environ.
# Seed new optional keys so pytest can import before a developer merges .env.example.
def _seed_strict_env_example_keys() -> None:
    for key, val in (
        ("CORE_DB_HOST", ""),
        ("CORE_DB_PORT", ""),
        ("CORE_DB_USER", ""),
        ("CORE_DB_PASSWORD", ""),
        ("CORE_DB_NAME", ""),
        ("CORE_DB_SSL", ""),
        ("ORG_DB_HOST_1", ""),
        ("ORG_DB_PORT_1", ""),
        ("ORG_DB_USER_1", ""),
        ("ORG_DB_PASSWORD_1", ""),
        ("ORG_DB_NAME_1", ""),
        ("ORG_DB_SSL_1", ""),
        ("ORG_DB_HOST_2", ""),
        ("ORG_DB_PORT_2", ""),
        ("ORG_DB_USER_2", ""),
        ("ORG_DB_PASSWORD_2", ""),
        ("ORG_DB_NAME_2", ""),
        ("ORG_DB_SSL_2", ""),
        ("DB_POOL_MIN_SIZE", "2"),
        ("DB_POOL_MAX_SIZE", "10"),
        ("DB_COMMAND_TIMEOUT", "60"),
        ("DB_CONNECT_TIMEOUT", "30"),
        ("KAFKA_BOOTSTRAP_SERVERS", ""),
        ("KAFKA_CHAT_JOBS_TOPIC", "cgr.chat.jobs"),
        ("KAFKA_CHAT_CONSUMER_GROUP_ID", "cgr-chat-jobs"),
        ("KAFKA_CHAT_MAX_CONCURRENCY", "4"),
        ("KAFKA_CHAT_AUTO_OFFSET_RESET", "latest"),
        ("KAFKA_CHAT_SHUTDOWN_GRACE_SECONDS", "30"),
        ("KAFKA_CHAT_FETCH_MAX_WAIT_MS", "500"),
        ("KAFKA_CHAT_SESSION_TIMEOUT_MS", "30000"),
        ("KAFKA_CHAT_RECONNECT_MAX_SECONDS", "60"),
        ("KAFKA_CHAT_RECONNECT_BACKOFF_INITIAL", "1"),
        ("KAFKA_CHAT_TOPIC_NUM_PARTITIONS", "3"),
        ("KAFKA_CHAT_TOPIC_REPLICATION_FACTOR", "1"),
        ("KAFKA_INDEX_JOBS_TOPIC", "cgr.index.jobs"),
        ("KAFKA_INDEX_CONSUMER_GROUP_ID", "cgr-index-jobs"),
        ("KAFKA_INDEX_MAX_CONCURRENCY", "2"),
        ("KAFKA_INDEX_AUTO_OFFSET_RESET", "latest"),
        ("KAFKA_INDEX_SHUTDOWN_GRACE_SECONDS", "30"),
        ("KAFKA_INDEX_FETCH_MAX_WAIT_MS", "500"),
        ("KAFKA_INDEX_SESSION_TIMEOUT_MS", "30000"),
        ("KAFKA_INDEX_RECONNECT_MAX_SECONDS", "60"),
        ("KAFKA_INDEX_RECONNECT_BACKOFF_INITIAL", "1"),
        ("KAFKA_INDEX_TOPIC_NUM_PARTITIONS", "1"),
        ("KAFKA_INDEX_TOPIC_REPLICATION_FACTOR", "1"),
    ):
        os.environ.setdefault(key, val)


_seed_strict_env_example_keys()

import shutil
import sys
import tempfile
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self
from unittest.mock import MagicMock, call

import pytest
from loguru import logger

from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers

if TYPE_CHECKING:
    pass  # ty: ignore[unresolved-import]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class NodeProtocol(Protocol):
    @property
    def type(self) -> str: ...
    @property
    def children(self) -> list[Self]: ...
    @property
    def parent(self) -> Self | None: ...
    @property
    def text(self) -> bytes: ...
    def child_by_field_name(self, name: str) -> Self | None: ...


@dataclass
class MockNode:
    node_type: str
    node_children: list[MockNode] = field(default_factory=list)
    node_parent: MockNode | None = None
    node_fields: dict[str, MockNode | None] = field(default_factory=dict)
    node_text: bytes = b""

    @property
    def type(self) -> str:
        return self.node_type

    @property
    def children(self) -> list[MockNode]:
        return self.node_children

    @property
    def parent(self) -> MockNode | None:
        return self.node_parent

    @parent.setter
    def parent(self, value: MockNode | None) -> None:
        self.node_parent = value

    @property
    def text(self) -> bytes:
        return self.node_text

    def child_by_field_name(self, name: str) -> MockNode | None:
        return self.node_fields.get(name)


def create_mock_node(
    node_type: str,
    text: str = "",
    fields: dict[str, MockNode | None] | None = None,
    children: list[MockNode] | None = None,
    parent: MockNode | None = None,
) -> MockNode:
    node = MockNode(
        node_type=node_type,
        node_children=children or [],
        node_parent=parent,
        node_fields=fields or {},
        node_text=text.encode(),
    )
    for child in node.node_children:
        child.node_parent = node
    return node


logger.remove()


@pytest.fixture
def temp_repo() -> Generator[Path, None, None]:
    """Creates a temporary repository path for a test and cleans up afterward."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class _MockIngestor:
    _TRACKED = (
        "fetch_all",
        "execute_write",
        "ensure_node_batch",
        "ensure_relationship_batch",
        "flush_all",
    )

    def __init__(self) -> None:
        self.fetch_all = MagicMock()
        self.execute_write = MagicMock()
        self.ensure_node_batch = MagicMock()
        self.ensure_relationship_batch = MagicMock()
        self.flush_all = MagicMock()
        self._fallback = MagicMock()

    def reset_mock(self) -> None:
        for name in (*self._TRACKED, "_fallback"):
            getattr(self, name).reset_mock()

    @property
    def method_calls(self) -> list:
        result = []
        for name in self._TRACKED:
            mock_attr = self.__dict__[name]
            for c in mock_attr.call_args_list:
                result.append(getattr(call, name)(*c.args, **c.kwargs))
        result.extend(self._fallback.method_calls)
        return result

    def __getattr__(self, name: str) -> MagicMock:
        return getattr(self._fallback, name)


@pytest.fixture
def mock_ingestor() -> _MockIngestor:
    return _MockIngestor()


def run_updater(
    repo_path: Path, mock_ingestor: MagicMock, skip_if_missing: str | None = None
) -> None:
    create_and_run_updater(repo_path, mock_ingestor, skip_if_missing)


def create_and_run_updater(
    repo_path: Path, mock_ingestor: MagicMock, skip_if_missing: str | None = None
) -> GraphUpdater:
    parsers, queries = load_parsers()
    if skip_if_missing and skip_if_missing not in parsers:
        pytest.skip(f"{skip_if_missing} parser not available")
    updater = GraphUpdater(
        ingestor=mock_ingestor,
        repo_path=repo_path,
        parsers=parsers,
        queries=queries,
    )
    updater.run()
    return updater


def get_relationships(mock_ingestor: MagicMock, rel_type: str) -> list:
    """Extract relationships of a specific type from mock_ingestor calls."""
    return [
        c
        for c in mock_ingestor.ensure_relationship_batch.call_args_list
        if c.args[1] == rel_type
    ]


def get_nodes(mock_ingestor: MagicMock, node_type: str) -> list:
    """Extract nodes of a specific type from mock_ingestor calls."""
    return [
        call
        for call in mock_ingestor.ensure_node_batch.call_args_list
        if call[0][0] == node_type
    ]


def get_qualified_names(calls: list) -> set[str]:
    """Extract qualified names from a list of node calls."""
    return {call[0][1]["qualified_name"] for call in calls}


def get_node_names(mock_ingestor: MagicMock, node_type: str) -> set[str]:
    """Get qualified names of all nodes of a specific type."""
    return get_qualified_names(get_nodes(mock_ingestor, node_type))


@pytest.fixture
def mock_updater(temp_repo: Path, mock_ingestor: MagicMock) -> MagicMock:
    """Provides a mocked GraphUpdater instance with necessary dependencies."""
    parsers, queries = load_parsers()
    mock = MagicMock(spec=GraphUpdater)
    mock.repo_path = temp_repo
    mock.ingestor = mock_ingestor
    mock.parsers = parsers
    mock.queries = queries

    mock.factory = MagicMock()
    mock.factory.definition_processor = MagicMock()
    mock.factory.structure_processor = MagicMock()
    mock.factory.structure_processor.structural_elements = {}

    mock_root_node = MagicMock()
    mock.factory.definition_processor.process_file.return_value = (
        mock_root_node,
        "python",
    )

    mock.ast_cache = {}

    return mock


@pytest.fixture(scope="session", autouse=True)
def cleanup_pgvector_client() -> Generator[None, None, None]:
    yield

    try:
        from codebase_rag.utils.dependencies import has_pgvector_client

        if has_pgvector_client():
            import codebase_rag.vector_store as vs

            vs.close_pgvector_client()
    except Exception:
        pass
