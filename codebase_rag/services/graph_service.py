from __future__ import annotations

import threading
import types
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import mgclient  # ty: ignore[unresolved-import]
from loguru import logger

from codebase_rag.config import settings
from codebase_rag.types_defs import CursorProtocol, ResultValue

from .. import exceptions as ex
from .. import logs as ls
from ..constants import (
    ERR_SUBSTR_ALREADY_EXISTS,
    ERR_SUBSTR_CONSTRAINT,
    KEY_CREATED,
    KEY_FROM_VAL,
    KEY_NAME,
    KEY_PROJECT_NAME,
    KEY_PROPS,
    KEY_TO_VAL,
    NODE_UNIQUE_CONSTRAINTS,
    REL_TYPE_CALLS,
)
from ..cypher_queries import (
    CYPHER_DELETE_ALL,
    CYPHER_DELETE_PROJECT,
    CYPHER_EXPORT_NODES,
    CYPHER_EXPORT_RELATIONSHIPS,
    CYPHER_LIST_PROJECTS,
    build_constraint_query,
    build_create_node_query,
    build_create_relationship_query,
    build_index_query,
    build_merge_node_query,
    build_merge_relationship_query,
    wrap_with_unwind,
)
from ..types_defs import (
    BatchParams,
    BatchWrapper,
    GraphData,
    GraphMetadata,
    NodeBatchRow,
    PropertyDict,
    PropertyValue,
    RelBatchRow,
    ResultRow,
)


class MemgraphConnection:
    """Lightweight adapter around mgclient connection and basic query helpers."""

    __slots__ = ("_host", "_port", "_username", "_password", "_conn", "_lock")

    def __init__(
        self,
        host: str,
        port: int,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._lock = threading.Lock()
        self._conn: mgclient.Connection | None = None

    @property
    def host(self) -> str: return self._host
    @property
    def port(self) -> int: return self._port
    @property
    def username(self) -> str | None: return self._username
    @property
    def password(self) -> str | None: return self._password

    def open(self) -> None:
        logger.info(ls.MG_CONNECTING.format(host=self._host, port=self._port))
        self._conn = self._create_connection()
        logger.info(ls.MG_CONNECTED)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info(ls.MG_DISCONNECTED)

    @contextmanager
    def cursor(self) -> Generator[CursorProtocol, None, None]:
        if not self._conn:
            raise ConnectionError(ex.CONN)
        with self._lock:
            cursor: CursorProtocol | None = None
            try:
                cursor = self._conn.cursor()
                yield cursor
            finally:
                if cursor:
                    cursor.close()

    def execute(
        self,
        query: str,
        params: dict[str, PropertyValue] | None = None,
    ) -> list[ResultRow]:
        params = params or {}
        with self.cursor() as cursor:
            try:
                cursor.execute(query, params)
                return self._cursor_to_results(cursor)
            except Exception as e:  # noqa: BLE001
                if (
                    ERR_SUBSTR_ALREADY_EXISTS not in str(e).lower()
                    and ERR_SUBSTR_CONSTRAINT not in str(e).lower()
                ):
                    logger.error(ls.MG_CYPHER_ERROR.format(error=e))
                    logger.error(ls.MG_CYPHER_QUERY.format(query=query))
                    logger.error(ls.MG_CYPHER_PARAMS.format(params=params))
                raise

    def execute_batch(
        self,
        query: str,
        params_list: Sequence[BatchParams],
    ) -> None:
        if not params_list:
            return

        cursor = None
        try:
            assert self._conn is not None, ex.CONN
            cursor = self._conn.cursor()
            cursor.execute(wrap_with_unwind(query), BatchWrapper(batch=params_list))
        except Exception as e:  # noqa: BLE001
            if ERR_SUBSTR_ALREADY_EXISTS not in str(e).lower():
                logger.error(ls.MG_BATCH_ERROR.format(error=e))
                logger.error(ls.MG_CYPHER_QUERY.format(query=query))
                if len(params_list) > 10:
                    logger.error(
                        ls.MG_BATCH_PARAMS_TRUNCATED.format(
                            count=len(params_list), params=params_list[:10]
                        )
                    )
                else:
                    logger.error(ls.MG_CYPHER_PARAMS.format(params=params_list))
            raise
        finally:
            if cursor:
                cursor.close()

    def execute_batch_with_return(
        self,
        query: str,
        params_list: Sequence[BatchParams],
    ) -> list[ResultRow]:
        if not params_list:
            return []

        cursor = None
        try:
            assert self._conn is not None, ex.CONN
            cursor = self._conn.cursor()
            cursor.execute(wrap_with_unwind(query), BatchWrapper(batch=params_list))
            return self._cursor_to_results(cursor)
        except Exception as e:  # noqa: BLE001
            logger.error(ls.MG_BATCH_ERROR.format(error=e))
            logger.error(ls.MG_CYPHER_QUERY.format(query=query))
            raise
        finally:
            if cursor:
                cursor.close()

    def _create_connection(self) -> mgclient.Connection:
        if self._username is not None:
            conn = mgclient.connect(
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
            )
        else:
            conn = mgclient.connect(host=self._host, port=self._port)
        conn.autocommit = True
        return conn

    @staticmethod
    def _cursor_to_results(cursor: CursorProtocol) -> list[ResultRow]:
        if not cursor.description:
            return []
        column_names = [desc.name for desc in cursor.description]
        return [
            dict[str, ResultValue](zip(column_names, row)) for row in cursor.fetchall()
        ]


class BaseBatchFlusher:
    """Common logic for buffering and flushing batches with thread pool support."""

    __slots__ = ("_connection", "_use_merge", "_batch_size", "_executor")

    def __init__(
        self,
        connection: MemgraphConnection,
        *,
        use_merge: bool,
        batch_size: int,
    ) -> None:
        self._connection = connection
        self._use_merge = use_merge
        self._batch_size = batch_size
        self._executor: ThreadPoolExecutor | None = None

    def start(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=settings.FLUSH_THREAD_POOL_SIZE)

    def stop(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _call_with_own_conn(self, fn: Callable, *args) -> tuple[int, int]:
        """Runs a function with its own dedicated connection for thread safety."""
        worker_connection = MemgraphConnection(
            host=self._connection.host,
            port=self._connection.port,
            username=self._connection.username,
            password=self._connection.password,
        )
        worker_connection.open()
        try:
            return fn(*args, conn=worker_connection)
        finally:
            worker_connection.close()

    def _run_parallel_flush(
        self,
        groups: dict[Any, list[Any]],
        flush_fn: Callable,
        log_label: str,
        error_msg_fmt: str,
        workers: int,
    ) -> tuple[int, int, Exception | None]:
        if not self._executor:
            return 0, 0, None

        logger.info(log_label.format(count=len(groups), workers=workers))

        # Helper to wrap the flush function for thread pool submission
        def _task(key: Any, items: list[Any]):
            return self._call_with_own_conn(flush_fn, key, items)

        futures = {self._executor.submit(_task, key, val): key for key, val in groups.items()}

        flushed_total = 0
        skipped_total = 0
        first_error = None

        for future in as_completed(futures):
            key = futures[future]
            try:
                flushed, skipped = future.result()
                flushed_total += flushed
                skipped_total += skipped
            except Exception as e:  # noqa: BLE001
                logger.error(error_msg_fmt.format(key=key, error=e))
                if first_error is None:
                    first_error = e

        return flushed_total, skipped_total, first_error


class NodeBatchFlusher(BaseBatchFlusher):
    """Handles buffering and flushing of node batches to Memgraph."""

    __slots__ = ("_buffer",)

    def __init__(
        self,
        connection: MemgraphConnection,
        *,
        use_merge: bool,
        batch_size: int,
    ) -> None:
        super().__init__(connection, use_merge=use_merge, batch_size=batch_size)
        self._buffer: list[tuple[str, dict[str, PropertyValue]]] = []

    def ensure_node_batch(
        self,
        label: str,
        properties: dict[str, PropertyValue],
    ) -> None:
        self._buffer.append((label, properties))
        if len(self._buffer) >= self._batch_size:
            logger.debug(ls.MG_NODE_BUFFER_FLUSH, size=self._batch_size)
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        buffer_size = len(self._buffer)
        nodes_by_label: defaultdict[str, list[dict[str, PropertyValue]]] = defaultdict(list)
        for label, props in self._buffer:
            nodes_by_label[label].append(props)

        flushed_total = 0
        skipped_total = 0
        first_error = None

        if self._executor and len(nodes_by_label) > 1:
            flushed_total, skipped_total, first_error = self._run_parallel_flush(
                groups=dict(nodes_by_label),
                flush_fn=self._flush_node_label_group,
                log_label=ls.MG_PARALLEL_FLUSH_NODES,
                error_msg_fmt=ls.MG_LABEL_FLUSH_ERROR.replace("{label}", "{key}"), # Adapter
                workers=settings.FLUSH_THREAD_POOL_SIZE,
            )
        else:
            for label, props_list in nodes_by_label.items():
                try:
                    flushed, skipped = self._flush_node_label_group(label, props_list)
                    flushed_total += flushed
                    skipped_total += skipped
                except Exception as e:  # noqa: BLE001
                    logger.error(ls.MG_LABEL_FLUSH_ERROR.format(label=label, error=e))
                    if first_error is None:
                        first_error = e

        logger.info(
            ls.MG_NODES_FLUSHED.format(flushed=flushed_total, total=buffer_size)
        )
        if skipped_total:
            logger.info(ls.MG_NODES_SKIPPED.format(count=skipped_total))
        self._buffer.clear()

        if first_error is not None:
            raise first_error


    def _flush_node_label_group(
        self,
        label: str,
        props_list: list[dict[str, PropertyValue]],
        conn: MemgraphConnection | None = None,
    ) -> tuple[int, int]:
        if not props_list:
            return 0, 0

        id_key = NODE_UNIQUE_CONSTRAINTS.get(label)
        if not id_key:
            logger.warning(ls.MG_NO_CONSTRAINT.format(label=label))
            return 0, len(props_list)

        batch_rows: list[NodeBatchRow] = []
        skipped = 0
        for props in props_list:
            if id_key not in props:
                logger.warning(
                    ls.MG_MISSING_PROP.format(
                        label=label, key=id_key, prop_keys=list(props.keys())
                    )
                )
                skipped += 1
                continue
            row_props: PropertyDict = {k: v for k, v in props.items() if k != id_key}
            batch_rows.append(NodeBatchRow(id=props[id_key], props=row_props))

        if not batch_rows:
            return 0, skipped

        build_query = (
            build_merge_node_query if self._use_merge else build_create_node_query
        )
        query = build_query(label, id_key)
        target_conn = conn or self._connection
        target_conn.execute_batch(query, batch_rows)
        return len(batch_rows), skipped


class RelationshipBatchFlusher(BaseBatchFlusher):
    """Handles buffering and flushing of relationship batches to Memgraph."""

    __slots__ = ("_rel_count", "_rel_groups")

    def __init__(
        self,
        connection: MemgraphConnection,
        *,
        use_merge: bool,
        batch_size: int,
    ) -> None:
        super().__init__(connection, use_merge=use_merge, batch_size=batch_size)
        self._rel_count = 0
        self._rel_groups: defaultdict[
            tuple[str, str, str, str, str], list[RelBatchRow]
        ] = defaultdict(list)

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, PropertyValue],
        rel_type: str,
        to_spec: tuple[str, str, PropertyValue],
        properties: dict[str, PropertyValue] | None = None,
    ) -> None:
        from_label, from_key, from_val = from_spec
        to_label, to_key, to_val = to_spec
        pattern = (from_label, from_key, rel_type, to_label, to_key)
        self._rel_groups[pattern].append(
            RelBatchRow(from_val=from_val, to_val=to_val, props=properties or {})
        )
        self._rel_count += 1
        if self._rel_count >= self._batch_size:
            logger.debug(ls.MG_REL_BUFFER_FLUSH, size=self._batch_size)
            self.flush()

    def flush(self) -> None:
        if not self._rel_count:
            return

        total_attempted = 0
        total_successful = 0
        first_error = None

        if self._executor and len(self._rel_groups) > 1:
            total_attempted, total_successful, first_error = self._run_parallel_flush(
                groups=dict(self._rel_groups),
                flush_fn=self._flush_rel_pattern_group,
                log_label=ls.MG_PARALLEL_FLUSH_RELS,
                error_msg_fmt=ls.MG_REL_FLUSH_ERROR.replace("{pattern}", "{key}"), # Adapter
                workers=settings.FLUSH_THREAD_POOL_SIZE,
            )
        else:
            for pattern, params_list in self._rel_groups.items():
                try:
                    attempted, successful = self._flush_rel_pattern_group(pattern, params_list)
                    total_attempted += attempted
                    total_successful += successful
                except Exception as e:  # noqa: BLE001
                    logger.error(ls.MG_REL_FLUSH_ERROR.format(pattern=pattern, error=e))
                    if first_error is None:
                        first_error = e

        logger.info(
            ls.MG_RELS_FLUSHED.format(
                total=self._rel_count,
                success=total_successful,
                failed=total_attempted - total_successful,
            )
        )
        self._rel_count = 0
        self._rel_groups.clear()

        if first_error is not None:
            raise first_error


    def _flush_rel_pattern_group(
        self,
        pattern: tuple[str, str, str, str, str],
        params_list: list[RelBatchRow],
        conn: MemgraphConnection | None = None,
    ) -> tuple[int, int]:
        from_label, from_key, rel_type, to_label, to_key = pattern
        build_rel_query = (
            build_merge_relationship_query
            if self._use_merge
            else build_create_relationship_query
        )
        has_props = any(p[KEY_PROPS] for p in params_list)
        query = build_rel_query(
            from_label,
            from_key,
            rel_type,
            to_label,
            to_key,
            has_props,
        )

        target_conn = conn or self._connection
        results = target_conn.execute_batch_with_return(query, params_list)
        batch_successful = 0
        for r in results:
            created = r.get(KEY_CREATED, 0)
            if isinstance(created, int):
                batch_successful += created

        if rel_type == REL_TYPE_CALLS:
            failed = len(params_list) - batch_successful
            if failed > 0:
                logger.warning(ls.MG_CALLS_FAILED.format(count=failed))
                for i, sample in enumerate(params_list[:3]):
                    logger.warning(
                        ls.MG_CALLS_SAMPLE.format(
                            index=i + 1,
                            from_label=from_label,
                            from_val=sample[KEY_FROM_VAL],
                            to_label=to_label,
                            to_val=sample[KEY_TO_VAL],
                        )
                    )

        return len(params_list), batch_successful


class GraphExporter:
    """Converts Memgraph data into the serializable GraphData format."""

    @staticmethod
    def export(fetch_all) -> GraphData:
        logger.info(ls.MG_EXPORTING)

        nodes_data = fetch_all(CYPHER_EXPORT_NODES)
        relationships_data = fetch_all(CYPHER_EXPORT_RELATIONSHIPS)

        metadata = GraphMetadata(
            total_nodes=len(nodes_data),
            total_relationships=len(relationships_data),
            exported_at=datetime.now(UTC).isoformat(),
        )

        logger.info(
            ls.MG_EXPORTED.format(nodes=len(nodes_data), rels=len(relationships_data))
        )
        return GraphData(
            nodes=nodes_data,
            relationships=relationships_data,
            metadata=metadata,
        )


class MemgraphIngestor:
    """Facade coordinating connection, batching, and graph export."""

    __slots__ = (
        "_connection",
        "_use_merge",
        "batch_size",
        "_node_flusher",
        "_rel_flusher",
    )

    def __init__(
        self,
        host: str,
        port: int,
        batch_size: int = 1000,
        username: str | None = None,
        password: str | None = None,
        use_merge: bool = True,
    ):
        username_clean = username.strip() if username and username.strip() else None
        password_clean = password.strip() if password and password.strip() else None
        if (username_clean is None) != (password_clean is None):
            raise ValueError(ex.AUTH_INCOMPLETE)
        if batch_size < 1:
            raise ValueError(ex.BATCH_SIZE)

        self.batch_size = batch_size
        self._use_merge = use_merge
        self._connection = MemgraphConnection(
            host=host,
            port=port,
            username=username_clean,
            password=password_clean,
        )
        self._node_flusher = NodeBatchFlusher(
            connection=self._connection,
            use_merge=self._use_merge,
            batch_size=self.batch_size,
        )
        self._rel_flusher = RelationshipBatchFlusher(
            connection=self._connection,
            use_merge=self._use_merge,
            batch_size=self.batch_size,
        )

    def __enter__(self) -> MemgraphIngestor:
        self._connection.open()
        self._node_flusher.start()
        self._rel_flusher.start()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        try:
            if exc_type:
                logger.exception(ls.MG_EXCEPTION.format(error=exc_val))
                try:
                    self.flush_all()
                except Exception as flush_err:  # noqa: BLE001
                    logger.error(ls.MG_FLUSH_ERROR.format(error=flush_err))
            else:
                self.flush_all()
        finally:
            self._node_flusher.stop()
            self._rel_flusher.stop()
            self._connection.close()

    async def __aenter__(self) -> MemgraphIngestor:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    # Connection / query façade

    def clean_database(self) -> None:
        logger.info(ls.MG_CLEANING_DB)
        self._connection.execute(CYPHER_DELETE_ALL)
        logger.info(ls.MG_DB_CLEANED)

    def list_projects(self) -> list[str]:
        result = self.fetch_all(CYPHER_LIST_PROJECTS)
        return [str(r[KEY_NAME]) for r in result]

    def delete_project(self, project_name: str) -> None:
        logger.info(ls.MG_DELETING_PROJECT.format(project_name=project_name))
        self._connection.execute(
            CYPHER_DELETE_PROJECT, {KEY_PROJECT_NAME: project_name}
        )
        logger.info(ls.MG_PROJECT_DELETED.format(project_name=project_name))

    def ensure_constraints(self) -> None:
        logger.info(ls.MG_ENSURING_CONSTRAINTS)
        for label, prop in NODE_UNIQUE_CONSTRAINTS.items():
            try:
                self._connection.execute(build_constraint_query(label, prop))
            except Exception as e:  # noqa: BLE001
                # Constraints are best-effort: log but do not fail startup.
                logger.warning(
                    ls.MG_CONSTRAINTS_FAILED.format(label=label, prop=prop, error=e)
                    if hasattr(ls, "MG_CONSTRAINTS_FAILED")
                    else f"Failed to create constraint on {label}({prop}): {e}"
                )
        logger.info(ls.MG_CONSTRAINTS_DONE)
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        logger.info(ls.MG_ENSURING_INDEXES)
        for label, prop in NODE_UNIQUE_CONSTRAINTS.items():
            try:
                self._connection.execute(build_index_query(label, prop))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    ls.MG_INDEX_FAILED.format(label=label, prop=prop, error=e)
                    if hasattr(ls, "MG_INDEX_FAILED")
                    else f"Failed to create index on {label}({prop}): {e}"
                )
        logger.info(ls.MG_INDEXES_DONE)

    # Batching façade

    def ensure_node_batch(
        self,
        label: str,
        properties: dict[str, PropertyValue],
    ) -> None:
        self._node_flusher.ensure_node_batch(label, properties)

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, PropertyValue],
        rel_type: str,
        to_spec: tuple[str, str, PropertyValue],
        properties: dict[str, PropertyValue] | None = None,
    ) -> None:
        self._rel_flusher.ensure_relationship_batch(
            from_spec,
            rel_type,
            to_spec,
            properties,
        )

    def flush_nodes(self) -> None:
        self._node_flusher.flush()

    def flush_relationships(self) -> None:
        self._rel_flusher.flush()

    def flush_all(self) -> None:
        logger.info(ls.MG_FLUSH_START)
        self.flush_nodes()
        self.flush_relationships()
        logger.info(ls.MG_FLUSH_COMPLETE)

    # Query façade used by other services

    def fetch_all(
        self,
        query: str,
        params: dict[str, PropertyValue] | None = None,
    ) -> list[ResultRow]:
        logger.debug(ls.MG_FETCH_QUERY, query=query, params=params)
        return self._connection.execute(query, params)

    def execute_write(
        self,
        query: str,
        params: dict[str, PropertyValue] | None = None,
    ) -> None:
        logger.debug(ls.MG_WRITE_QUERY, query=query, params=params)
        self._connection.execute(query, params)

    # Graph export façade

    def export_graph_to_dict(self) -> GraphData:
        return GraphExporter.export(self.fetch_all)
