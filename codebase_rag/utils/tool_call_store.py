from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from ..utils.dependencies import has_pgvector
from ..utils.pg import PgInitKey, ensure_pg_initialized, pg_connect, pg_init_key

_TABLE_NAME = "cgr_tool_call_logs"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


if has_pgvector():
    from psycopg.types.json import Jsonb

    def _get_conn():
        return pg_connect(autocommit=True)

    def _migrate_invocation_id_column(conn) -> None:
        """
        Forward-only migration:
        - Add invocation_id column if missing
        - Backfill invocation_id from legacy run_id
        - Add index on invocation_id
        - Keep legacy run_id column for backwards compatibility
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                  AND column_name = 'invocation_id'
                """,
                (_TABLE_NAME,),
            )
            has_invocation_id = cur.fetchone() is not None
            if not has_invocation_id:
                cur.execute(f'ALTER TABLE "{_TABLE_NAME}" ADD COLUMN invocation_id text')
                cur.execute(
                    f'UPDATE "{_TABLE_NAME}" SET invocation_id = run_id WHERE invocation_id IS NULL'
                )
            # Ensure index exists (safe even if column pre-existed)
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{_TABLE_NAME}_invocation_id" ON "{_TABLE_NAME}" (invocation_id)'
            )

    def _migrate_stage_status_columns(conn) -> None:
        """Add stage_status/error_message columns if missing; backfill to success."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                  AND column_name IN ('stage_status', 'error_message')
                """,
                (_TABLE_NAME,),
            )
            existing = {r[0] for r in (cur.fetchall() or [])}
            if "stage_status" not in existing:
                # Cannot use a parameter for DEFAULT in some Postgres versions;
                # inline the literal instead.
                cur.execute(
                    f'ALTER TABLE "{_TABLE_NAME}" ADD COLUMN stage_status text NOT NULL DEFAULT \'success\''
                )
            if "error_message" not in existing:
                cur.execute(f'ALTER TABLE "{_TABLE_NAME}" ADD COLUMN error_message text')
            # Backfill any NULL/empty statuses to success (defensive)
            cur.execute(
                f'UPDATE "{_TABLE_NAME}" SET stage_status = %s WHERE stage_status IS NULL OR stage_status = %s',
                ("success", ""),
            )
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{_TABLE_NAME}_invocation_stage_status" ON "{_TABLE_NAME}" (invocation_id, stage, stage_status)'
            )

    def _init_db() -> None:
        conn = _get_conn()
        try:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{_TABLE_NAME}" (
                    tool_call_id bigserial PRIMARY KEY,
                    run_id text NOT NULL,
                    invocation_id text,
                    cache_key text NOT NULL,
                    repo_path text NOT NULL,
                    repo_state_hash text NOT NULL,
                    stage text NOT NULL,
                    stage_status text NOT NULL DEFAULT 'success',
                    error_message text,
                    input_json jsonb NOT NULL,
                    output_json jsonb NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{_TABLE_NAME}_run_id" ON "{_TABLE_NAME}" (run_id)'
            )
            _migrate_invocation_id_column(conn)
            _migrate_stage_status_columns(conn)
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{_TABLE_NAME}_cache_key" ON "{_TABLE_NAME}" (cache_key)'
            )
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{_TABLE_NAME}_stage" ON "{_TABLE_NAME}" (stage)'
            )
        finally:
            conn.close()

    def _ensure_db() -> None:
        ensure_pg_initialized(pg_init_key(PgInitKey.TOOL_CALL_STORE), _init_db)

    def new_run_id() -> str:
        return str(uuid.uuid4())

    def store_tool_call(
        *,
        run_id: str,
        cache_key: str,
        repo_path: str,
        repo_state_hash: str,
        stage: str,
        stage_status: str = "success",
        error_message: str | None = None,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
    ) -> None:
        _ensure_db()
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO "{_TABLE_NAME}"
                        (run_id, invocation_id, cache_key, repo_path, repo_state_hash, stage, stage_status, error_message, input_json, output_json)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        run_id,  # invocation_id (canonical); run_id is kept as legacy alias
                        cache_key,
                        repo_path,
                        repo_state_hash,
                        stage,
                        stage_status,
                        error_message,
                        Jsonb(tool_input),
                        Jsonb(tool_output),
                    ),
                )
        finally:
            conn.close()

    def fetch_latest_stage_status(*, run_id: str, stage: str) -> str | None:
        """Return latest stage_status for (invocation_id, stage), or None."""
        _ensure_db()
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT stage_status
                    FROM "{_TABLE_NAME}"
                    WHERE invocation_id = %s
                      AND stage = %s
                    ORDER BY tool_call_id DESC
                    LIMIT 1
                    """,
                    (run_id, stage),
                )
                row = cur.fetchone()
                return str(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()

    def fetch_latest_stage_output(*, run_id: str, stage: str) -> dict[str, Any] | None:
        """
        Return the most recent persisted output_json for a given (invocation_id, stage),
        or None if missing.
        """
        _ensure_db()
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT output_json
                    FROM "{_TABLE_NAME}"
                    WHERE invocation_id = %s
                      AND stage = %s
                    ORDER BY tool_call_id DESC
                    LIMIT 1
                    """,
                    (run_id, stage),
                )
                row = cur.fetchone()
                if not row or row[0] is None:
                    return None
                return row[0]
        finally:
            conn.close()

else:

    def new_run_id() -> str:
        return f"no-pgvector-{_now_iso()}"

    def store_tool_call(
        *,
        run_id: str,
        cache_key: str,
        repo_path: str,
        repo_state_hash: str,
        stage: str,
        stage_status: str = "success",
        error_message: str | None = None,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
    ) -> None:
        logger.debug(
            "Skipping tool-call persistence (pgvector extras not installed). "
            f"stage={stage} run_id={run_id} cache_key={cache_key}"
        )

    def fetch_latest_stage_status(*, run_id: str, stage: str) -> str | None:
        logger.warning(
            "Cannot fetch tool-call stage status without Postgres extras. "
            "Install pgvector dependencies to run stage-separated Kafka consumers. "
            "run_id={} stage={}",
            run_id,
            stage,
        )
        return None

    def fetch_latest_stage_output(*, run_id: str, stage: str) -> dict[str, Any] | None:
        logger.warning(
            "Cannot fetch tool-call stage output without Postgres extras. "
            "Install pgvector dependencies to run stage-separated Kafka consumers. "
            "run_id={} stage={}",
            run_id,
            stage,
        )
        return None

