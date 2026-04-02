from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from .dependencies import has_pgvector
from .pg import pg_connect

PRIORITY_SCORE_THRESHOLD = 75.0
TAG_PRIORITY = "priority"
TAG_OTHER = "other"


def persist_org_tool_finding_scores(
    *, org_id: str, scored_findings: Sequence[tuple[str, float, str | None]]
) -> int:
    """
    Persist score + tag + explanation + score_version into public.org_tool_findings.

    - score is always overwritten with the latest value.
    - tag rule:
        * score >= 75  -> 'priority'
        * score < 75   -> 'other'
    - score_version:
        * NULL or missing -> set to 1
        * otherwise       -> increment by 1
    """
    if not scored_findings:
        return 0

    if not has_pgvector():
        logger.warning(
            "Skipping org_tool_findings score persistence: Postgres dependencies missing."
        )
        return 0

    conn = pg_connect(autocommit=True)
    updated = 0
    try:
        with conn.cursor() as cur:
            for finding_id, score, explanation in scored_findings:
                cur.execute(
                    """
                    UPDATE public.org_tool_findings
                    SET
                        score = %s,
                        explanation = COALESCE(%s, explanation),
                        tag = CASE
                            WHEN %s >= %s THEN %s::tool_finding_tag_enum
                            ELSE %s::tool_finding_tag_enum
                        END,
                        score_version = COALESCE(score_version, 0) + 1,
                        updated_at = now()
                    WHERE id = %s::uuid
                      AND org_id = %s::uuid
                    """,
                    (
                        float(score),
                        explanation,
                        float(score),
                        PRIORITY_SCORE_THRESHOLD,
                        TAG_PRIORITY,
                        TAG_OTHER,
                        finding_id,
                        org_id,
                    ),
                )
                updated += cur.rowcount
    finally:
        conn.close()

    return updated


def get_branch_name_for_index_asset(asset_id: str, org_id: str) -> str | None:
    """
    Branch name for Kafka index jobs when the backend sends the **branch asset** row id.

    Expects ``org_tool_findings.id`` for a row with ``type = 'asset'``; returns that row's
    ``name`` (git branch). No parent / findings walk.
    """
    if not has_pgvector():
        return None

    try:
        conn = pg_connect(autocommit=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name
                FROM public.org_tool_findings
                WHERE id = %s::uuid
                  AND org_id = %s::uuid
                  AND type = 'asset'
                """,
                (asset_id, org_id),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            return str(row[0]).strip() or None
    except Exception as e:
        logger.warning(
            "Failed to fetch branch name for index asset {} (org={}): {}",
            asset_id,
            org_id,
            e,
        )
        return None
    finally:
        if "conn" in locals():
            conn.close()


def get_all_child_findings_for_branch_asset(asset_id: str, org_id: str) -> list[str]:
    """
    All ``type = 'findings'`` rows whose ``parent_id`` equals the branch asset's ``uid``.

    ``asset_id`` is the index job's org_tool_findings row (``type = 'asset'``).
    """
    if not has_pgvector():
        return []

    try:
        conn = pg_connect(autocommit=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uid
                FROM public.org_tool_findings
                WHERE id = %s::uuid
                  AND org_id = %s::uuid
                  AND type = 'asset'
                """,
                (asset_id, org_id),
            )
            row = cur.fetchone()
            if not row or row[0] is None:
                return []
            asset_uid = row[0]

            cur.execute(
                """
                SELECT id::text
                FROM public.org_tool_findings
                WHERE org_id = %s::uuid
                  AND type = 'findings'
                  AND parent_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (org_id, asset_uid),
            )
            rows = cur.fetchall() or []
            return [str(r[0]) for r in rows if r and r[0]]
    except Exception as e:
        logger.warning(
            "Failed to fetch child findings for branch asset {} (org {}): {}",
            asset_id,
            org_id,
            e,
        )
        return []
    finally:
        if "conn" in locals():
            conn.close()


def get_branch_name_for_finding(finding_id: str, org_id: str) -> str | None:
    """
    Retrieves the branch name from the 'name' column by:
    1. Finding the finding with type='findings' and getting its parent_id.
    2. Finding the asset with type='asset' where uid=parent_id and getting its 'name'.
    Used for **evidence / chat** paths where the id is a ``findings`` row.
    """
    if not has_pgvector():
        return None

    try:
        conn = pg_connect(autocommit=True)
        with conn.cursor() as cur:
            # 1. Get branch asset ID (parent_id) from the finding
            cur.execute(
                """
                SELECT parent_id
                FROM public.org_tool_findings
                WHERE id = %s::uuid
                  AND org_id = %s::uuid
                  AND type = 'findings'
                """,
                (finding_id, org_id),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            branch_asset_uid = row[0]

            # 2. Get the branch name from the asset
            cur.execute(
                """
                SELECT name
                FROM public.org_tool_findings
                WHERE org_id = %s::uuid
                  AND type = 'asset'
                  AND (uid = %s OR id::text = %s)
                """,
                (org_id, branch_asset_uid, str(branch_asset_uid)),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            return str(row[0])

    except Exception as e:
        logger.warning(
            "Failed to fetch branch name for finding {} (org={}): {}",
            finding_id,
            org_id,
            e,
        )
        return None
    finally:
        if "conn" in locals():
            conn.close()


def get_all_child_findings_for_branch(finding_id: str, org_id: str) -> list[str]:
    """
    Given a single finding id, return ALL finding ids for the same branch asset:

    1. Look up that finding (type='findings') to get its parent_id = branch_asset_uid.
    2. Find all rows with type='findings' and parent_id = branch_asset_uid.
    """
    if not has_pgvector():
        return [finding_id]

    try:
        conn = pg_connect(autocommit=True)
        with conn.cursor() as cur:
            # 1. Get branch asset UID from the seed finding
            cur.execute(
                """
                SELECT parent_id
                FROM public.org_tool_findings
                WHERE id = %s::uuid
                  AND org_id = %s::uuid
                  AND type = 'findings'
                """,
                (finding_id, org_id),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return [finding_id]
            branch_asset_uid = row[0]

            # 2. Get all child findings that share this branch_asset_uid as parent_id
            cur.execute(
                """
                SELECT id::text
                FROM public.org_tool_findings
                WHERE org_id = %s::uuid
                  AND type = 'findings'
                  AND parent_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (org_id, branch_asset_uid),
            )
            ids = [str(r[0]) for r in (cur.fetchall() or []) if r and r[0]]
            return ids or [finding_id]
    except Exception as e:
        logger.warning(
            "Failed to fetch child findings for branch (seed finding {}, org {}): {}",
            finding_id,
            org_id,
            e,
        )
        return [finding_id]
    finally:
        if "conn" in locals():
            conn.close()


def get_chat_finding_metadata(finding_id: str, org_id: str) -> tuple[str | None, str | None]:
    """
    Retrieves (repo_url, branch_name) from the database by:
    1. Finding the finding (type='findings') -> get parent_id (Branch Asset).
    2. Finding the Branch Asset (type='asset', uid=parent_id) -> get its 'name' (branch) and parent_id (Repo Asset).
    3. Finding the Repo Asset (type='asset', uid=parent_id_2) -> get its 'metadata' (clone URL).

    Kafka chat jobs do not use this: they resolve the local clone via ``get_branch_name_for_finding``
    and ``RepoManager.require_existing_local_clone`` only (no remote URL).
    """
    if not has_pgvector():
        return None, None

    try:
        conn = pg_connect(autocommit=True)
        with conn.cursor() as cur:
            # 1. Get branch asset ID (parent_id) from the finding
            cur.execute(
                """
                SELECT parent_id
                FROM public.org_tool_findings
                WHERE id = %s::uuid
                  AND org_id = %s::uuid
                  AND type = 'findings'
                """,
                (finding_id, org_id),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None, None
            branch_asset_uid = row[0]

            # 2. Get branch name and repo asset ID from the branch asset
            cur.execute(
                """
                SELECT name, parent_id
                FROM public.org_tool_findings
                WHERE org_id = %s::uuid
                  AND type = 'asset'
                  AND (uid = %s OR id::text = %s)
                """,
                (org_id, branch_asset_uid, str(branch_asset_uid)),
            )
            row = cur.fetchone()
            if not row:
                return None, None
            branch_name = str(row[0]) if row[0] else None
            repo_asset_uid = row[1]

            if not repo_asset_uid:
                return None, branch_name

            # 3. Get the repo URL from the repository asset metadata
            cur.execute(
                """
                SELECT metadata
                FROM public.org_tool_findings
                WHERE org_id = %s::uuid
                  AND type = 'asset'
                  AND (uid = %s OR id::text = %s)
                """,
                (org_id, repo_asset_uid, str(repo_asset_uid)),
            )
            row = cur.fetchone()
            repo_url = None
            if row and row[0]:
                metadata = row[0]
                if isinstance(metadata, dict):
                    repo_url = metadata.get("url")

            return repo_url, branch_name
    except Exception as e:
        logger.warning(
            "Failed to fetch metadata for chat job (finding {}, org {}): {}",
            finding_id,
            org_id,
            e,
        )
        return None, None
    finally:
        if "conn" in locals():
            conn.close()
