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


def get_branch_name_for_finding(finding_id: str, org_id: str) -> str | None:
    """
    Retrieves the branch name from the 'name' column by:
    1. Finding the finding with type='findings' and getting its parent_id.
    2. Finding the asset with type='asset' where uid=parent_id and getting its 'name'.
    Used primarily for indexing where the repo URL is already known.
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
                WHERE id = %s::uuid AND org_id = %s::uuid AND type = 'findings'
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
                WHERE uid = %s AND org_id = %s::uuid AND type = 'asset'
                """,
                (branch_asset_uid, org_id),
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


def get_chat_finding_metadata(finding_id: str, org_id: str) -> tuple[str | None, str | None]:
    """
    Retrieves (repo_url, branch_name) from the database by:
    1. Finding the finding (type='findings') -> get parent_id (Branch Asset).
    2. Finding the Branch Asset (type='asset', uid=parent_id) -> get its 'name' (branch) and parent_id (Repo Asset).
    3. Finding the Repo Asset (type='asset', uid=parent_id_2) -> get its 'metadata' (clone URL).
    Used for chat where the repo URL must be resolved internally.
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
                WHERE id = %s::uuid AND org_id = %s::uuid AND type = 'findings'
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
                WHERE uid = %s AND org_id = %s::uuid AND type = 'asset'
                """,
                (branch_asset_uid, org_id),
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
                WHERE uid = %s AND org_id = %s::uuid AND type = 'asset'
                """,
                (repo_asset_uid, org_id),
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
