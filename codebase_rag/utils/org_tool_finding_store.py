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
