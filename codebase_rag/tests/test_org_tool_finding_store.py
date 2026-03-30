from __future__ import annotations

import codebase_rag.utils.org_tool_finding_store as store


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.rowcount = 1

    def execute(self, sql: str, params: tuple) -> None:
        self.calls.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeConn:
    def __init__(self) -> None:
        self.cur = _FakeCursor()

    def cursor(self):
        return self.cur

    def close(self) -> None:
        return None


def test_persist_org_tool_finding_scores_updates_score_tag_version_and_explanation(
    monkeypatch,
) -> None:
    # Force code path to run without real DB.
    monkeypatch.setattr(store, "has_pgvector", lambda: True)
    fake_conn = _FakeConn()
    monkeypatch.setattr(store, "pg_connect", lambda autocommit=True: fake_conn)

    org_id = "3b393436-119f-45ca-8d53-842d7ec96771"
    finding_id = "00ffe5f0-8e13-4b39-b29b-512dd40baba0"
    updated = store.persist_org_tool_finding_scores(
        org_id=org_id,
        scored_findings=[(finding_id, 80.0, "clear explanation")],
    )
    assert updated == 1
    assert len(fake_conn.cur.calls) == 1

    sql, params = fake_conn.cur.calls[0]
    assert "COALESCE(score_version, 0) + 1" in sql
    assert "::tool_finding_tag_enum" in sql
    # Params include score (twice), threshold, tag strings, finding_id, org_id.
    assert params[0] == 80.0
    assert params[1] == "clear explanation"
    assert params[2] == 80.0
    assert params[3] == store.PRIORITY_SCORE_THRESHOLD
    assert params[4] == store.TAG_PRIORITY
    assert params[5] == store.TAG_OTHER
    assert params[6] == finding_id
    assert params[7] == org_id

