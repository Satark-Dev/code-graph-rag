from __future__ import annotations

import codebase_rag.utils.org_tool_finding_store as store


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.rowcount = 1
        self._fetch_queue: list[tuple | None] = []

    def execute(self, sql: str, params: tuple) -> None:
        self.calls.append((sql, params))

    def fetchone(self):
        if self._fetch_queue:
            return self._fetch_queue.pop(0)
        return None

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


def test_get_branch_name_for_finding_accepts_finding_type_and_asset_id_fallback(monkeypatch) -> None:
    monkeypatch.setattr(store, "has_pgvector", lambda: True)
    fake_conn = _FakeConn()
    monkeypatch.setattr(store, "pg_connect", lambda autocommit=True: fake_conn)

    # 1st fetchone(): parent_id from the finding
    # 2nd fetchone(): name from the asset
    fake_conn.cur._fetch_queue = [("branch_asset_uuid",), ("main",)]

    org_id = "3b393436-119f-45ca-8d53-842d7ec96771"
    finding_id = "00ffe5f0-8e13-4b39-b29b-512dd40baba0"
    branch = store.get_branch_name_for_finding(finding_id, org_id)
    assert branch == "main"

    assert len(fake_conn.cur.calls) == 2
    sql1, params1 = fake_conn.cur.calls[0]
    assert "type = 'findings'" in sql1
    assert params1 == (finding_id, org_id)

    sql2, params2 = fake_conn.cur.calls[1]
    assert "AND (uid = %s OR id::text = %s)" in sql2
    assert params2[0] == org_id
    assert params2[1] == "branch_asset_uuid"
    assert params2[2] == "branch_asset_uuid"


def test_get_branch_name_for_index_asset_reads_asset_name(monkeypatch) -> None:
    monkeypatch.setattr(store, "has_pgvector", lambda: True)
    fake_conn = _FakeConn()
    monkeypatch.setattr(store, "pg_connect", lambda autocommit=True: fake_conn)

    fake_conn.cur._fetch_queue = [("feature/foo",)]

    org_id = "3b393436-119f-45ca-8d53-842d7ec96771"
    asset_id = "bc5f5a3d-0c7a-4d5c-ad47-835094e35356"
    branch = store.get_branch_name_for_index_asset(asset_id, org_id)
    assert branch == "feature/foo"

    assert len(fake_conn.cur.calls) == 1
    sql, params = fake_conn.cur.calls[0]
    assert "type = 'asset'" in sql
    assert params == (asset_id, org_id)


def test_get_all_child_findings_for_branch_asset_uses_parent_id_keys(monkeypatch) -> None:
    monkeypatch.setattr(store, "has_pgvector", lambda: True)
    fake_conn = _FakeConn()
    monkeypatch.setattr(store, "pg_connect", lambda autocommit=True: fake_conn)

    asset_id = "bc5f5a3d-0c7a-4d5c-ad47-835094e35356"
    org_id = "3b393436-119f-45ca-8d53-842d7ec96771"
    uid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    class _CursorWithFetchall(_FakeCursor):
        def __init__(self) -> None:
            super().__init__()
            self._fetchall_queue: list[list] = []

        def fetchall(self):
            if self._fetchall_queue:
                return self._fetchall_queue.pop(0)
            return []

    cur = _CursorWithFetchall()
    fake_conn.cur = cur
    cur._fetch_queue = [(uid,)]
    cur._fetchall_queue = [[("f1",), ("f2",)]]

    ids = store.get_all_child_findings_for_branch_asset(asset_id, org_id)
    assert ids == ["f1", "f2"]

    assert len(cur.calls) == 2
    sql2, params2 = cur.calls[1]
    assert "AND parent_id = %s" in sql2
    assert params2 == (org_id, uid)

