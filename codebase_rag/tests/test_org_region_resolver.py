"""Tests for org → region → org DB DSN resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.utils.org_region_resolver import OrgRegionResolver


@pytest.fixture
def mock_settings_core_only():
    return SimpleNamespace(
        CORE_DB_HOST="core.example",
        CORE_DB_PORT=5432,
        CORE_DB_USER="u",
        CORE_DB_PASSWORD="p@ss word",
        CORE_DB_NAME="coredb",
        CORE_DB_SSL="false",
        ORG_DB_HOST_1=None,
        ORG_DB_PORT_1=None,
        ORG_DB_USER_1=None,
        ORG_DB_PASSWORD_1=None,
        ORG_DB_NAME_1=None,
        ORG_DB_SSL_1=None,
        ORG_DB_HOST_2=None,
        ORG_DB_PORT_2=None,
        ORG_DB_USER_2=None,
        ORG_DB_PASSWORD_2=None,
        ORG_DB_NAME_2=None,
        ORG_DB_SSL_2=None,
    )


def test_core_dsn_from_fields(mock_settings_core_only, monkeypatch):
    monkeypatch.setattr("codebase_rag.config.settings", mock_settings_core_only)

    r = OrgRegionResolver()
    dsn = r._get_core_db_dsn()
    assert "core.example" in dsn
    assert "5432" in dsn
    assert "coredb" in dsn
    assert "p%40ss+word" in dsn


def test_org_shard_missing_raises(monkeypatch):
    mock = SimpleNamespace(ORG_DB_HOST_1=None)
    monkeypatch.setattr("codebase_rag.config.settings", mock)

    r = OrgRegionResolver()
    with pytest.raises(RuntimeError, match="Org DB region 1 is not configured"):
        r.get_org_db_connection_string(1)


def test_org_dsn_from_explicit_fields(mock_settings_core_only, monkeypatch):
    merged = {**vars(mock_settings_core_only)}
    merged.update(
        {
            "ORG_DB_HOST_1": "org1.example",
            "ORG_DB_PORT_1": 5433,
            "ORG_DB_USER_1": "ou",
            "ORG_DB_PASSWORD_1": "x",
            "ORG_DB_NAME_1": "orgdb1",
            "ORG_DB_SSL_1": "true",
        }
    )
    monkeypatch.setattr("codebase_rag.config.settings", SimpleNamespace(**merged))

    r = OrgRegionResolver()
    dsn = r.get_org_db_connection_string(1)
    assert "org1.example" in dsn
    assert "5433" in dsn
    assert "orgdb1" in dsn
    assert "sslmode=require" in dsn


def test_get_region_queries_core_and_caches(monkeypatch):
    mock_settings = SimpleNamespace(
        CORE_DB_HOST="localhost",
        CORE_DB_PORT=5432,
        CORE_DB_USER="c",
        CORE_DB_PASSWORD="pass",
        CORE_DB_NAME="core",
        CORE_DB_SSL="false",
        DB_CONNECT_TIMEOUT=5,
    )
    monkeypatch.setattr("codebase_rag.config.settings", mock_settings)

    cur = MagicMock()
    cur.fetchone.return_value = (2,)
    cursor_mgr = MagicMock()
    cursor_mgr.__enter__.return_value = cur
    cursor_mgr.__exit__.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cursor_mgr

    with patch("psycopg.connect", return_value=conn) as pconnect:
        r = OrgRegionResolver()
        assert r.get_region_for_org_id("abc") == 2
        assert r.get_region_for_org_id("abc") == 2
        assert pconnect.call_count == 1


def test_get_region_defaults_when_missing_in_user_org(monkeypatch):
    mock_settings = SimpleNamespace(
        CORE_DB_HOST="localhost",
        CORE_DB_PORT=5432,
        CORE_DB_USER="c",
        CORE_DB_PASSWORD="pass",
        CORE_DB_NAME="core",
        CORE_DB_SSL="false",
        DB_CONNECT_TIMEOUT=5,
    )
    monkeypatch.setattr("codebase_rag.config.settings", mock_settings)

    cur = MagicMock()
    cur.fetchone.return_value = None
    cursor_mgr = MagicMock()
    cursor_mgr.__enter__.return_value = cur
    cursor_mgr.__exit__.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cursor_mgr

    with patch("psycopg.connect", return_value=conn):
        r = OrgRegionResolver()
        assert r.get_region_for_org_id("unknown-org") == 1
