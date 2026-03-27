from __future__ import annotations

import pytest
from pydantic import ValidationError

from codebase_rag.api import IndexRequest


def test_index_request_requires_org_id() -> None:
    with pytest.raises(ValidationError):
        IndexRequest(repo_path="/tmp/repo", clean=True, exclude=None)


def test_index_request_rejects_blank_org_id() -> None:
    with pytest.raises(ValidationError):
        IndexRequest(repo_path="/tmp/repo", clean=True, exclude=None, org_id="   ")


def test_index_request_normalizes_org_id_whitespace() -> None:
    req = IndexRequest(repo_path="/tmp/repo", clean=False, exclude=None, org_id="  org-1  ")
    assert req.org_id == "org-1"
