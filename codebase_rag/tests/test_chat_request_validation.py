from __future__ import annotations

import pytest
from pydantic import ValidationError

from codebase_rag.api import ChatRequest


def test_chat_request_requires_org_id() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(org_tool_findings_ids=["finding-1"])


def test_chat_request_rejects_blank_org_id() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(org_tool_findings_ids=["finding-1"], org_id="   ")


def test_chat_request_rejects_empty_org_tool_findings_ids() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(org_tool_findings_ids=[], org_id="org-1")


def test_chat_request_rejects_user_id_field() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            org_tool_findings_ids=["finding-1"], org_id="org-1", user_id="ignored"
        )


def test_chat_request_normalizes_org_id_and_finding_ids_whitespace() -> None:
    req = ChatRequest(
        org_tool_findings_ids=["  finding-1  ", "finding-2"],
        org_id="  org-1  ",
    )
    assert req.org_id == "org-1"
    assert req.org_tool_findings_ids == ["finding-1", "finding-2"]


