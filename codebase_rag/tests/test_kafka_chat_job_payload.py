from __future__ import annotations

from codebase_rag.services.kafka.chat_job_payload import ChatJobPayloadV1


def test_chat_job_payload_snake_case() -> None:
    p = ChatJobPayloadV1.model_validate(
        {
            "org_id": "org-1",
            "org_tool_findings_ids": ["a", "b"],
            "invocation_id": "inv-1",
        }
    )
    assert p.org_id == "org-1"
    assert p.org_tool_findings_ids == ["a", "b"]
    assert p.invocation_id == "inv-1"


def test_chat_job_payload_camel_case_aliases() -> None:
    p = ChatJobPayloadV1.model_validate(
        {
            "orgId": "org-2",
            "orgToolFindingsIds": ["x"],
            "invocationId": "z",
        }
    )
    assert p.org_id == "org-2"
    assert p.org_tool_findings_ids == ["x"]
    assert p.invocation_id == "z"
