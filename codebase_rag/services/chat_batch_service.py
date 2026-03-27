from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from ..config import settings
from ..services.chat_orchestrator import ChatOrchestratorService
from ..services.findings_client import fetch_org_tool_finding
from ..services.scoring_service import extract_scored_findings, normalize_scoring_output


async def _process_one_finding(
    *,
    org_id: str,
    org_tool_findings_id: str,
    ingestor: Any,
    target_repo_path: str,
) -> tuple[dict[str, Any], tuple[str, float, str | None]]:
    finding_payload = await fetch_org_tool_finding(
        org_id=org_id, org_tool_findings_id=org_tool_findings_id
    )
    per_request_query = {"findings": [finding_payload]}
    per_response = await ChatOrchestratorService.process_query(
        request_query=per_request_query,
        ingestor=ingestor,
        target_repo_path=target_repo_path,
    )
    normalize_scoring_output(per_response)

    scored = extract_scored_findings(
        org_tool_findings_ids=[org_tool_findings_id],
        response_data=per_response,
    )
    if len(scored) != 1:
        raise HTTPException(
            status_code=502,
            detail=f"Scoring output missing for org_tool_findings_id={org_tool_findings_id}",
        )
    return per_response, scored[0]


async def process_chat_for_findings_ids(
    *,
    org_id: str,
    org_tool_findings_ids: list[str],
    ingestor: Any,
    target_repo_path: str,
) -> tuple[dict[str, Any], list[tuple[str, float, str | None]]]:
    """
    Run independent chat pipeline for each finding id (in parallel),
    then aggregate into a single /api/chat response payload.
    """
    tasks = [
        _process_one_finding(
            org_id=org_id,
            org_tool_findings_id=finding_id,
            ingestor=ingestor,
            target_repo_path=target_repo_path,
        )
        for finding_id in org_tool_findings_ids
    ]
    results = await asyncio.gather(*tasks)

    evidence_findings: list[dict[str, Any]] = []
    scoring_findings: list[dict[str, Any]] = []
    remediation_findings: list[dict[str, Any]] = []
    evidence_ms = 0
    scoring_ms = 0
    remediation_ms = 0
    evidence_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    scoring_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    remediation_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    models: dict[str, Any] | None = None

    scored_findings: list[tuple[str, float, str | None]] = []

    for per_response, scored in results:
        scored_findings.append(scored)

        if isinstance(per_response.get("evidence"), dict):
            ev = per_response["evidence"]
            if isinstance(ev.get("findings"), list) and ev["findings"]:
                evidence_findings.append(ev["findings"][0])
            if isinstance(ev.get("timings_ms"), int):
                evidence_ms += int(ev["timings_ms"])
            tu = ev.get("token_usage")
            if isinstance(tu, dict):
                evidence_usage["input_tokens"] += int(tu.get("input_tokens", 0) or 0)
                evidence_usage["output_tokens"] += int(tu.get("output_tokens", 0) or 0)
                evidence_usage["total_tokens"] += int(tu.get("total_tokens", 0) or 0)

        if isinstance(per_response.get("scoring"), dict):
            sc = per_response["scoring"]
            if isinstance(sc.get("findings"), list) and sc["findings"]:
                scoring_findings.append(sc["findings"][0])
            if isinstance(sc.get("timings_ms"), int):
                scoring_ms += int(sc["timings_ms"])
            tu = sc.get("token_usage")
            if isinstance(tu, dict):
                scoring_usage["input_tokens"] += int(tu.get("input_tokens", 0) or 0)
                scoring_usage["output_tokens"] += int(tu.get("output_tokens", 0) or 0)
                scoring_usage["total_tokens"] += int(tu.get("total_tokens", 0) or 0)

        if isinstance(per_response.get("remediation"), dict):
            re = per_response["remediation"]
            if isinstance(re.get("findings"), list) and re["findings"]:
                remediation_findings.append(re["findings"][0])
            if isinstance(re.get("timings_ms"), int):
                remediation_ms += int(re["timings_ms"])
            tu = re.get("token_usage")
            if isinstance(tu, dict):
                remediation_usage["input_tokens"] += int(tu.get("input_tokens", 0) or 0)
                remediation_usage["output_tokens"] += int(tu.get("output_tokens", 0) or 0)
                remediation_usage["total_tokens"] += int(tu.get("total_tokens", 0) or 0)

        if models is None and isinstance(per_response.get("models"), dict):
            models = per_response["models"]

    response_data = {
        # run_id is assigned by API layer
        "evidence": {
            "findings": evidence_findings,
            "timings_ms": evidence_ms,
            "token_usage": evidence_usage,
        },
        "scoring": {
            "findings": scoring_findings,
            "timings_ms": scoring_ms,
            "token_usage": scoring_usage,
        },
        "remediation": {
            "findings": remediation_findings,
            "timings_ms": remediation_ms,
            "token_usage": remediation_usage,
        },
        "models": models
        or {
            "orchestrator": {
                "provider": settings.active_orchestrator_config.provider,
                "model": settings.active_orchestrator_config.model_id,
            },
            "cypher": {
                "provider": settings.active_cypher_config.provider,
                "model": settings.active_cypher_config.model_id,
            },
        },
    }

    return response_data, scored_findings

