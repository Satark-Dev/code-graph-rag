from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from ..config import settings
from ..services.chat_orchestrator import (
    ChatOrchestratorService,
    evidence_to_markdown,
    remediation_to_markdown,
    scoring_to_markdown,
)
from ..services.findings_client import fetch_org_tool_finding
from ..services.scoring_service import extract_scored_findings, normalize_scoring_output


def _aggregate_token_usage(usage_dict: dict[str, int], token_usage: Any) -> None:
    if isinstance(token_usage, dict):
        usage_dict["input_tokens"] += int(token_usage.get("input_tokens", 0) or 0)
        usage_dict["output_tokens"] += int(token_usage.get("output_tokens", 0) or 0)
        usage_dict["total_tokens"] += int(token_usage.get("total_tokens", 0) or 0)


def _aggregate_stage_results(
    per_response: dict[str, Any],
    stage_key: str,
    list_key: str,
    target_list: list[dict[str, Any]],
    usage_dict: dict[str, int],
) -> int:
    stage_data = per_response.get(stage_key)
    if not isinstance(stage_data, dict):
        return 0

    items = stage_data.get(list_key)
    if isinstance(items, list) and items:
        target_list.extend(items)

    _aggregate_token_usage(usage_dict, stage_data.get("token_usage"))

    timings = stage_data.get("timings_ms")
    return int(timings) if isinstance(timings, int) else 0


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
        # For API calls we still raise an HTTPException so the client sees a 502,
        # but for background Kafka workers we log and return a sentinel score so
        # the pipeline can continue without crashing the consumer.
        logger.warning(
            "Scoring output missing or ambiguous for org_tool_findings_id=%s; "
            "returning default score.",
            org_tool_findings_id,
        )
        # Default: zero score and no verdict text.
        return per_response, (org_tool_findings_id, 0.0, None)

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

    evidence_items: list[dict[str, Any]] = []
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

        evidence_ms += _aggregate_stage_results(
            per_response, "evidence", "items", evidence_items, evidence_usage
        )
        scoring_ms += _aggregate_stage_results(
            per_response, "scoring", "findings", scoring_findings, scoring_usage
        )
        remediation_ms += _aggregate_stage_results(
            per_response, "remediation", "findings", remediation_findings, remediation_usage
        )

        if models is None and isinstance(per_response.get("models"), dict):
            models = per_response["models"]

    response_data = {
        # run_id is assigned by API layer
        "evidence": {
            "items": evidence_items,
            "timings_ms": evidence_ms,
            "token_usage": evidence_usage,
            "markdown": evidence_to_markdown({"items": evidence_items}),
        },
        "scoring": {
            "findings": scoring_findings,
            "timings_ms": scoring_ms,
            "token_usage": scoring_usage,
            "markdown": scoring_to_markdown({"findings": scoring_findings}),
        },
        "remediation": {
            "findings": remediation_findings,
            "timings_ms": remediation_ms,
            "token_usage": remediation_usage,
            "markdown": remediation_to_markdown({"findings": remediation_findings}),
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
