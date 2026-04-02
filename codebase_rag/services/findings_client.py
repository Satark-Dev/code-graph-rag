from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException

from ..config import settings


def backend_api_base_url() -> str:
    return settings.BACKEND_API_BASE_URL.rstrip("/")


async def fetch_org_tool_finding(*, org_id: str, org_tool_findings_id: str) -> dict[str, Any]:
    base = backend_api_base_url()
    url = f"{base}/v1/org/{org_id}/ai-findings/{org_tool_findings_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502, detail=f"Failed to reach findings backend at {url}: {e}"
        ) from e

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                "Findings backend request failed for "
                f"org_tool_findings_id={org_tool_findings_id}: "
                f"HTTP {response.status_code}"
            ),
        )

    try:
        payload = response.json()
    except ValueError as e:
        raise HTTPException(
            status_code=502,
            detail=(
                "Findings backend returned invalid JSON for "
                f"org_tool_findings_id={org_tool_findings_id}"
            ),
        ) from e

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail=(
                "Findings backend payload must be a JSON object for "
                f"org_tool_findings_id={org_tool_findings_id}"
            ),
        )
    return payload
