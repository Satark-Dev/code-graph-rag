from __future__ import annotations

from typing import Any


def extract_scored_findings(
    *, org_tool_findings_ids: list[str], response_data: dict[str, Any]
) -> list[tuple[str, float, str | None]]:
    """Map scoring outputs back to org_tool_findings_ids by list index."""
    scoring_findings = (
        response_data.get("scoring", {}).get("findings", [])
        if isinstance(response_data, dict)
        else []
    )
    if not isinstance(scoring_findings, list):
        return []

    updates: list[tuple[str, float, str | None]] = []
    for idx, finding_id in enumerate(org_tool_findings_ids):
        update = _extract_updates_from_item(finding_id, scoring_findings[idx])
        if update:
            updates.append(update)
    return updates


def _extract_updates_from_item(finding_id: str, item: Any) -> tuple[str, float, str | None] | None:
    if not isinstance(item, dict):
        return None
    analysis = item.get("analysis", {})
    if not isinstance(analysis, dict):
        return None
        
    score_val = analysis.get("score")
    if not isinstance(score_val, (int, float)):
        return None
        
    explanation = analysis.get("explanation")
    return finding_id, float(score_val), explanation


def _get_top_scoring_factor(breakdown: dict[str, Any]) -> tuple[str | None, str | None]:
    top_reason = None
    top_name = None
    top_score = -1.0
    for name, item in breakdown.items():
        if not isinstance(item, dict):
            continue
        bs = item.get("score")
        reason = item.get("reason")
        if isinstance(bs, (int, float)) and isinstance(reason, str) and bs > top_score:
            top_score = float(bs)
            top_reason = reason.strip()
            top_name = str(name).replace("_", " ")
    return top_name, top_reason


def build_human_scoring_explanation(analysis: dict[str, Any]) -> str:
    """Generate a plain-language scoring explanation if model output is weak/missing."""
    verdict = analysis.get("verdict")
    score = analysis.get("score")
    breakdown = analysis.get("scoring_breakdown")
    parts: list[str] = []

    if isinstance(score, (int, float)):
        score_text = f"rated {int(score)}"
        if isinstance(verdict, str):
            score_text += f" ({verdict})"
        parts.append(f"This finding is {score_text} based on the available code evidence.")

    if isinstance(breakdown, dict):
        top_name, top_reason = _get_top_scoring_factor(breakdown)
        if top_reason and top_name:
            parts.append(f"The strongest factor is {top_name}: {top_reason}")

    if not parts:
        return "The score reflects the observed exploitability, reachability, and control evidence from the analyzed code."
    return " ".join(parts)


def normalize_scoring_output(response_data: dict[str, Any]) -> None:
    """
    Ensure scoring.analysis key order is:
    verdict, score, explanation, scoring_breakdown, ...
    """
    scoring = response_data.get("scoring")
    if not isinstance(scoring, dict):
        return
    findings = scoring.get("findings")
    if not isinstance(findings, list):
        return

    for item in findings:
        _normalize_single_finding_analysis(item)


def _normalize_single_finding_analysis(item: Any) -> None:
    if not isinstance(item, dict):
        return
    analysis = item.get("analysis")
    if not isinstance(analysis, dict):
        return

    explanation = analysis.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = build_human_scoring_explanation(analysis)
    else:
        explanation = explanation.strip()

    normalized: dict[str, Any] = {}
    for key in ["verdict", "score", "explanation", "scoring_breakdown"]:
        if key == "explanation":
            normalized[key] = explanation
        elif key in analysis:
            normalized[key] = analysis[key]

    for k, v in analysis.items():
        if k not in normalized:
            normalized[k] = v
    item["analysis"] = normalized
