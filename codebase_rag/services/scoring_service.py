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
        if idx >= len(scoring_findings):
            break
        item = scoring_findings[idx]
        if not isinstance(item, dict):
            continue
        analysis = item.get("analysis", {})
        if not isinstance(analysis, dict):
            continue
        score_val = analysis.get("score")
        if not isinstance(score_val, (int, float)):
            continue
        explanation = analysis.get("explanation")
        if explanation is not None and not isinstance(explanation, str):
            explanation = str(explanation)
        updates.append((finding_id, float(score_val), explanation))
    return updates


def build_human_scoring_explanation(analysis: dict[str, Any]) -> str:
    """Generate a plain-language scoring explanation if model output is weak/missing."""
    verdict = analysis.get("verdict")
    score = analysis.get("score")
    breakdown = analysis.get("scoring_breakdown")
    parts: list[str] = []
    if isinstance(verdict, str) and isinstance(score, (int, float)):
        parts.append(
            f"This finding is rated {int(score)} ({verdict}) based on the available code evidence."
        )
    elif isinstance(score, (int, float)):
        parts.append(f"This finding is rated {int(score)} based on the available code evidence.")

    if isinstance(breakdown, dict):
        top_reason = None
        top_name = None
        top_score = -1.0
        for name, item in breakdown.items():
            if isinstance(item, dict):
                bs = item.get("score")
                reason = item.get("reason")
                if isinstance(bs, (int, float)) and isinstance(reason, str) and bs > top_score:
                    top_score = float(bs)
                    top_reason = reason.strip()
                    top_name = str(name).replace("_", " ")
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
        if not isinstance(item, dict):
            continue
        analysis = item.get("analysis")
        if not isinstance(analysis, dict):
            continue

        explanation = analysis.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = build_human_scoring_explanation(analysis)
        else:
            explanation = explanation.strip()

        normalized: dict[str, Any] = {}
        if "verdict" in analysis:
            normalized["verdict"] = analysis.get("verdict")
        if "score" in analysis:
            normalized["score"] = analysis.get("score")
        normalized["explanation"] = explanation
        if "scoring_breakdown" in analysis:
            normalized["scoring_breakdown"] = analysis.get("scoring_breakdown")
        for k, v in analysis.items():
            if k not in {"verdict", "score", "explanation", "scoring_breakdown"}:
                normalized[k] = v
        item["analysis"] = normalized
