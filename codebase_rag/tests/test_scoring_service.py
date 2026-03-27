from __future__ import annotations

from codebase_rag.services.scoring_service import (
    extract_scored_findings,
    normalize_scoring_output,
)


def test_extract_scored_findings_maps_by_index() -> None:
    response_data = {
        "scoring": {
            "findings": [
                {"analysis": {"score": 92, "explanation": "high risk"}},
                {"analysis": {"score": 40.5, "explanation": "lower risk"}},
            ]
        }
    }
    result = extract_scored_findings(
        org_tool_findings_ids=["id-1", "id-2"],
        response_data=response_data,
    )
    assert result == [
        ("id-1", 92.0, "high risk"),
        ("id-2", 40.5, "lower risk"),
    ]


def test_normalize_scoring_output_keeps_explanation_and_order() -> None:
    response_data = {
        "scoring": {
            "findings": [
                {
                    "analysis": {
                        "verdict": "True Positive",
                        "score": 80,
                        "explanation": "This is likely exploitable and lacks controls.",
                        "scoring_breakdown": {},
                    }
                }
            ]
        }
    }

    normalize_scoring_output(response_data)
    analysis = response_data["scoring"]["findings"][0]["analysis"]
    assert analysis["explanation"] == "This is likely exploitable and lacks controls."
    assert list(analysis.keys())[:3] == ["verdict", "score", "explanation"]


def test_normalize_scoring_output_generates_explanation_when_missing() -> None:
    response_data = {
        "scoring": {
            "findings": [
                {
                    "analysis": {
                        "verdict": "Likely True Positive",
                        "score": 78,
                        "scoring_breakdown": {
                            "reachability": {
                                "score": 16,
                                "reason": "Execution path reaches vulnerable sink.",
                            }
                        },
                    }
                }
            ]
        }
    }
    normalize_scoring_output(response_data)
    explanation = response_data["scoring"]["findings"][0]["analysis"]["explanation"]
    assert isinstance(explanation, str)
    assert explanation.strip()
