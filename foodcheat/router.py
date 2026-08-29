"""Confidence-based multi-model router (cheap → premium escalation)."""

from __future__ import annotations

from pathlib import Path

from foodcheat.analyzer import analyze_image
from foodcheat.client import OpenRouterClient
from foodcheat.config import HIGH_CONFIDENCE, MODEL_ORDER
from foodcheat.schema import RunResult


def route_analyze(
    image_path: str | Path,
    *,
    client: OpenRouterClient | None = None,
    model_order: list[str] | None = None,
    accept_threshold: float = HIGH_CONFIDENCE,
) -> dict:
    """Try models cheap→premium until confidence is high enough.

    Returns {accepted: RunResult, attempts: list[RunResult], escalated: bool}.
    """
    client = client or OpenRouterClient()
    order = model_order or MODEL_ORDER
    attempts: list[RunResult] = []

    for key in order:
        result = analyze_image(image_path, key, client=client, rep=1)
        attempts.append(result)
        if not result.success or not result.analysis:
            continue
        if result.analysis.overall_confidence >= accept_threshold:
            return {
                "accepted": result,
                "attempts": attempts,
                "escalated": len(attempts) > 1,
                "reason": "confidence_ok",
            }

    # Fall back to last successful, else last attempt
    successful = [a for a in attempts if a.success and a.analysis]
    accepted = successful[-1] if successful else attempts[-1]
    return {
        "accepted": accepted,
        "attempts": attempts,
        "escalated": len(attempts) > 1,
        "reason": "exhausted_or_best_effort",
    }
