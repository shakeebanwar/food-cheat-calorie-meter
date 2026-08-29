"""Orchestrator: image → FoodAnalysis with usage/latency envelope."""

from __future__ import annotations

import uuid
from pathlib import Path

from foodcheat.calculator import apply_deterministic_calories
from foodcheat.client import OpenRouterClient, OpenRouterError
from foodcheat.config import MODELS, ModelSpec
from foodcheat.parse import parse_food_analysis
from foodcheat.preprocess import to_data_url
from foodcheat.prompts import build_messages
from foodcheat.schema import RunResult, UsageStats


def analyze_image(
    image_path: str | Path,
    model_key: str,
    *,
    client: OpenRouterClient | None = None,
    rep: int = 1,
    model_spec: ModelSpec | None = None,
) -> RunResult:
    """Run one food analysis. Never raises — failures captured in RunResult."""
    image_path = Path(image_path)
    spec = model_spec or MODELS[model_key]
    run_id = f"{image_path.stem}__{model_key}__r{rep}__{uuid.uuid4().hex[:8]}"
    image_id = image_path.stem

    client = client or OpenRouterClient()
    attempts = 0
    latency_ms = 0.0
    usage = UsageStats()
    provider = None
    raw_content = None

    try:
        data_url, _meta = to_data_url(image_path)
        messages = build_messages(data_url)

        # Plain JSON is more reliable across providers than strict structured outputs
        # (Luna emits off-schema fields; GLM sometimes truncates under structured mode).
        result = client.chat(
            spec.id,
            messages,
            model_key=model_key,
            use_structured=False,
        )

        attempts = result["attempts"]
        latency_ms = result["latency_ms"]
        usage = result["usage"]
        provider = result.get("provider")
        raw_content = result["content"]

        analysis = parse_food_analysis(raw_content)
        analysis = apply_deterministic_calories(analysis)

        return RunResult(
            run_id=run_id,
            image_path=str(image_path),
            image_id=image_id,
            model_key=model_key,
            model_id=spec.id,
            rep=rep,
            success=True,
            latency_ms=latency_ms,
            attempts=attempts,
            usage=usage,
            analysis=analysis,
            raw_content=raw_content,
            provider=provider,
        )
    except Exception as e:  # noqa: BLE001 — capture all for benchmark
        return RunResult(
            run_id=run_id,
            image_path=str(image_path),
            image_id=image_id,
            model_key=model_key,
            model_id=spec.id,
            rep=rep,
            success=False,
            latency_ms=latency_ms,
            attempts=attempts or 1,
            error=f"{type(e).__name__}: {e}",
            usage=usage,
            analysis=None,
            raw_content=raw_content,
            provider=provider,
        )
