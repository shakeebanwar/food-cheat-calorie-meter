"""Model registry, thresholds, and environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

SAMPLES_DIR = ROOT / "samples"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"

# Image preprocess
MAX_IMAGE_SIDE = 1024
JPEG_QUALITY = 85

# Confidence thresholds
HIGH_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.50
CONFIRMATION_THRESHOLD = 0.70

# API
DEFAULT_MAX_TOKENS = 8192  # reasoning models need headroom after reasoning tokens
DEFAULT_TEMPERATURE = 0.2
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0
REQUEST_TIMEOUT_S = 240.0


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    provider_hint: str
    input_per_m: float  # USD per 1M prompt tokens
    output_per_m: float  # USD per 1M completion tokens
    notes: str = ""


# Locked model set for this R&D run.
# deepseek/deepseek-v4-flash-0731 is text-only — replaced by Qwen VL.
MODELS: dict[str, ModelSpec] = {
    "glm": ModelSpec(
        id="z-ai/glm-5.3-flash",
        label="GLM 5.3 Flash",
        provider_hint="Z.ai",
        input_per_m=0.075,
        output_per_m=0.25,
        notes="Vision-capable flash model; strong cost/performance candidate.",
    ),
    "luna": ModelSpec(
        id="openai/gpt-5.6-luna",
        label="GPT-5.6 Luna",
        provider_hint="OpenAI/Azure",
        input_per_m=0.20,
        output_per_m=1.20,
        notes="Premium vision model; baseline for accuracy ceiling.",
    ),
    "qwen": ModelSpec(
        id="qwen/qwen3-vl-32b-instruct",
        label="Qwen3-VL 32B",
        provider_hint="Qwen",
        input_per_m=0.104,
        output_per_m=0.416,
        notes=(
            "Replacement for deepseek/deepseek-v4-flash-0731 "
            "(text-only, no image endpoints on OpenRouter)."
        ),
    ),
    "gpt4o_mini": ModelSpec(
        id="openai/gpt-4o-mini",
        label="GPT-4o Mini",
        provider_hint="OpenAI",
        input_per_m=0.15,
        output_per_m=0.60,
        notes="Affordable OpenAI vision model; good cost/accuracy balance.",
    ),
}

MODEL_ORDER = ["qwen", "glm", "gpt4o_mini", "luna"]  # cheap → premium for router


def estimated_cost(prompt_tokens: int, completion_tokens: int, model_key: str) -> float:
    """Fallback cost estimate from static pricing table."""
    m = MODELS[model_key]
    return (prompt_tokens / 1_000_000) * m.input_per_m + (
        completion_tokens / 1_000_000
    ) * m.output_per_m


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "judgments").mkdir(parents=True, exist_ok=True)
