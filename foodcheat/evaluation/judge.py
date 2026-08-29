"""Blind LLM-as-judge scaffold + rubric.

The judging agent (this session's AI) scores anonymized outputs.
Scripts prepare blind packs and apply revealed scores.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from foodcheat.config import RESULTS_DIR, ensure_dirs
from foodcheat.evaluation.metrics import load_results

# Fixed rubric — used by the human/AI judge for consistent scoring.
JUDGE_RUBRIC = """
Score each dimension 0–10, then overall /10.

1. Food identification (0–10)
   - Correct primary dish / cuisine recognition
   - Penalize wrong dish identity heavily

2. Component decomposition (0–10)  [MOST IMPORTANT for FoodCheat]
   - Did it split multi-item plates correctly?
   - Penalties: merged plate (-4), missed side/sauce (-2 each), hallucinated item (-2 each)
   - Bonus for correctly separating dips/dressings/sides

3. Portion plausibility (0–10)
   - Are gram estimates in a realistic band for what's visible?
   - Counted discrete units correctly?

4. Calorie / nutrition plausibility (0–10)
   - kcal/100g sensible for cooking method?
   - Total calories in a believable range for the plate size?

5. Confidence calibration (0–10)
   - High confidence on clear images; lower when occluded/ambiguous
   - needs_user_confirmation set appropriately

Overall /10 = weighted:
  0.15*food + 0.35*components + 0.20*portion + 0.20*calories + 0.10*confidence

Also note: cost-efficiency opinion (separate from quality) — prefer high quality at low cost.
"""


def _blind_code(image_id: str, model_key: str, salt: str) -> str:
    h = hashlib.sha256(f"{salt}|{image_id}|{model_key}".encode()).hexdigest()[:6]
    return f"M-{h.upper()}"


def build_blind_pack(
    jsonl_path: Path,
    *,
    rep: int = 1,
    salt: str = "foodcheat-rd-2026",
    out_path: Path | None = None,
) -> Path:
    """Create anonymized judging pack (one successful run per image×model at given rep)."""
    ensure_dirs()
    rows = load_results(jsonl_path)
    selected = [
        r
        for r in rows
        if r.get("success") and r.get("analysis") and r.get("rep") == rep
    ]

    pack_items = []
    key_map = {}  # blind_code -> model_key

    # Group by image
    by_image: dict[str, list] = {}
    for r in selected:
        by_image.setdefault(r["image_id"], []).append(r)

    for image_id, group in sorted(by_image.items()):
        entries = []
        for r in group:
            code = _blind_code(image_id, r["model_key"], salt)
            key_map[f"{image_id}|{code}"] = {
                "model_key": r["model_key"],
                "model_id": r["model_id"],
                "run_id": r["run_id"],
                "cost_usd": r.get("usage", {}).get("cost_usd"),
                "latency_ms": r.get("latency_ms"),
            }
            a = r["analysis"]
            entries.append(
                {
                    "blind_code": code,
                    "meal_summary": a.get("meal_summary"),
                    "components": [
                        {
                            "name": c.get("name"),
                            "canonical_name": c.get("canonical_name"),
                            "category": c.get("category"),
                            "count": c.get("count"),
                            "portion_g": (c.get("portion") or {}).get("estimate_g"),
                            "household": (c.get("portion") or {}).get("household_measure"),
                            "kcal_per_100g": (c.get("nutrition_per_100g") or {}).get("kcal"),
                            "calories": (c.get("calories") or {}).get("value"),
                            "confidence": c.get("confidence"),
                        }
                        for c in a.get("components") or []
                    ],
                    "totals_kcal": (a.get("totals") or {}).get("calories"),
                    "overall_confidence": a.get("overall_confidence"),
                    "warnings": a.get("warnings"),
                    # cost/latency HIDDEN during blind scoring
                }
            )
        random.Random(salt + image_id).shuffle(entries)
        pack_items.append({"image_id": image_id, "outputs": entries})

    out_path = out_path or (RESULTS_DIR / "judgments" / "blind_pack.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rubric": JUDGE_RUBRIC,
        "salt": salt,
        "rep": rep,
        "items": pack_items,
        "key_map": key_map,  # keep for reveal — judge should ignore until scored
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    # Also write a judge-facing version WITHOUT key_map
    judge_view = {
        "rubric": JUDGE_RUBRIC,
        "instruction": (
            "Score each blind_code output. Do NOT try to guess which model. "
            "Write scores to judgments/scores.json"
        ),
        "items": pack_items,
    }
    (out_path.parent / "blind_pack_judge_view.json").write_text(
        json.dumps(judge_view, indent=2, ensure_ascii=False)
    )
    return out_path


def reveal_scores(scores_path: Path, blind_pack_path: Path) -> Path:
    """Join blind scores with key_map → revealed judgments."""
    pack = json.loads(blind_pack_path.read_text())
    scores = json.loads(scores_path.read_text())
    key_map = pack["key_map"]

    revealed = []
    for entry in scores.get("scores", []):
        image_id = entry["image_id"]
        code = entry["blind_code"]
        meta = key_map.get(f"{image_id}|{code}", {})
        revealed.append({**entry, **meta})

    out = RESULTS_DIR / "judgments" / "revealed_scores.json"
    out.write_text(
        json.dumps(
            {"rubric": JUDGE_RUBRIC, "scores": revealed},
            indent=2,
            ensure_ascii=False,
        )
    )
    return out


def aggregate_judge_scores(revealed_path: Path) -> dict[str, Any]:
    data = json.loads(revealed_path.read_text())
    by_model: dict[str, list] = {}
    for s in data["scores"]:
        by_model.setdefault(s["model_key"], []).append(s)

    summary = {}
    for mk, items in by_model.items():
        def avg(field: str) -> float:
            vals = [i[field] for i in items if field in i and i[field] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        summary[mk] = {
            "n": len(items),
            "overall_mean": avg("overall"),
            "food_id_mean": avg("food_id"),
            "components_mean": avg("components"),
            "portion_mean": avg("portion"),
            "calories_mean": avg("calories"),
            "confidence_mean": avg("confidence_cal"),
            "per_image": [
                {
                    "image_id": i["image_id"],
                    "overall": i.get("overall"),
                    "reason": i.get("reason"),
                }
                for i in items
            ],
        }
    return summary
