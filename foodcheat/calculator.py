"""Deterministic calorie calculation from model-provided kcal/100g + grams."""

from __future__ import annotations

from foodcheat.config import CONFIRMATION_THRESHOLD
from foodcheat.schema import Calories, FoodAnalysis, FoodComponent, Totals


def _component_calories(c: FoodComponent) -> Calories:
    per = c.nutrition_per_100g.kcal
    est = c.portion.estimate_g * per / 100.0
    lo = c.portion.min_g * per / 100.0
    hi = c.portion.max_g * per / 100.0
    return Calories(value=round(est, 1), min=round(lo, 1), max=round(hi, 1))


def apply_deterministic_calories(analysis: FoodAnalysis) -> FoodAnalysis:
    """Fill component.calories and totals; recompute confirmation flag."""
    total_kcal = 0.0
    total_min = 0.0
    total_max = 0.0
    protein = carbs = fat = 0.0

    for c in analysis.components:
        cal = _component_calories(c)
        c.calories = cal
        total_kcal += cal.value
        total_min += cal.min
        total_max += cal.max
        g = c.portion.estimate_g
        protein += c.nutrition_per_100g.protein_g * g / 100.0
        carbs += c.nutrition_per_100g.carbs_g * g / 100.0
        fat += c.nutrition_per_100g.fat_g * g / 100.0

    analysis.totals = Totals(
        calories=round(total_kcal, 1),
        min=round(total_min, 1),
        max=round(total_max, 1),
        protein_g=round(protein, 1),
        carbs_g=round(carbs, 1),
        fat_g=round(fat, 1),
    )

    # Confidence-weighted confirmation
    if analysis.components:
        weight_sum = sum(max(c.calories.value, 1.0) for c in analysis.components)
        weighted = (
            sum(c.confidence * max(c.calories.value, 1.0) for c in analysis.components)
            / weight_sum
        )
        # Keep model's overall if present, else fill
        if analysis.overall_confidence <= 0:
            analysis.overall_confidence = round(weighted, 3)

    low_major = any(
        c.confidence < 0.55 and (c.calories.value if c.calories else 0) > 50
        for c in analysis.components
    )
    if (
        analysis.overall_confidence < CONFIRMATION_THRESHOLD
        or low_major
        or not analysis.image_quality.usable
    ):
        analysis.needs_user_confirmation = True

    # Arithmetic drift warning
    if analysis.model_self_total_kcal > 0 and analysis.totals:
        drift = abs(analysis.model_self_total_kcal - analysis.totals.calories)
        if drift > max(30.0, 0.15 * analysis.totals.calories):
            msg = (
                f"Arithmetic drift: model_self_total_kcal="
                f"{analysis.model_self_total_kcal} vs computed={analysis.totals.calories}"
            )
            if msg not in analysis.warnings:
                analysis.warnings.append(msg)

    return analysis


def arithmetic_drift_pct(analysis: FoodAnalysis) -> float | None:
    if not analysis.totals or analysis.model_self_total_kcal <= 0:
        return None
    base = max(analysis.totals.calories, 1.0)
    return abs(analysis.model_self_total_kcal - analysis.totals.calories) / base * 100.0
