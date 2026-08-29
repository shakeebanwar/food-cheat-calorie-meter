"""Pydantic schemas for multi-component food analysis."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Ingredient(BaseModel):
    name: str
    proportion: float = Field(
        ge=0.0,
        le=1.0,
        description="Rough mass fraction of this ingredient in the component (0–1).",
    )


class Portion(BaseModel):
    estimate_g: float = Field(gt=0, description="Best-estimate edible weight in grams.")
    min_g: float = Field(gt=0, description="Lower bound of plausible weight.")
    max_g: float = Field(gt=0, description="Upper bound of plausible weight.")
    method: str = Field(
        description="How portion was estimated, e.g. volume_from_bowl_diameter."
    )
    reference_object: str = Field(
        default="",
        description="Visual reference used (plate ~26cm, fork ~19cm, etc.).",
    )
    household_measure: str = Field(
        default="",
        description="Human-readable measure, e.g. '1.5 cups', '2 pieces'.",
    )

    @field_validator("max_g")
    @classmethod
    def max_ge_min(cls, v: float, info) -> float:  # noqa: ANN001
        min_g = info.data.get("min_g")
        if min_g is not None and v < min_g:
            raise ValueError("max_g must be >= min_g")
        return v


class NutritionPer100g(BaseModel):
    kcal: float = Field(ge=0, description="Kilocalories per 100g edible portion.")
    protein_g: float = Field(ge=0, default=0)
    carbs_g: float = Field(ge=0, default=0)
    fat_g: float = Field(ge=0, default=0)
    source: Literal["llm_knowledge"] = "llm_knowledge"


class Calories(BaseModel):
    value: float = Field(ge=0)
    min: float = Field(ge=0)
    max: float = Field(ge=0)


class FoodComponent(BaseModel):
    id: str
    name: str
    canonical_name: str = Field(
        description="snake_case stable identifier, e.g. chickpea_curry."
    )
    category: str = Field(
        description="e.g. rice_dish, curry, bread, sauce, garnish, protein, dessert."
    )
    ingredients: list[Ingredient] = Field(default_factory=list)
    count: int | None = Field(
        default=None,
        description="If discrete units (kofta balls, skewers), the count; else null.",
    )
    portion: Portion
    nutrition_per_100g: NutritionPer100g
    calories: Calories | None = Field(
        default=None,
        description="Filled by deterministic calculator; model may omit.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class ImageQuality(BaseModel):
    usable: bool = True
    occlusion: Literal["none", "low", "medium", "high"] = "low"
    issues: list[str] = Field(default_factory=list)


class MealSummary(BaseModel):
    dish_name: str
    cuisine: str = ""
    meal_type: str = Field(
        default="unknown",
        description="breakfast | lunch | dinner | snack | dessert | unknown",
    )


class Totals(BaseModel):
    calories: float = 0
    min: float = 0
    max: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0


class FoodAnalysis(BaseModel):
    """Strict structured output expected from the vision model."""

    schema_version: Literal["1.0"] = "1.0"
    image_quality: ImageQuality
    meal_summary: MealSummary
    components: list[FoodComponent] = Field(
        min_length=1,
        description="Every visibly distinct edible component. NEVER merge a plate into one item.",
    )
    model_self_total_kcal: float = Field(
        ge=0,
        description="Model's own sum of component calories (for arithmetic-drift check).",
    )
    overall_confidence: float = Field(ge=0.0, le=1.0)
    needs_user_confirmation: bool = True
    warnings: list[str] = Field(default_factory=list)
    # Filled post-hoc by calculator
    totals: Totals | None = None


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_estimated: bool = False


class RunResult(BaseModel):
    """Envelope around one model call."""

    run_id: str
    image_path: str
    image_id: str
    model_key: str
    model_id: str
    rep: int
    success: bool
    latency_ms: float
    attempts: int
    error: str | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    analysis: FoodAnalysis | None = None
    raw_content: str | None = None
    provider: str | None = None


# JSON Schema for OpenRouter response_format (strict)
def food_analysis_json_schema() -> dict[str, Any]:
    """Export a JSON Schema suitable for OpenRouter structured outputs.

    Keeps additionalProperties:false and required fields for strict mode.
    """
    schema = FoodAnalysis.model_json_schema()

    def _strictify(node: dict[str, Any]) -> dict[str, Any]:
        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties", {})
            node["properties"] = {k: _strictify(v) for k, v in props.items()}
            node["additionalProperties"] = False
            # Strict OpenAI-style schemas require every property to be listed
            node["required"] = list(node["properties"].keys())
        if "items" in node and isinstance(node["items"], dict):
            node["items"] = _strictify(node["items"])
        if "$defs" in node:
            node["$defs"] = {k: _strictify(v) for k, v in node["$defs"].items()}
        # Remove defaults that confuse some providers' strict validators
        node.pop("default", None)
        # Pydantic may emit anyOf for Optional — flatten simple cases
        if "anyOf" in node:
            non_null = [x for x in node["anyOf"] if x.get("type") != "null"]
            if len(non_null) == 1:
                merged = {**non_null[0]}
                for k, v in node.items():
                    if k != "anyOf":
                        merged.setdefault(k, v)
                return _strictify(merged)
        return node

    schema = _strictify(schema)
    schema["title"] = "FoodAnalysis"
    schema["additionalProperties"] = False
    return schema
