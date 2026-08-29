"""Robust JSON extraction and coercion into FoodAnalysis."""

from __future__ import annotations

import json
import re
from typing import Any

from foodcheat.schema import FoodAnalysis

# Common LLM number-word mistakes inside JSON numbers
_WORD_NUMS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
}


def extract_json_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model content")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return text


def repair_json_text(text: str) -> str:
    """Fix common LLM JSON defects before json.loads."""
    words = "|".join(_WORD_NUMS.keys())

    def _num_repl(m: re.Match) -> str:
        word = m.group(1).lower()
        frac = m.group(2) or ""
        return _WORD_NUMS[word] + (("." + frac) if frac else "")

    # : twenty.0  / : "twenty"  / : twenty,
    text = re.sub(
        rf':\s*"?({words})(?:\.(\d+))?"?(?=\s*[,}}\]])',
        lambda m: ": " + _num_repl(m),
        text,
        flags=re.IGNORECASE,
    )
    # Bare word-number tokens that still broke parsing
    text = re.sub(
        rf'\b({words})\.(\d+)\b',
        lambda m: _WORD_NUMS[m.group(1).lower()] + "." + m.group(2),
        text,
        flags=re.IGNORECASE,
    )
    # Trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    # NaN / Infinity
    text = re.sub(r"\bNaN\b", "0", text)
    text = re.sub(r"\bInfinity\b", "0", text)
    return text


def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower().replace(",", "")
        if s in _WORD_NUMS:
            return float(_WORD_NUMS[s])
        # "twenty.0"
        m = re.match(r"(" + "|".join(_WORD_NUMS.keys()) + r")\.?(\d*)", s)
        if m:
            base = float(_WORD_NUMS[m.group(1)])
            return base + (float("0." + m.group(2)) if m.group(2) else 0)
        try:
            return float(s)
        except ValueError:
            return default
    return default


def coerce_to_food_analysis(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize alternate / partial shapes into FoodAnalysis-compatible dict."""
    out: dict[str, Any] = {"schema_version": "1.0"}

    # image_quality
    iq = data.get("image_quality") or {}
    if not isinstance(iq, dict):
        iq = {}
    occlusion = iq.get("occlusion", "low")
    if occlusion in ("partial", "moderate", "some"):
        occlusion = "medium"
    if occlusion not in ("none", "low", "medium", "high"):
        occlusion = "low"
    issues = iq.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    for k in ("notes", "blur_level"):
        if iq.get(k):
            issues.append(f"{k}: {iq[k]}")
    out["image_quality"] = {
        "usable": bool(iq.get("usable", True)),
        "occlusion": occlusion,
        "issues": issues,
    }

    # meal_summary
    ms = data.get("meal_summary")
    if isinstance(ms, str):
        out["meal_summary"] = {"dish_name": ms, "cuisine": "", "meal_type": "unknown"}
    elif isinstance(ms, dict):
        out["meal_summary"] = {
            "dish_name": ms.get("dish_name") or ms.get("name") or "Unknown meal",
            "cuisine": ms.get("cuisine") or "",
            "meal_type": ms.get("meal_type") or "unknown",
        }
    else:
        out["meal_summary"] = {
            "dish_name": data.get("dish_name") or "Unknown meal",
            "cuisine": "",
            "meal_type": "unknown",
        }

    # components
    comps_in = data.get("components") or data.get("foods") or data.get("items") or []
    comps_out = []
    for i, c in enumerate(comps_in, 1):
        if not isinstance(c, dict):
            continue
        name = (
            c.get("name")
            or c.get("display_name")
            or c.get("food")
            or c.get("description")
            or (c.get("canonical_name") or "").replace("_", " ").title()
            or f"item_{i}"
        )
        if isinstance(name, str) and len(name) > 80:
            # Prefer short canonical over long description
            canon = c.get("canonical_name")
            if canon:
                name = str(canon).replace("_", " ").title()
            else:
                name = name[:77] + "..."
        canonical = c.get("canonical_name") or re.sub(
            r"[^a-z0-9]+", "_", str(name).lower()
        ).strip("_")
        portion = c.get("portion") or {}
        if not isinstance(portion, dict):
            portion = {}
        # Flat portion fields
        est = portion.get("estimate_g") or portion.get("grams") or c.get("portion_g") or c.get("grams")
        est = _as_float(est, 100.0)
        min_g = _as_float(portion.get("min_g"), est * 0.8)
        max_g = _as_float(portion.get("max_g"), est * 1.2)
        if min_g <= 0:
            min_g = est * 0.8
        if max_g < min_g:
            max_g = est * 1.2

        nutr = c.get("nutrition_per_100g") or c.get("nutrition") or {}
        if not isinstance(nutr, dict):
            nutr = {}
        kcal = _as_float(
            nutr.get("kcal") or nutr.get("calories") or c.get("kcal_per_100g"),
            150.0,
        )

        ingredients = c.get("ingredients") or []
        ing_out = []
        if isinstance(ingredients, list):
            for ing in ingredients:
                if isinstance(ing, str):
                    ing_out.append({"name": ing, "proportion": 0.0})
                elif isinstance(ing, dict):
                    ing_out.append(
                        {
                            "name": ing.get("name") or "unknown",
                            "proportion": min(
                                1.0, max(0.0, _as_float(ing.get("proportion"), 0.0))
                            ),
                        }
                    )

        conf = _as_float(c.get("confidence"), 0.7)
        conf = min(1.0, max(0.0, conf))

        count = c.get("count")
        if count is not None:
            try:
                count = int(float(count))
            except (TypeError, ValueError):
                count = None

        comps_out.append(
            {
                "id": c.get("id") or f"c{i}",
                "name": name,
                "canonical_name": canonical,
                "category": c.get("category") or "unknown",
                "ingredients": ing_out,
                "count": count,
                "portion": {
                    "estimate_g": est,
                    "min_g": min_g,
                    "max_g": max_g,
                    "method": portion.get("method") or c.get("portion_method") or "visual_estimate",
                    "reference_object": portion.get("reference_object") or "",
                    "household_measure": portion.get("household_measure")
                    or portion.get("household")
                    or "",
                },
                "nutrition_per_100g": {
                    "kcal": kcal,
                    "protein_g": _as_float(nutr.get("protein_g")),
                    "carbs_g": _as_float(nutr.get("carbs_g") or nutr.get("carbohydrates_g")),
                    "fat_g": _as_float(nutr.get("fat_g")),
                    "source": "llm_knowledge",
                },
                "calories": None,
                "confidence": conf,
                "notes": c.get("notes") or "",
            }
        )

    if not comps_out:
        raise ValueError("No components found after coercion")

    out["components"] = comps_out
    out["model_self_total_kcal"] = _as_float(
        data.get("model_self_total_kcal") or data.get("total_calories"), 0.0
    )
    oc = _as_float(data.get("overall_confidence"), 0.7)
    out["overall_confidence"] = min(1.0, max(0.0, oc))
    out["needs_user_confirmation"] = bool(data.get("needs_user_confirmation", True))
    warnings = data.get("warnings") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    out["warnings"] = warnings
    out["totals"] = None
    return out


def repair_truncated_json(text: str) -> str:
    """Best-effort close of truncated JSON objects/arrays."""
    # If already parseable, return
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Drop trailing incomplete string / token
    text = text.rstrip()
    # If ends mid-string, close quote
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
    if in_str:
        text += '"'

    # Remove trailing comma / colon dangling
    text = re.sub(r"[,:\s]+$", "", text)

    # Close open brackets
    stack = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    closers = {"{": "}", "[": "]"}
    while stack:
        text += closers[stack.pop()]
    return text


def parse_food_analysis(content: str) -> FoodAnalysis:
    text = extract_json_text(content)
    text = repair_json_text(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        text2 = repair_json_text(text)
        text2 = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text2)
        try:
            data = json.loads(text2)
        except json.JSONDecodeError:
            text3 = repair_truncated_json(text2)
            data = json.loads(text3)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON is not an object")

    try:
        return FoodAnalysis.model_validate(data)
    except Exception:
        coerced = coerce_to_food_analysis(data)
        return FoodAnalysis.model_validate(coerced)
