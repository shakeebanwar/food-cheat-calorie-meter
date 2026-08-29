"""Objective metrics from benchmark JSONL (no judge needed)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pctile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize_model(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("success")]
    fail = [r for r in rows if not r.get("success")]
    latencies = [r["latency_ms"] for r in ok]
    costs = [r.get("usage", {}).get("cost_usd", 0) or 0 for r in ok]
    prompts = [r.get("usage", {}).get("prompt_tokens", 0) or 0 for r in ok]
    comps_tok = [r.get("usage", {}).get("completion_tokens", 0) or 0 for r in ok]
    reasoning = [r.get("usage", {}).get("reasoning_tokens", 0) or 0 for r in ok]
    n_components = [
        len((r.get("analysis") or {}).get("components") or []) for r in ok
    ]
    confidences = [
        (r.get("analysis") or {}).get("overall_confidence", 0) or 0 for r in ok
    ]
    drifts = []
    for r in ok:
        a = r.get("analysis") or {}
        totals = a.get("totals") or {}
        self_t = a.get("model_self_total_kcal") or 0
        computed = totals.get("calories") or 0
        if self_t > 0 and computed > 0:
            drifts.append(abs(self_t - computed) / computed * 100)

    retries = [r.get("attempts", 1) for r in rows]
    schema_fail = sum(
        1
        for r in fail
        if r.get("error") and ("ValidationError" in r["error"] or "JSON" in r["error"])
    )

    return {
        "n_runs": len(rows),
        "n_success": len(ok),
        "n_fail": len(fail),
        "success_rate": len(ok) / len(rows) if rows else 0,
        "failure_rate": len(fail) / len(rows) if rows else 0,
        "schema_fail": schema_fail,
        "avg_attempts": statistics.mean(retries) if retries else 0,
        "latency_p50_ms": _pctile(latencies, 0.5),
        "latency_p95_ms": _pctile(latencies, 0.95),
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0,
        "cost_mean_usd": statistics.mean(costs) if costs else 0,
        "cost_total_usd": sum(costs),
        "cost_per_1000_usd": (statistics.mean(costs) * 1000) if costs else 0,
        "prompt_tokens_mean": statistics.mean(prompts) if prompts else 0,
        "completion_tokens_mean": statistics.mean(comps_tok) if comps_tok else 0,
        "reasoning_tokens_mean": statistics.mean(reasoning) if reasoning else 0,
        "components_mean": statistics.mean(n_components) if n_components else 0,
        "confidence_mean": statistics.mean(confidences) if confidences else 0,
        "arithmetic_drift_pct_mean": statistics.mean(drifts) if drifts else None,
        "errors": [r.get("error") for r in fail][:10],
    }


def determinism_scores(rows: list[dict]) -> dict[str, Any]:
    """Measure consistency across reps and across duplicate image pair."""
    by_key: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if not r.get("success") or not r.get("analysis"):
            continue
        by_key[(r["image_id"], r["model_key"])].append(r)

    rep_consistency = []
    for (_iid, _mk), group in by_key.items():
        if len(group) < 2:
            continue
        names_sets = [
            frozenset(
                c.get("canonical_name") or c.get("name", "").lower()
                for c in (r["analysis"]["components"])
            )
            for r in group
        ]
        # pairwise Jaccard
        pairs = 0
        jacc = 0.0
        for i in range(len(names_sets)):
            for j in range(i + 1, len(names_sets)):
                a, b = names_sets[i], names_sets[j]
                if not a and not b:
                    continue
                pairs += 1
                jacc += len(a & b) / len(a | b)
        if pairs:
            rep_consistency.append(jacc / pairs)

    # Duplicate pair: Pasted image (4) vs (5)
    dup_scores = []
    for model_key in {r["model_key"] for r in rows}:
        g4 = [
            r
            for r in rows
            if r.get("success")
            and r["model_key"] == model_key
            and "Pasted image (4)" in r["image_id"]
        ]
        g5 = [
            r
            for r in rows
            if r.get("success")
            and r["model_key"] == model_key
            and "Pasted image (5)" in r["image_id"]
        ]
        if not g4 or not g5:
            continue
        s4 = frozenset(
            c.get("canonical_name") or c.get("name", "").lower()
            for c in g4[0]["analysis"]["components"]
        )
        s5 = frozenset(
            c.get("canonical_name") or c.get("name", "").lower()
            for c in g5[0]["analysis"]["components"]
        )
        if s4 or s5:
            dup_scores.append(len(s4 & s5) / len(s4 | s5))

    return {
        "rep_jaccard_mean": statistics.mean(rep_consistency) if rep_consistency else None,
        "duplicate_pair_jaccard_mean": statistics.mean(dup_scores) if dup_scores else None,
    }


def summarize_all(rows: list[dict]) -> dict[str, Any]:
    by_model: dict[str, list] = defaultdict(list)
    for r in rows:
        by_model[r["model_key"]].append(r)

    per_model = {k: summarize_model(v) for k, v in by_model.items()}
    # Attach determinism per model
    for k, v in by_model.items():
        per_model[k]["determinism"] = determinism_scores(v)

    return {
        "per_model": per_model,
        "global_determinism": determinism_scores(rows),
        "n_total": len(rows),
    }
