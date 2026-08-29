"""Generate Markdown + HTML evaluation report."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from foodcheat.config import MODELS, REPORTS_DIR, SAMPLES_DIR, ensure_dirs
from foodcheat.evaluation.metrics import load_results, summarize_all


def _img_b64(path: Path, max_side: int = 480) -> str:
    from foodcheat.preprocess import load_and_preprocess

    data, mime, _ = load_and_preprocess(path, max_side=max_side, quality=70)
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _find_sample(image_id: str) -> Path | None:
    for p in SAMPLES_DIR.iterdir():
        if p.stem == image_id:
            return p
    return None


def _pick_rep1(rows: list[dict]) -> dict[str, dict]:
    """image_id -> model_key -> row (rep==1 preferred)."""
    out: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if not r.get("success"):
            continue
        iid, mk = r["image_id"], r["model_key"]
        cur = out[iid].get(mk)
        if cur is None or (r.get("rep") == 1 and cur.get("rep") != 1):
            out[iid][mk] = r
    return out


def build_report(
    jsonl_path: Path,
    *,
    judgments_path: Path | None = None,
    title: str = "FoodCheat AI Vision — Model Benchmark Report",
) -> tuple[Path, Path]:
    ensure_dirs()
    rows = load_results(jsonl_path)
    metrics = summarize_all(rows)
    by_image = _pick_rep1(rows)

    judgments = None
    if judgments_path and judgments_path.exists():
        judgments = json.loads(judgments_path.read_text())

    # Index judgments by image+model
    judge_idx: dict[tuple, dict] = {}
    if judgments:
        for s in judgments.get("scores", []):
            judge_idx[(s["image_id"], s["model_key"])] = s

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md: list[str] = []
    md.append(f"# {title}\n")
    md.append(f"_Generated: {ts}_\n")
    md.append("## Executive summary\n")
    md.append(
        "Compared three OpenRouter vision models on multi-component food calorie "
        "estimation. DeepSeek V4 Flash 0731 was **excluded** (text-only — no image "
        "endpoints); replaced by `qwen/qwen3-vl-32b-instruct`.\n"
    )
    md.append("Nutrition values come from each model's trained knowledge; calorie "
              "**arithmetic** is deterministic in Python.\n")

    # Leaderboard
    md.append("## Leaderboard (objective metrics)\n")
    md.append(
        "| Model | Success | Latency p50 | Cost/scan | Cost/1000 | "
        "Components ⌀ | Confidence ⌀ | Drift % |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for mk, m in metrics["per_model"].items():
        label = MODELS[mk].label if mk in MODELS else mk
        drift = m["arithmetic_drift_pct_mean"]
        drift_s = f"{drift:.1f}" if drift is not None else "—"
        md.append(
            f"| {label} (`{MODELS[mk].id if mk in MODELS else mk}`) | "
            f"{m['success_rate']*100:.0f}% | {m['latency_p50_ms']:.0f} ms | "
            f"${m['cost_mean_usd']:.5f} | ${m['cost_per_1000_usd']:.2f} | "
            f"{m['components_mean']:.1f} | {m['confidence_mean']:.2f} | {drift_s} |\n"
        )

    if judgments:
        md.append("\n## Judge scores (blind, /10)\n")
        md.append("| Model | Overall | Food ID | Components | Portion | Calories | Conf. cal. |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|\n")
        from foodcheat.evaluation.judge import aggregate_judge_scores

        # Write temp if needed — judgments_path already revealed
        agg = {}
        by_m: dict[str, list] = defaultdict(list)
        for s in judgments["scores"]:
            by_m[s["model_key"]].append(s)
        for mk, items in by_m.items():
            def avg(field: str, items=items) -> float:
                vals = [i[field] for i in items if i.get(field) is not None]
                return sum(vals) / len(vals) if vals else 0

            agg[mk] = {
                "overall": avg("overall"),
                "food_id": avg("food_id"),
                "components": avg("components"),
                "portion": avg("portion"),
                "calories": avg("calories"),
                "confidence_cal": avg("confidence_cal"),
            }
        for mk, a in agg.items():
            label = MODELS[mk].label if mk in MODELS else mk
            md.append(
                f"| {label} | {a['overall']:.1f} | {a['food_id']:.1f} | "
                f"{a['components']:.1f} | {a['portion']:.1f} | "
                f"{a['calories']:.1f} | {a['confidence_cal']:.1f} |\n"
            )

    md.append("\n## Pricing reference (from cost-estimation.png / OpenRouter)\n")
    md.append("| Model | Input $/M | Output $/M |\n|---|---:|---:|\n")
    for mk, spec in MODELS.items():
        md.append(f"| {spec.label} | ${spec.input_per_m} | ${spec.output_per_m} |\n")

    md.append("\n## Per-image results\n")

    for image_id in sorted(by_image.keys()):
        models_map = by_image[image_id]
        sample = _find_sample(image_id)
        md.append(f"\n### `{image_id}`\n")
        if sample:
            md.append(f"![input]({sample.name}) <!-- path: {sample} -->\n")

        for mk, r in sorted(models_map.items()):
            spec = MODELS.get(mk)
            label = spec.label if spec else mk
            a = r["analysis"]
            u = r.get("usage") or {}
            md.append(f"\n#### {label}\n")
            md.append(
                f"- **Latency:** {r['latency_ms']:.0f} ms · "
                f"**Cost:** ${u.get('cost_usd', 0):.6f} · "
                f"**Tokens:** in={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
                f"(reasoning={u.get('reasoning_tokens', 0)})\n"
            )
            md.append(
                f"- **Dish:** {a['meal_summary'].get('dish_name')} "
                f"({a['meal_summary'].get('cuisine')}) · "
                f"**Confidence:** {a.get('overall_confidence')} · "
                f"**Total kcal:** {(a.get('totals') or {}).get('calories')}\n"
            )
            md.append(
                "| # | Component | Portion (g) | kcal/100g | Calories | Conf |\n"
                "|---|---|---:|---:|---:|---:|\n"
            )
            for i, c in enumerate(a.get("components") or [], 1):
                md.append(
                    f"| {i} | {c.get('name')} | "
                    f"{(c.get('portion') or {}).get('estimate_g')} | "
                    f"{(c.get('nutrition_per_100g') or {}).get('kcal')} | "
                    f"{(c.get('calories') or {}).get('value')} | "
                    f"{c.get('confidence')} |\n"
                )

            js = judge_idx.get((image_id, mk))
            if js:
                md.append(
                    f"\n**Judge score: {js.get('overall')}/10** — {js.get('reason', '')}\n"
                )

    md.append("\n## Recommendation\n")
    md.append("_Filled after judging — see final section in HTML report / judgments._\n")

    md_path = REPORTS_DIR / "benchmark_report.md"
    md_path.write_text("".join(md), encoding="utf-8")

    # HTML with embedded images
    html = _render_html(title, ts, metrics, by_image, judge_idx, judgments)
    html_path = REPORTS_DIR / "benchmark_report.html"
    html_path.write_text(html, encoding="utf-8")
    return md_path, html_path


def _render_html(
    title: str,
    ts: str,
    metrics: dict,
    by_image: dict,
    judge_idx: dict,
    judgments: dict | None,
) -> str:
    parts: list[str] = []
    parts.append(
        f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{title}</title>
<style>
  :root {{ --bg:#0f1419; --card:#1a222c; --text:#e7ecf1; --muted:#9aa7b5; --accent:#3dd6c6; --warn:#f0a030; }}
  body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background:var(--bg); color:var(--text);
         margin:0; padding:2rem; line-height:1.5; }}
  h1,h2,h3,h4 {{ font-family: "IBM Plex Serif", Georgia, serif; }}
  h1 {{ color:var(--accent); }}
  .meta {{ color:var(--muted); margin-bottom:2rem; }}
  table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:0.92rem; }}
  th,td {{ border:1px solid #2a3542; padding:0.45rem 0.6rem; text-align:left; }}
  th {{ background:#243040; }}
  .card {{ background:var(--card); border-radius:10px; padding:1.2rem; margin:1.5rem 0;
           border:1px solid #2a3542; }}
  .grid {{ display:grid; grid-template-columns: 280px 1fr; gap:1.2rem; }}
  img.sample {{ width:100%; border-radius:8px; background:#111; }}
  .score {{ font-size:1.4rem; color:var(--accent); font-weight:700; }}
  .reason {{ color:var(--muted); font-size:0.9rem; }}
  .model-block {{ margin:1rem 0; padding:0.8rem; background:#121820; border-radius:8px; }}
  .pill {{ display:inline-block; background:#243040; padding:0.15rem 0.5rem; border-radius:4px;
           font-size:0.8rem; margin-right:0.4rem; }}
  @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style></head><body>
<h1>{title}</h1>
<p class="meta">{ts}</p>
"""
    )

    parts.append("<h2>Leaderboard</h2><table><tr>"
                 "<th>Model</th><th>Success</th><th>p50 latency</th><th>Cost/scan</th>"
                 "<th>Cost/1000</th><th>Components</th><th>Confidence</th></tr>")
    for mk, m in metrics["per_model"].items():
        label = MODELS[mk].label if mk in MODELS else mk
        parts.append(
            f"<tr><td>{label}</td><td>{m['success_rate']*100:.0f}%</td>"
            f"<td>{m['latency_p50_ms']:.0f} ms</td>"
            f"<td>${m['cost_mean_usd']:.5f}</td>"
            f"<td>${m['cost_per_1000_usd']:.2f}</td>"
            f"<td>{m['components_mean']:.1f}</td>"
            f"<td>{m['confidence_mean']:.2f}</td></tr>"
        )
    parts.append("</table>")

    for image_id, models_map in sorted(by_image.items()):
        sample = _find_sample(image_id)
        img_tag = ""
        if sample:
            try:
                img_tag = f'<img class="sample" src="{_img_b64(sample)}" alt="{image_id}"/>'
            except Exception:  # noqa: BLE001
                img_tag = f"<p>Image: {sample.name}</p>"

        parts.append(f'<div class="card"><h3>{image_id}</h3><div class="grid">')
        parts.append(f"<div>{img_tag}</div><div>")

        for mk, r in sorted(models_map.items()):
            label = MODELS[mk].label if mk in MODELS else mk
            a = r["analysis"]
            u = r.get("usage") or {}
            js = judge_idx.get((image_id, mk))
            score_html = ""
            if js:
                score_html = (
                    f'<div class="score">{js.get("overall")}/10</div>'
                    f'<div class="reason">{js.get("reason", "")}</div>'
                )
            parts.append(f'<div class="model-block"><h4>{label}</h4>{score_html}')
            parts.append(
                f'<span class="pill">${u.get("cost_usd", 0):.5f}</span>'
                f'<span class="pill">{r["latency_ms"]:.0f} ms</span>'
                f'<span class="pill">kcal {(a.get("totals") or {}).get("calories")}</span>'
                f'<span class="pill">conf {a.get("overall_confidence")}</span>'
            )
            parts.append(
                "<table><tr><th>Component</th><th>g</th><th>kcal/100g</th>"
                "<th>Cal</th><th>Conf</th></tr>"
            )
            for c in a.get("components") or []:
                parts.append(
                    f"<tr><td>{c.get('name')}</td>"
                    f"<td>{(c.get('portion') or {}).get('estimate_g')}</td>"
                    f"<td>{(c.get('nutrition_per_100g') or {}).get('kcal')}</td>"
                    f"<td>{(c.get('calories') or {}).get('value')}</td>"
                    f"<td>{c.get('confidence')}</td></tr>"
                )
            parts.append("</table></div>")

        parts.append("</div></div></div>")

    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/benchmark.jsonl")
    jp = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    md, html = build_report(path, judgments_path=jp)
    print(md, html)
