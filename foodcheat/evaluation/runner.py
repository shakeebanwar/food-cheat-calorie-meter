"""Benchmark harness — resumable JSONL runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from foodcheat.analyzer import analyze_image
from foodcheat.client import OpenRouterClient
from foodcheat.config import MODELS, RESULTS_DIR, SAMPLES_DIR, ensure_dirs


def list_sample_images(samples_dir: Path | None = None) -> list[Path]:
    d = samples_dir or SAMPLES_DIR
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    return files


def image_id_for(path: Path) -> str:
    return path.stem


def load_done_keys(jsonl_path: Path, *, retry_failures: bool = True) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_failures and not row.get("success"):
                continue
            key = f"{row.get('image_id')}|{row.get('model_key')}|{row.get('rep')}"
            done.add(key)
    return done


def prune_failures(jsonl_path: Path) -> int:
    """Remove unsuccessful rows so they can be re-run. Returns removed count."""
    if not jsonl_path.exists():
        return 0
    rows = []
    removed = 0
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("success"):
                rows.append(row)
            else:
                removed += 1
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return removed


def append_result(jsonl_path: Path, result_dict: dict) -> None:
    with jsonl_path.open("a") as f:
        f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")


def run_benchmark(
    *,
    models: list[str] | None = None,
    reps: int = 3,
    samples_dir: Path | None = None,
    out_name: str = "benchmark.jsonl",
    limit: int | None = None,
) -> Path:
    ensure_dirs()
    models = models or list(MODELS.keys())
    images = list_sample_images(samples_dir)
    if limit:
        images = images[:limit]

    out_path = RESULTS_DIR / out_name
    removed = prune_failures(out_path)
    if removed:
        print(f"Pruned {removed} failed rows for retry")
    done = load_done_keys(out_path)
    client = OpenRouterClient()

    total = len(images) * len(models) * reps
    completed = 0
    print(f"Benchmark: {len(images)} images × {len(models)} models × {reps} reps = {total}")
    print(f"Already done: {len(done)}  → writing to {out_path}")

    for img in images:
        iid = image_id_for(img)
        for model_key in models:
            for rep in range(1, reps + 1):
                key = f"{iid}|{model_key}|{rep}"
                completed += 1
                if key in done:
                    print(f"[{completed}/{total}] SKIP {key}")
                    continue
                print(f"[{completed}/{total}] RUN  {key} ...", flush=True)
                result = analyze_image(img, model_key, client=client, rep=rep)
                append_result(out_path, result.model_dump(mode="json"))
                status = "OK" if result.success else f"FAIL:{result.error}"
                cost = result.usage.cost_usd
                comps = len(result.analysis.components) if result.analysis else 0
                print(
                    f"         → {status}  {result.latency_ms:.0f}ms  "
                    f"${cost:.6f}  components={comps}"
                )

    print(f"Done. Results: {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="FoodCheat vision benchmark")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--models", nargs="+", default=None, help="Model keys: qwen glm luna")
    p.add_argument("--limit", type=int, default=None, help="Limit number of images")
    p.add_argument("--out", default="benchmark.jsonl")
    p.add_argument("--smoke", action="store_true", help="1 image × all models × 1 rep")
    args = p.parse_args(argv)

    if args.smoke:
        run_benchmark(models=args.models, reps=1, out_name="smoke.jsonl", limit=1)
    else:
        run_benchmark(
            models=args.models,
            reps=args.reps,
            out_name=args.out,
            limit=args.limit,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
