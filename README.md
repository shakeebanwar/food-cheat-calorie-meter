# FoodCheat AI Vision R&D

Multi-component food calorie estimation benchmark via OpenRouter.

## Models under test

| Key | Model | Notes |
|-----|-------|-------|
| `qwen` | `qwen/qwen3-vl-32b-instruct` | Replacement for DeepSeek V4 Flash (text-only) |
| `glm` | `z-ai/glm-5.3-flash` | Cost-efficient vision flash |
| `luna` | `openai/gpt-5.6-luna` | Premium accuracy baseline |

Nutrition comes from the LLM; calorie **math** is deterministic in Python.

## Setup

```bash
uv sync
# .env must contain: OPENROUTER_API_KEY=...
```

## Run benchmark

```bash
uv run python run_benchmark.py --smoke          # 1 image × 3 models
uv run python run_benchmark.py --reps 3         # full run (resumable)
```

Results → `results/benchmark.jsonl`

## Streamlit UI

```bash
uv run streamlit run app_streamlit.py
```

## Package layout

```
foodcheat/
  analyzer.py      # image → FoodAnalysis
  calculator.py    # deterministic kcal
  client.py        # OpenRouter
  prompts.py       # multi-component anti-merge prompting
  schema.py        # frontend-ready JSON schema
  evaluation/      # benchmark, metrics, judge, report
app_streamlit.py
```
