"""FoodCheat Streamlit UI — upload image, pick model, view multi-component calories."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from foodcheat.analyzer import analyze_image
from foodcheat.config import MODELS, MODEL_ORDER, SAMPLES_DIR
from foodcheat.router import route_analyze

st.set_page_config(
    page_title="FoodCheat Vision",
    page_icon="🍽️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; max-width: 900px; }
      .hero-sub { color: #94a3b8; font-size: 1.02rem; margin-bottom: 1.25rem; }
      .kcal-big { font-size: 2.8rem; font-weight: 700; color: #2dd4bf; line-height: 1.1; }
      .metric-label { color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem; }
      .metric-value { font-size: 1.35rem; font-weight: 650; color: #f1f5f9; }
      .metric-hint { color: #64748b; font-size: 0.78rem; margin-top: 0.15rem; }
      .muted { color: #94a3b8; font-size: 0.9rem; }
      .comp-card {
        background: #1e293b; border: 1px solid #334155; border-radius: 12px;
        padding: 1rem 1.1rem; margin-bottom: 0.85rem; color: #e2e8f0;
      }
      .comp-card strong { color: #f8fafc; font-size: 1.08rem; }
      .comp-kcal { color: #2dd4bf; font-weight: 700; font-size: 1.05rem; }
      .help-box {
        background: #0f172a; border: 1px solid #334155; border-left: 4px solid #2dd4bf;
        border-radius: 0 10px 10px 0; padding: 0.85rem 1.1rem;
        margin: 0.75rem 0 1.1rem; color: #cbd5e1; font-size: 0.95rem; line-height: 1.55;
      }
      .stat-box {
        background: #1e293b; border: 1px solid #334155; border-radius: 12px;
        padding: 0.9rem 1rem; height: 100%;
      }
      .tag {
        display: inline-block; background: #334155; color: #e2e8f0;
        border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.78rem; margin-right: 0.35rem;
      }
      .img-wrap img { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

FRONTEND_MAPPING_MD = """
### How to wire this JSON in the frontend

**Golden rule:**  
Total calories = sum of each `components[].calories.value`  
Do **not** add `ingredients` into the total — they only describe what is *inside* one component.

| UI element | JSON key | Notes |
|---|---|---|
| Dish title | `meal_summary.dish_name` | e.g. "Chicken Curry" |
| Cuisine / meal type | `meal_summary.cuisine`, `meal_summary.meal_type` | Subtitle |
| Big calorie number | `totals.calories` | Already summed by backend |
| Calorie range | `totals.min`, `totals.max` | Uncertainty band |
| Macros | `totals.protein_g`, `carbs_g`, `fat_g` | Meal totals |
| Confidence meter | `overall_confidence` | 0–1 (1 = very sure) |
| Show Edit / Confirm | `needs_user_confirmation` | `true` → ask user to review |
| Photo quality | `image_quality.usable`, `issues` | Retake if not usable |
| Food cards list | `components[]` | One card per plate item |
| Card title | `components[i].name` | Human-readable |
| Stable id | `components[i].canonical_name` | snake_case for DB |
| Category chip | `components[i].category` | curry / bread / garnish… |
| Portion | `components[i].portion.estimate_g` | Also `household_measure` |
| Card calories | `components[i].calories.value` | Row total |
| Piece count | `components[i].count` | e.g. 2; may be `null` |
| Recipe chips | `components[i].ingredients[]` | **Not** separate calorie lines |

**Component vs ingredient:**
- **Component** = separate item on the plate (chicken, gravy, chili).
- **Ingredient** = what is inside that item (oil in gravy). Do not sum again.
"""


def _save_upload(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()
    return Path(tmp.name)


def _fmt_latency(latency_ms: float) -> tuple[str, str]:
    """Return (main label, hint). Always clear: seconds + milliseconds."""
    sec = latency_ms / 1000.0
    main = f"{sec:.1f} s"
    hint = f"{latency_ms:.0f} milliseconds (how long the AI took)"
    return main, hint


def _render_result(result) -> None:
    if not result.success or not result.analysis:
        st.error(f"Analysis failed: {result.error}")
        if result.raw_content:
            with st.expander("Raw model output"):
                st.code(result.raw_content[:4000])
        return

    a = result.analysis
    u = result.usage
    lat_main, lat_hint = _fmt_latency(result.latency_ms)

    st.markdown("---")
    st.markdown("## Analysis result")

    st.markdown(
        """
        <div class="help-box">
          <b>How to read this (simple):</b><br/>
          1. <b>Total kcal</b> = all food items below added together.<br/>
          2. Each <b>component</b> = one separate food on the plate (chicken, gravy, chili…).<br/>
          3. <b>Ingredients</b> under a card = recipe inside that item only (do not add again).<br/>
          4. <b>Latency</b> = AI response time in <b>seconds</b> (also shown in milliseconds).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dish name first — clear hierarchy under the photo
    ms = a.meal_summary
    st.markdown(f"### {ms.dish_name}")
    st.caption(f"{ms.cuisine or 'Cuisine n/a'} · {ms.meal_type}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="kcal-big">{(a.totals.calories if a.totals else "—")}</div>'
            f'<div class="metric-label">Total kcal</div>'
            f'<div class="metric-hint">sum of all components</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="metric-value">{a.overall_confidence:.2f}</div>'
            f'<div class="metric-label">Confidence</div>'
            f'<div class="metric-hint">0 = unsure · 1 = very sure</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="metric-value">{lat_main}</div>'
            f'<div class="metric-label">Latency (seconds)</div>'
            f'<div class="metric-hint">{lat_hint}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="stat-box">'
            f'<div class="metric-value">${u.cost_usd:.5f}</div>'
            f'<div class="metric-label">API cost (USD)</div>'
            f'<div class="metric-hint">OpenRouter price for this scan</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if a.totals:
        st.markdown("")
        st.markdown(
            f"**Calorie range:** {a.totals.min:.0f} – {a.totals.max:.0f} kcal  ·  "
            f"**Protein** {a.totals.protein_g:.0f}g  ·  "
            f"**Carbs** {a.totals.carbs_g:.0f}g  ·  "
            f"**Fat** {a.totals.fat_g:.0f}g"
        )
    st.caption(
        f"Model: `{result.model_id}` · "
        f"tokens in / out / reasoning: {u.prompt_tokens} / {u.completion_tokens} / {u.reasoning_tokens}"
    )

    if a.needs_user_confirmation:
        st.warning("Low confidence — ask the user to confirm or edit portions.")

    if not a.image_quality.usable:
        st.error("Image may not be usable for accurate analysis.")
    elif a.image_quality.issues:
        st.info("Image notes: " + "; ".join(a.image_quality.issues))

    st.markdown(f"#### Components — {len(a.components)} items on the plate")
    for idx, c in enumerate(a.components, 1):
        cal = c.calories.value if c.calories else "—"
        title = c.name or c.canonical_name or f"Component {idx}"
        cat = c.category if c.category and c.category != "unknown" else "food"
        count_bit = f" · {c.count} pcs" if c.count else ""
        st.markdown(
            f"""
            <div class="comp-card">
              <strong>{idx}. {title}</strong>
              <span class="tag">{cat}</span>
              <span class="muted">confidence {c.confidence:.2f}{count_bit}</span><br/><br/>
              Portion: <b>{c.portion.estimate_g:.0f} g</b>
              <span class="muted">({c.portion.household_measure or c.portion.method})</span>
              &nbsp;·&nbsp; {c.nutrition_per_100g.kcal:.0f} kcal / 100g
              &nbsp;→&nbsp; <span class="comp-kcal">{cal} kcal</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if c.ingredients:
            ing = ", ".join(
                f"{i.name} ({i.proportion:.0%})" if i.proportion else i.name
                for i in c.ingredients[:8]
            )
            st.caption(f"Recipe inside this item (not extra calories): {ing}")

    if a.warnings:
        with st.expander("Warnings"):
            for w in a.warnings:
                st.write(f"- {w}")

    with st.expander("Frontend key mapping"):
        st.markdown(FRONTEND_MAPPING_MD)

    with st.expander("Raw JSON (copy for frontend)"):
        payload = json.loads(a.model_dump_json())
        st.json(payload)
        st.download_button(
            "Download JSON",
            data=json.dumps(payload, indent=2),
            file_name="food_analysis.json",
            mime="application/json",
            key=f"dl_{result.run_id}",
        )


def main() -> None:
    st.title("FoodCheat Vision")
    st.markdown(
        '<p class="hero-sub">'
        "Upload a meal photo → detect each food item separately → estimate portion &amp; calories "
        "→ sum for the meal total. Pick a model in the sidebar, then click <b>Analyze</b>."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Mode",
            ["Single model", "Compare all four", "Smart router (cheap → premium)"],
            help="Single = one model. Compare = run all four. Router = cheap first, escalate if unsure.",
        )
        model_key = st.selectbox(
            "Model",
            options=list(MODELS.keys()),
            format_func=lambda k: f"{MODELS[k].label} ({MODELS[k].id})",
            disabled=mode != "Single model",
        )
        st.markdown("---")
        st.subheader("Test image")
        samples = sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg"))
        sample_names = ["— use upload below —"] + [p.name for p in samples]
        sample_choice = st.selectbox("Sample from folder", sample_names)

        with st.expander("Frontend mapping cheat-sheet"):
            st.markdown(
                """
| Screen | Key |
|---|---|
| Title | `meal_summary.dish_name` |
| Total kcal | `totals.calories` |
| Food list | `components[]` |
| Row kcal | `components[i].calories.value` |
| Confirm CTA | `needs_user_confirmation` |
                """
            )

    uploaded = st.file_uploader(
        "Upload a meal photo",
        type=["png", "jpg", "jpeg", "webp"],
        help="Clear, top-down photos work best.",
    )

    # ----- Image on top -----
    image_path: Path | None = None
    st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
    img_col, _ = st.columns([1.1, 0.9])
    with img_col:
        if uploaded:
            image_path = _save_upload(uploaded)
            st.image(uploaded, caption="Input photo", use_container_width=True)
        elif sample_choice != "— use upload below —":
            image_path = SAMPLES_DIR / sample_choice
            st.image(str(image_path), caption=sample_choice, use_container_width=True)
        else:
            st.info("Upload a photo or pick a sample from the sidebar.")
    st.markdown("</div>", unsafe_allow_html=True)

    run = st.button(
        "Analyze",
        type="primary",
        disabled=image_path is None,
        use_container_width=True,
    )

    # ----- Full description automatically below the image -----
    if run and image_path is not None:
        if mode == "Single model":
            with st.spinner(f"Analyzing with {MODELS[model_key].label}…"):
                result = analyze_image(image_path, model_key)
            st.session_state["last_result"] = ("single", result, None)
        elif mode == "Compare all four":
            results = {}
            for k in MODEL_ORDER:
                with st.spinner(f"Running {MODELS[k].label}…"):
                    results[k] = analyze_image(image_path, k)
            st.session_state["last_result"] = ("compare", results, None)
        else:
            with st.spinner("Routing cheap → premium…"):
                routed = route_analyze(image_path)
            st.session_state["last_result"] = ("router", routed, None)

    # Persist / show last analysis under the image
    last = st.session_state.get("last_result")

    # ── Guard: discard stale compare results if MODEL_ORDER has changed ──
    if last:
        kind, payload, _ = last
        if kind == "compare" and isinstance(payload, dict):
            if set(payload.keys()) != set(MODEL_ORDER):
                # Old run had different models — clear it so user re-analyzes
                st.session_state.pop("last_result", None)
                last = None
                st.warning(
                    "⚠️ Model list changed (GPT-4o Mini added). "
                    "Please click **Analyze** again to run all four models."
                )

    if last:
        kind, payload, _ = last
        if kind == "single":
            _render_result(payload)
        elif kind == "compare":
            st.markdown("---")
            st.markdown("## Analysis result (compare)")
            tabs = st.tabs([MODELS[k].label for k in MODEL_ORDER])
            for tab, k in zip(tabs, MODEL_ORDER):
                with tab:
                    _render_result(payload.get(k))
            st.markdown("#### Cost / latency comparison")
            rows = []
            for k in MODEL_ORDER:
                r = payload.get(k)
                if r is None:
                    continue
                sec = r.latency_ms / 1000.0
                rows.append(
                    {
                        "Model": MODELS[k].label,
                        "Success": r.success,
                        "kcal": (
                            r.analysis.totals.calories
                            if r.analysis and r.analysis.totals
                            else None
                        ),
                        "Components": (
                            len(r.analysis.components) if r.analysis else 0
                        ),
                        "Latency (seconds)": round(sec, 2),
                        "Latency (ms)": round(r.latency_ms),
                        "Cost USD": round(r.usage.cost_usd, 6),
                    }
                )
            st.dataframe(rows, use_container_width=True)
        elif kind == "router":
            routed = payload
            st.info(
                f"Escalated: {routed['escalated']} · reason: `{routed['reason']}` · "
                f"attempts: {len(routed['attempts'])}"
            )
            _render_result(routed["accepted"])


if __name__ == "__main__":
    main()
