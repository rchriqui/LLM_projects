"""
Signal Extraction Benchmark — Senior AI Engineer Take-Home Assignment

Benchmarks several frontier LLMs on structured signal extraction from
VIP casino guest conversations using micro-averaged Precision / Recall / F1.

Usage:
    python solution.py                        # run benchmark, print leaderboard
    python solution.py --gradio               # run benchmark + launch Gradio dashboard
    python solution.py --dataset path.jsonl   # use a custom dataset
"""

import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

COMPETITORS = [
    "google/gemini-3.1-flash-lite-preview",
    "xiaomi/mimo-v2-flash",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-4.1-fast",
]

ALLOWED_LABELS = {
    "intent": {"trip_planning", "room_booking", "dining_booking"},
    "value": {"suite_preference", "high_budget", "large_group", "vip_expectation"},
    "sentiment": {"positive_experience", "negative_experience"},
    "life_event": {"birthday", "anniversary", "honeymoon", "promotion", "celebration"},
    "competitive": {
        "competitor_wynn",
        "competitor_cosmo",
        "competitor_bellagio",
        "competitor_offer",
    },
}

SYSTEM_PROMPT = """
You are an information extraction system for VIP casino guest conversations.

Extract all signals present in the conversation.

Allowed categories and values:

intent: trip_planning, room_booking, dining_booking
value: suite_preference, high_budget, large_group, vip_expectation
sentiment: positive_experience, negative_experience
life_event: birthday, anniversary, honeymoon, promotion, celebration
competitive: competitor_wynn, competitor_cosmo, competitor_bellagio, competitor_offer

Return JSON only in this format:
{
  "signals": [
    {"category": "intent", "value": "trip_planning"}
  ]
}

If no signal is present, return:
{"signals": []}
""".strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def safe_parse_json(text: str) -> dict:
    """Parse model output leniently, falling back to the first { ... } block."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"signals": []}


def normalize_signals(signals: list) -> set[tuple]:
    """Return a set of validated (category, value) tuples."""
    out = set()
    if not isinstance(signals, list):
        return out
    for s in signals:
        if not isinstance(s, dict):
            continue
        c, v = s.get("category"), s.get("value")
        if c in ALLOWED_LABELS and v in ALLOWED_LABELS[c]:
            out.add((c, v))
    return out


def build_prompt(conversation: list[str]) -> str:
    turns = "\n".join(f"- {t}" for t in conversation)
    return f"Conversation:\n{turns}\n\nReturn the JSON now."


def call_model(model_name: str, conversation: list[str]) -> dict:
    """Call one model via OpenRouter and return normalized signals + raw text."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(conversation)},
        ],
    )
    text = response.choices[0].message.content or ""
    parsed = safe_parse_json(text)
    return {
        "raw_text": text,
        "signals": [
            {"category": c, "value": v}
            for c, v in sorted(normalize_signals(parsed.get("signals", [])))
        ],
    }


def compute_prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def score_predictions(gold_rows: list[dict], pred_rows: list[dict]) -> dict:
    """Compute micro-averaged Precision / Recall / F1 across the full dataset."""
    gold_by_id = {x["id"]: x for x in gold_rows}
    pred_by_id = {x["id"]: x for x in pred_rows}
    tp = fp = fn = 0
    for row_id, gold in gold_by_id.items():
        pred = pred_by_id.get(row_id, {"signals": []})
        gold_set = normalize_signals(gold["signals"])
        pred_set = normalize_signals(pred["signals"])
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    return compute_prf(tp, fp, fn)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(dataset_path: str = "dataset_vip.jsonl") -> dict:
    """
    Evaluate all models in COMPETITORS on the given dataset.

    Returns a results dict with a ranked 'results' list and per-model 'metrics',
    plus a 'model_results' dict containing raw predictions for each model.
    """
    dataset = load_jsonl(dataset_path)
    leaderboard = []
    model_results = {}

    for model_name in COMPETITORS:
        print(f"Testing {model_name} ...")
        predictions = []

        for row in dataset:
            pred = call_model(model_name, row["conversation"])
            predictions.append({
                "id": row["id"],
                "signals": pred["signals"],
                "raw_text": pred["raw_text"],
            })

        scores = score_predictions(dataset, predictions)
        print(f"  → P={scores['precision']}  R={scores['recall']}  F1={scores['f1']}")

        model_results[model_name] = {"metrics": scores, "predictions": predictions}
        leaderboard.append({"model": model_name, **scores})

    leaderboard.sort(key=lambda x: x["f1"], reverse=True)

    results_dict = {
        "results": [str(COMPETITORS.index(r["model"]) + 1) for r in leaderboard],
        "metrics": {
            r["model"]: {k: r[k] for k in ("precision", "recall", "f1")}
            for r in leaderboard
        },
        "model_results": model_results,
    }

    print("\nFinal ranking:")
    for rank, row in enumerate(leaderboard, 1):
        print(f"  {rank}. {row['model']}  —  F1={row['f1']}")

    return results_dict


# ---------------------------------------------------------------------------
# Gradio dashboard (optional)
# ---------------------------------------------------------------------------


def launch_dashboard(results: dict) -> None:
    import gradio as gr
    import pandas as pd

    metrics = results.get("metrics", {})
    model_results = results.get("model_results", {})

    def _color(v: float) -> str:
        return "#22c55e" if v >= 0.9 else "#f97316" if v >= 0.8 else "#ef4444"

    def _card(label: str, value: float) -> str:
        return (
            f"<div style='padding:10px 12px;border-radius:10px;background:#111827;"
            f"border:1px solid #1f2937;box-shadow:0 1px 3px rgba(0,0,0,.25);'>"
            f"<div style='font-size:11px;color:#9ca3af;margin-bottom:4px;'>{label}</div>"
            f"<div style='font-size:20px;font-weight:600;color:{_color(value)};'>{value:.4f}</div>"
            f"</div>"
        )

    def build_overview():
        rows = []
        for model_name, m in metrics.items():
            rows.append({
                "Model": model_name,
                "Short Model": model_name.split("/")[-1],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
            })
        df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)

        best = df.iloc[0]
        summary = (
            f"<div style='padding:8px 0 4px 0;'>"
            f"<div style='margin-bottom:16px;'>"
            f"<div style='font-size:13px;color:#9ca3af;'>Best Model</div>"
            f"<div style='font-size:22px;font-weight:700;'>{best['Model']}</div>"
            f"</div>"
            f"<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:10px;'>"
            f"{_card('Best F1', float(best['F1']))}"
            f"{_card('Avg F1', float(df['F1'].mean()))}"
            f"{_card('Avg Precision', float(df['Precision'].mean()))}"
            f"{_card('Avg Recall', float(df['Recall'].mean()))}"
            f"</div>"
            f"<div style='margin-top:10px;font-size:10px;color:#6b7280;'>"
            f"green ≥ 0.9 · orange ≥ 0.8 · red &lt; 0.8</div></div>"
        )
        cols = ["Rank", "Short Model", "Model", "Precision", "Recall", "F1"]
        return (
            summary,
            df[["Short Model", "F1"]],
            df[cols],
            gr.update(choices=list(df["Short Model"]), value=df["Short Model"].iloc[0]),
        )

    def show_detail(short: str):
        empty = pd.DataFrame({"id": [], "signals": []})
        short_to_full = {n.split("/")[-1]: n for n in metrics}
        full = short_to_full.get(short)
        if not full:
            return "<div style='color:#ef4444;'>Model not found.</div>", empty
        m = metrics[full]
        detail = (
            f"<div style='padding:8px 0;'>"
            f"<div style='font-size:14px;color:#9ca3af;'>Model</div>"
            f"<div style='font-size:20px;font-weight:600;margin-bottom:12px;'>{full}</div>"
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;'>"
            f"{_card('Precision', m['precision'])}"
            f"{_card('Recall', m['recall'])}"
            f"{_card('F1', m['f1'])}"
            f"</div></div>"
        )
        preds = model_results.get(full, {}).get("predictions", [])
        df_preds = pd.DataFrame([{"id": p["id"], "signals": p["signals"]} for p in preds[:50]])
        return detail, df_preds

    with gr.Blocks(title="LLM Evaluation Dashboard") as app:
        gr.Markdown("## LLM Evaluation Dashboard")
        gr.Markdown("Results from the last benchmark run.")

        refresh_btn = gr.Button("Refresh results", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                overview_html = gr.HTML("<div style='color:#9ca3af;'>Click Refresh to load.</div>")
            with gr.Column(scale=1):
                bar_plot = gr.BarPlot(
                    value=None, x="Short Model", y="F1",
                    title="F1 score per model", y_lim=[0, 1], height=360,
                )

        gr.Markdown("### Leaderboard")
        models_table = gr.Dataframe(
            headers=["Rank", "Short Model", "Model", "Precision", "Recall", "F1"],
            row_count=(0, "dynamic"),
            wrap=True,
        )

        gr.Markdown("### Model detail")
        with gr.Row():
            model_dropdown = gr.Dropdown(label="Select model", choices=[], value=None)
            detail_btn = gr.Button("Show details")

        detail_html = gr.HTML("<div style='color:#9ca3af;'>Select a model above.</div>")
        preds_table = gr.Dataframe(headers=["id", "signals"], row_count=(0, "dynamic"), wrap=True)

        refresh_btn.click(fn=build_overview, inputs=None,
                          outputs=[overview_html, bar_plot, models_table, model_dropdown])
        detail_btn.click(fn=show_detail, inputs=model_dropdown,
                         outputs=[detail_html, preds_table])

    app.launch(
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="indigo", font=["Inter", "system-ui", "sans-serif"]),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="VIP Casino Signal Extraction Benchmark")
    parser.add_argument("--dataset", default="dataset_vip.jsonl", help="Path to the JSONL dataset")
    parser.add_argument("--gradio", action="store_true", help="Launch the Gradio dashboard after benchmarking")
    args = parser.parse_args()

    results = run_benchmark(dataset_path=args.dataset)

    print("\nResults JSON:")
    printable = {k: v for k, v in results.items() if k != "model_results"}
    print(json.dumps(printable, indent=2))

    if args.gradio:
        launch_dashboard(results)


if __name__ == "__main__":
    main()
