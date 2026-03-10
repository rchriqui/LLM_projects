import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Create a .env file with this key.")


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

MODEL_NAME = "google/gemini-3.1-flash-lite-preview"

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

Extract all signals present in the conversation, and provide a confidence score
for each signal based on your own uncertainty.

Allowed categories and values:

intent: trip_planning, room_booking, dining_booking
value: suite_preference, high_budget, large_group, vip_expectation
sentiment: positive_experience, negative_experience
life_event: birthday, anniversary, honeymoon, promotion, celebration
competitive: competitor_wynn, competitor_cosmo, competitor_bellagio, competitor_offer

Return JSON only in this format:
{
  "signals": [
    {
      "category": "intent",
      "value": "trip_planning",
      "confidence": 0.92
    }
  ]
}

Where:
- confidence is a number between 0 and 1 (inclusive)
- use higher values when you are very sure, lower values when you are unsure

If no signal is present, return:
{"signals": []}
""".strip()


def normalize_signals(signals: Any) -> List[Dict[str, Any]]:
    if not isinstance(signals, list):
        return []

    # Aggregate by (category, value) and keep the highest confidence
    norm: Dict[tuple, float] = {}
    for item in signals:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        value = item.get("value")
        if category in ALLOWED_LABELS and value in ALLOWED_LABELS[category]:
            raw_conf = item.get("confidence")
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = None

            # Clamp confidence to [0, 1] if present
            if conf is not None:
                conf = max(0.0, min(1.0, conf))

            key = (category, value)
            if key not in norm:
                norm[key] = conf if conf is not None else -1.0
            else:
                # keep max confidence
                existing = norm[key]
                if conf is not None and conf > existing:
                    norm[key] = conf

    result: List[Dict[str, Any]] = []
    for (c, v), conf in sorted(norm.items()):
        if conf < 0:
            result.append({"category": c, "value": v})
        else:
            result.append({"category": c, "value": v, "confidence": round(conf, 3)})

    return result


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {"signals": []}


def build_prompt_from_text(user_text: str) -> str:
    """
    user_text can be a single message or multiple lines
    representing a conversation. We keep it simple and
    pass it as bullets as in the notebook.
    """
    lines = [line.strip() for line in user_text.splitlines() if line.strip()]
    if not lines:
        return "Conversation:\n- \n\nReturn the JSON now."
    joined = "\n".join(f"- {line}" for line in lines)
    return f"Conversation:\n{joined}\n\nReturn the JSON now."


def extract_signals(conversation_text: str) -> Dict[str, Any]:
    if not conversation_text.strip():
        return {
        "signals": [],
        "raw_text": "",
        "error": "Conversation is empty.",
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt_from_text(conversation_text)},
        ],
        extra_headers={
            "HTTP-Referer": "https://your-app.example",
            "X-Title": "vip-casino-signal-prototype",
        },
    )

    text = response.choices[0].message.content
    parsed = safe_parse_json(text)
    normalized = normalize_signals(parsed.get("signals", []))

    return {
        "signals": normalized,
        "raw_text": text,
        "error": "",
    }


def gradio_pipeline(conversation_text: str):
    result = extract_signals(conversation_text)
    signals = result["signals"]

    if result["error"]:
        status = result["error"]
    elif not signals:
        status = "No valid signals detected."
    else:
        # Compute an average confidence if available
        confidences = [
            s.get("confidence")
            for s in signals
            if isinstance(s, dict) and isinstance(s.get("confidence"), (int, float))
        ]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            status = f"{len(signals)} signal(s) detected. Avg confidence: {avg_conf:.2f}"
        else:
            status = f"{len(signals)} signal(s) detected."

    return signals, result["raw_text"], status


with gr.Blocks(
    title="VIP Casino Signal Extractor (Gemini prototype)",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # VIP Casino Signal Extractor

        Prototype for the take‑home assignment: given a VIP casino guest conversation,
        extract structured signals using **google/gemini-3.1-flash-lite-preview** via OpenRouter.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Conversation")
            conversation_input = gr.Textbox(
                label="",
                placeholder=(
                    "Paste the conversation here.\n"
                    "Use one line per turn, for example:\n"
                    "Hi we're thinking about coming back in March.\n"
                    "Our host usually books us a suite when we visit."
                ),
                lines=10,
            )

            with gr.Accordion("Examples", open=False):
                gr.Markdown(
                    "- Trip planning: `Hi we're thinking about coming back in March.`  \n"
                    "- Suite / VIP expectation: `Our host usually books us a suite when we visit.`  \n"
                    "- Competitor: `We normally stay at Wynn when we're in Vegas.`"
                )

            run_button = gr.Button("✨ Extract signals", variant="primary")

        with gr.Column(scale=4):
            gr.Markdown("### Results")

            signals_output = gr.JSON(
                label="Extracted signals (normalized)",
            )

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
            )

            raw_output = gr.Textbox(
                label="Raw model output (JSON from Gemini)",
                lines=10,
            )

    run_button.click(
        fn=gradio_pipeline,
        inputs=conversation_input,
        outputs=[signals_output, raw_output, status_box],
    )


if __name__ == "__main__":
    demo.launch()


