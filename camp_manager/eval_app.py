"""
Evaluation Dashboard — Camp Registration Assistant

Two-tab Gradio app:
  Tab 1 — Run Evaluation: executes all 30 test cases live with progress updates.
  Tab 2 — Last Saved Results: loads the most recent run from eval_results.json.

Three evaluation modes:
  Keyword only  — fast, deterministic, checks presence/absence of expected strings.
  LLM judge     — GPT-4o-mini reads each response and rates correctness with a reason.
  Both          — runs both; flags disagreements with ⚠️.

Usage:
    uv run python eval_app.py
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
from agents import Agent, Runner
from pydantic import BaseModel

from agent import CampAssistant

EVALS_PATH = Path(__file__).parent / "evals.json"
DB_PATH = Path(__file__).parent / "mock_db.json"
DB_BACKUP_PATH = Path(__file__).parent / "mock_db.backup.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

JUDGE_MODES = ["Keyword only", "LLM judge", "Both"]
DEFAULT_MODE = "Keyword only"

# =============================================================================
# LLM judge
# =============================================================================


class _JudgeOutput(BaseModel):
    passed: bool
    reason: str


_judge_agent = Agent(
    name="Eval Judge",
    model="gpt-4o-mini",
    instructions="""You are an expert evaluator for a summer camp registration chatbot.

Given a test description (what correct behaviour looks like), a user input, and the
chatbot's response, decide whether the chatbot responded correctly.

Be strict but fair:
- The response must address what the test description requires.
- Minor wording differences are fine as long as the substance is correct.
- If the chatbot hedges, loops, or gives irrelevant information, mark it as failed.

Return:
  passed — true if the response satisfies the test description, false otherwise.
  reason — one concise sentence explaining your verdict.""",
    output_type=_JudgeOutput,
)


async def _judge_async(description: str, user_input: str, response: str) -> tuple[bool, str]:
    prompt = (
        f"Test description: {description}\n\n"
        f"User input: {user_input}\n\n"
        f"Chatbot response: {response}"
    )
    result = await Runner.run(_judge_agent, prompt)
    return result.final_output.passed, result.final_output.reason


def _llm_judge(description: str, user_input: str, response: str) -> tuple[bool, str]:
    try:
        return asyncio.run(_judge_async(description, user_input, response))
    except Exception as exc:
        return False, f"Judge error: {exc}"


# =============================================================================
# Core eval logic
# =============================================================================


def _load_cases() -> list[dict]:
    with open(EVALS_PATH) as f:
        return json.load(f)


def _backup_db() -> None:
    shutil.copy(DB_PATH, DB_BACKUP_PATH)


def _restore_db() -> None:
    if DB_BACKUP_PATH.exists():
        shutil.copy(DB_BACKUP_PATH, DB_PATH)
        DB_BACKUP_PATH.unlink()


def _keyword_check(case: dict, response: str) -> tuple[bool, list[str]]:
    response_lower = response.lower()
    failures = []
    for kw in case.get("expect_contains", []):
        if kw.lower() not in response_lower:
            failures.append(f"missing '{kw}'")
    for kw in case.get("expect_not_contains", []):
        if kw.lower() in response_lower:
            failures.append(f"should NOT contain '{kw}'")
    return not failures, failures


def _run_case(case: dict, judge_mode: str) -> dict:
    """Run one eval case and return a result dict that includes both verdicts."""
    agent = CampAssistant()
    try:
        response = agent.chat(case["input"])
        exception = None
    except Exception as exc:
        response = f"[EXCEPTION] {type(exc).__name__}: {exc}"
        exception = str(exc)

    keyword_passed: bool | None = None
    keyword_failures: list[str] = []
    llm_passed: bool | None = None
    llm_reason: str | None = None

    if exception:
        keyword_passed = False
        keyword_failures = [f"raised {exception}"]
    else:
        if judge_mode in ("Keyword only", "Both"):
            keyword_passed, keyword_failures = _keyword_check(case, response)
        if judge_mode in ("LLM judge", "Both"):
            llm_passed, llm_reason = _llm_judge(
                case.get("description", case["id"]), case["input"], response
            )

    # Primary verdict used for the summary score
    if judge_mode == "Keyword only":
        passed = keyword_passed
    elif judge_mode == "LLM judge":
        passed = llm_passed
    else:  # Both — pass only when both agree
        passed = bool(keyword_passed and llm_passed)

    return {
        "id": case["id"],
        "description": case.get("description", case["id"]),
        "input": case["input"],
        "response": response,
        "passed": passed,
        # Keyword verdict
        "keyword_passed": keyword_passed,
        "keyword_failures": keyword_failures,
        # LLM verdict
        "llm_passed": llm_passed,
        "llm_reason": llm_reason,
    }


# =============================================================================
# Display helpers
# =============================================================================

_COLS_KEYWORD = ["", "Test ID", "Description", "Input", "Failures", "Response"]
_COLS_LLM = ["", "Test ID", "Description", "Input", "LLM Reason", "Response"]
_COLS_BOTH = ["", "Keyword", "LLM", "Test ID", "Description", "Input", "Failures", "LLM Reason", "Response"]


def _cols_for(judge_mode: str) -> list[str]:
    if judge_mode == "LLM judge":
        return _COLS_LLM
    if judge_mode == "Both":
        return _COLS_BOTH
    return _COLS_KEYWORD


def _verdict(passed: bool | None) -> str:
    if passed is None:
        return "—"
    return "✅" if passed else "❌"


def _overall(r: dict) -> str:
    kp, lp = r["keyword_passed"], r["llm_passed"]
    if kp is None:
        return _verdict(lp)
    if lp is None:
        return _verdict(kp)
    if kp and lp:
        return "✅"
    if not kp and not lp:
        return "❌"
    return "⚠️"  # disagreement


def _to_df(results: list[dict], judge_mode: str) -> pd.DataFrame:
    cols = _cols_for(judge_mode)
    if not results:
        return pd.DataFrame(columns=cols)

    rows = []
    for r in results:
        row: dict = {
            "Test ID": r["id"],
            "Description": r["description"],
            "Input": r["input"],
        }
        resp = r["response"]
        row["Response"] = resp[:120] + "…" if len(resp) > 120 else resp

        if judge_mode == "Keyword only":
            row[""] = _verdict(r["keyword_passed"])
            row["Failures"] = ", ".join(r["keyword_failures"]) if r["keyword_failures"] else "—"
        elif judge_mode == "LLM judge":
            row[""] = _verdict(r["llm_passed"])
            row["LLM Reason"] = r["llm_reason"] or "—"
        else:  # Both
            row[""] = _overall(r)
            row["Keyword"] = _verdict(r["keyword_passed"])
            row["LLM"] = _verdict(r["llm_passed"])
            row["Failures"] = ", ".join(r["keyword_failures"]) if r["keyword_failures"] else "—"
            row["LLM Reason"] = r["llm_reason"] or "—"
        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


def _summary_md(passed: int, total: int, judge_mode: str, run_at: str = "") -> str:
    pct = passed / total * 100 if total else 0
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    color = "🟢" if pct >= 90 else ("🟡" if pct >= 70 else "🔴")
    mode_label = f" *(via {judge_mode})*"
    ts = f"\n\n*Last run: {run_at}*" if run_at else ""
    return f"## {color} {passed}/{total} passed — {pct:.0f}%{mode_label}\n`{bar}`{ts}"


def _save_results(results: list[dict], passed: int, total: int, judge_mode: str) -> None:
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "judge_mode": judge_mode,
                "passed": passed,
                "total": total,
                "score_pct": round(passed / total * 100, 1),
                "results": results,
            },
            f,
            indent=2,
        )


# =============================================================================
# Gradio handlers
# =============================================================================


def run_evals(judge_mode: str):
    """
    Generator consumed by Gradio — yields (progress, df, summary, state)
    after each completed test case so the table updates live.
    """
    cases = _load_cases()
    results: list[dict] = []
    _backup_db()

    try:
        for i, case in enumerate(cases, 1):
            yield (
                f"⏳ Running **{i} / {len(cases)}** — `{case['id']}`…",
                _to_df(results, judge_mode),
                "",
                results,
            )
            results.append(_run_case(case, judge_mode))

        passed = sum(r["passed"] for r in results)
        _save_results(results, passed, len(cases), judge_mode)
        run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        yield (
            f"✅ Finished — **{passed}/{len(cases)}** passed",
            _to_df(results, judge_mode),
            _summary_md(passed, len(cases), judge_mode, run_at),
            results,
        )
    finally:
        _restore_db()


def load_last_results() -> tuple[str, pd.DataFrame, list]:
    if not RESULTS_PATH.exists():
        return (
            "No saved results yet. Go to **▶ Run Evaluation** to run the test suite.",
            pd.DataFrame(columns=_COLS_KEYWORD),
            [],
        )
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    judge_mode = data.get("judge_mode", "Keyword only")
    return (
        _summary_md(data["passed"], data["total"], judge_mode, data["run_at"]),
        _to_df(data["results"], judge_mode),
        data["results"],
    )


def show_detail(state: list, evt: gr.SelectData) -> str:
    """Show full response + both verdicts for the clicked table row."""
    if not state or evt.index[0] >= len(state):
        return "*Click any row to see the full agent response.*"
    r = state[evt.index[0]]

    lines = [f"### `{r['id']}`\n", f"**Input:** {r['input']}\n", "---\n"]

    if r["keyword_passed"] is not None:
        kf = ", ".join(r["keyword_failures"]) if r["keyword_failures"] else "none"
        lines.append(f"**Keyword verdict:** {_verdict(r['keyword_passed'])}  (failures: {kf})\n")

    if r["llm_passed"] is not None:
        lines.append(
            f"**LLM judge verdict:** {_verdict(r['llm_passed'])}  — {r['llm_reason']}\n"
        )

    lines += ["---\n", f"**Agent response:**\n\n{r['response']}"]
    return "\n".join(lines)


# =============================================================================
# UI
# =============================================================================

with gr.Blocks(title="Camp Assistant — Eval Dashboard") as demo:
    gr.Markdown(
        "# 🏕️ Camp Assistant — Evaluation Dashboard\n"
        "30 automated test cases covering camp discovery, data integrity, "
        "age validation, schedule conflicts, guardrails, and conversational behaviour."
    )

    with gr.Tabs():

        # ── Tab 1: Run ─────────────────────────────────────────────────────
        with gr.Tab("▶  Run Evaluation"):
            with gr.Row():
                judge_radio = gr.Radio(
                    choices=JUDGE_MODES,
                    value=DEFAULT_MODE,
                    label="Evaluation method",
                    info=(
                        "Keyword — fast & deterministic. "
                        "LLM judge — robust to paraphrasing, costs extra API calls. "
                        "Both — runs both and flags ⚠️ where they disagree."
                    ),
                )
                run_btn = gr.Button("Run All 30 Tests", variant="primary", scale=0)

            run_progress = gr.Markdown("Choose a method and click **Run All 30 Tests**.")
            run_summary = gr.Markdown()
            run_df = gr.DataFrame(
                value=pd.DataFrame(columns=_COLS_KEYWORD),
                wrap=True,
                interactive=False,
            )
            run_detail = gr.Markdown("*Click any row to see the full agent response.*")
            run_state = gr.State([])

            run_btn.click(
                fn=run_evals,
                inputs=[judge_radio],
                outputs=[run_progress, run_df, run_summary, run_state],
            )
            run_df.select(fn=show_detail, inputs=[run_state], outputs=[run_detail])

        # ── Tab 2: Last Saved Results ───────────────────────────────────────
        with gr.Tab("📊  Last Saved Results"):
            refresh_btn = gr.Button("↻  Refresh", variant="secondary")
            last_summary = gr.Markdown()
            last_df = gr.DataFrame(
                value=pd.DataFrame(columns=_COLS_KEYWORD),
                wrap=True,
                interactive=False,
            )
            last_detail = gr.Markdown("*Click any row to see the full agent response.*")
            last_state = gr.State([])

            refresh_btn.click(
                fn=load_last_results,
                outputs=[last_summary, last_df, last_state],
            )
            last_df.select(fn=show_detail, inputs=[last_state], outputs=[last_detail])
            demo.load(fn=load_last_results, outputs=[last_summary, last_df, last_state])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
