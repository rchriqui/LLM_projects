"""
Evaluation script for the Camp Registration Assistant.

Runs a set of test cases against a fresh CampAssistant instance and checks
that each response contains (or does not contain) expected keywords.

Usage:
    uv run python eval.py
"""

import json
import shutil
import sys
from pathlib import Path

from agent import CampAssistant

EVALS_PATH = Path(__file__).parent / "evals.json"
DB_PATH = Path(__file__).parent / "mock_db.json"
DB_BACKUP_PATH = Path(__file__).parent / "mock_db.backup.json"


def _backup_db() -> None:
    shutil.copy(DB_PATH, DB_BACKUP_PATH)


def _restore_db() -> None:
    if DB_BACKUP_PATH.exists():
        shutil.copy(DB_BACKUP_PATH, DB_PATH)
        DB_BACKUP_PATH.unlink()


def run_case(case: dict) -> tuple[bool, str]:
    """Run a single eval case. Returns (passed, detail_message)."""
    agent = CampAssistant()
    try:
        response = agent.chat(case["input"]).lower()
    except Exception as exc:
        detail = (
            f"  Input   : {case['input']}\n"
            f"  Failures: agent raised exception: {type(exc).__name__}: {exc}"
        )
        return False, detail

    failures = []

    for kw in case.get("expect_contains", []):
        if kw.lower() not in response:
            failures.append(f"missing '{kw}'")

    for kw in case.get("expect_not_contains", []):
        if kw.lower() in response:
            failures.append(f"should NOT contain '{kw}'")

    if failures:
        detail = (
            f"  Input   : {case['input']}\n"
            f"  Failures: {', '.join(failures)}\n"
            f"  Response: {response[:300]}{'...' if len(response) > 300 else ''}"
        )
        return False, detail

    return True, ""


def main() -> None:
    with open(EVALS_PATH) as f:
        cases = json.load(f)

    print(f"Running {len(cases)} eval cases...\n")

    # Back up the DB so write-op tests don't corrupt it for the next run.
    _backup_db()

    passed, failed = 0, 0
    try:
        for case in cases:
            ok, detail = run_case(case)
            status = "PASS" if ok else "FAIL"
            desc = case.get("description", case["id"])
            print(f"[{status}] {case['id']}: {desc}")
            if not ok:
                print(detail)
            if ok:
                passed += 1
            else:
                failed += 1
    finally:
        _restore_db()

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print(" ✓")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
