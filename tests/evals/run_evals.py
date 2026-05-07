#!/usr/bin/env python3
"""
Eval runner CLI — standalone script for running evals and generating reports.

Usage:
    python tests/evals/run_evals.py --mode offline     # CI gate: pass/fail only
    python tests/evals/run_evals.py --mode live         # Manual: scored rubric report
    python tests/evals/run_evals.py --mode all          # Both in sequence
    python tests/evals/run_evals.py --mode offline --output  # Write Markdown report

Exit codes:
    0 — all offline tests passed (or live-only run with informational failures)
    1 — at least one offline test failed (CI gate failure)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pytest invocation
# ---------------------------------------------------------------------------

def _run_pytest(marker: str) -> dict[str, Any]:
    """
    Invoke pytest with the given marker expression, capture JSON report.

    Returns the parsed JSON report dict, or an error dict if pytest
    itself fails to produce a report.
    """
    json_report_path = _RESULTS_DIR / f".tmp_{marker}_report.json"

    cmd = [
        sys.executable, "-m", "pytest",
        f"-m", marker,
        "-v",
        "--tb=short",
        f"--json-report",
        f"--json-report-file={json_report_path}",
        # Suppress live log clutter in CI
        "--no-header",
        "-q",
    ]

    print(f"\n{'='*70}")
    print(f"  Running: pytest -m {marker}")
    print(f"{'='*70}\n")

    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        capture_output=False,  # let pytest output stream to terminal
    )
    elapsed = time.monotonic() - t0

    # Parse JSON report
    report: dict[str, Any] = {
        "exit_code": result.returncode,
        "elapsed": elapsed,
        "marker": marker,
        "tests": [],
    }

    if json_report_path.exists():
        try:
            raw = json.loads(json_report_path.read_text(encoding="utf-8"))
            report["summary"] = raw.get("summary", {})
            report["tests"] = [
                {
                    "nodeid": t.get("nodeid", ""),
                    "outcome": t.get("outcome", "unknown"),
                    "duration": t.get("duration", 0.0),
                }
                for t in raw.get("tests", [])
            ]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  ⚠  Could not parse JSON report: {exc}")
        finally:
            json_report_path.unlink(missing_ok=True)
    else:
        print(f"  ⚠  No JSON report generated (is pytest-json-report installed?)")

    return report


# ---------------------------------------------------------------------------
# Summary table rendering
# ---------------------------------------------------------------------------

def _print_summary_table(report: dict[str, Any]) -> str:
    """Print a test-result summary table to stdout and return it as a string."""
    lines: list[str] = []

    marker = report["marker"]
    lines.append("")
    lines.append(f"{'─'*80}")
    lines.append(f"  {marker.upper()} EVAL RESULTS")
    lines.append(f"{'─'*80}")
    lines.append(f"  {'Test ID':<60} {'Result':<8} {'Duration':>8}")
    lines.append(f"  {'─'*60} {'─'*8} {'─'*8}")

    for t in report["tests"]:
        nodeid = t["nodeid"]
        # Shorten: strip leading tests/evals/ prefix
        short_id = nodeid.replace("tests/evals/", "")
        outcome = t["outcome"].upper()
        duration_ms = t["duration"] * 1000
        lines.append(f"  {short_id:<60} {outcome:<8} {duration_ms:>6.0f}ms")

    # Footer
    summary = report.get("summary", {})
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    errors = summary.get("error", 0)
    elapsed = report.get("elapsed", 0.0)

    lines.append(f"  {'─'*78}")
    lines.append(
        f"  {passed} passed, {failed} failed, {skipped} skipped, "
        f"{errors} errors in {elapsed:.1f}s"
    )
    lines.append(f"{'─'*80}")

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Live rubric table (reads JSONL produced by test_live_quality.py)
# ---------------------------------------------------------------------------

def _get_latest_jsonl() -> Path | None:
    """Find the most recent live_*.jsonl file in the results directory."""
    candidates = sorted(_RESULTS_DIR.glob("live_*.jsonl"), reverse=True)
    return candidates[0] if candidates else None


def _print_live_rubric() -> str:
    """Print the scored rubric from the latest JSONL file. Returns text."""
    jsonl = _get_latest_jsonl()
    if not jsonl:
        msg = "\n  ℹ  No live JSONL results found. Run live evals first.\n"
        print(msg)
        return msg

    results: list[dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").strip().splitlines():
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not results:
        msg = f"\n  ℹ  JSONL file {jsonl.name} is empty.\n"
        print(msg)
        return msg

    lines: list[str] = []
    lines.append("")
    lines.append(f"{'='*90}")
    lines.append(f"  LIVE EVAL SCORED RUBRIC  (source: {jsonl.name})")
    lines.append(f"{'='*90}")
    lines.append(
        f"  {'Scenario':<28} {'Check':<34} {'Status':<8} {'Score':<7} Detail"
    )
    lines.append(f"  {'─'*86}")

    blocking_checks = {
        "schema_valid", "no_blocked_terms", "no_injected_content_in_output",
    }
    passed_count = 0
    blocking_failures = 0

    for r in results:
        if r.get("passed"):
            status = "PASS"
            passed_count += 1
        elif r.get("check") == "injection_detected":
            status = "WARN"
            passed_count += 1
        else:
            status = "FAIL"
            if r.get("check") in blocking_checks:
                blocking_failures += 1

        detail = r.get("detail", "")
        detail_trunc = detail[:38] + "…" if len(detail) > 38 else detail
        lines.append(
            f"  {r.get('scenario', ''):<28} "
            f"{r.get('check', ''):<34} "
            f"{status:<8} "
            f"{r.get('score', 0.0):<7.2f} "
            f"{detail_trunc}"
        )

    lines.append(f"  {'─'*86}")
    lines.append(
        f"  Overall: {passed_count}/{len(results)} checks passed, "
        f"{blocking_failures} blocking failure(s)"
    )
    lines.append(f"{'='*90}")

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _write_markdown_report(sections: list[str]) -> Path:
    """Write a Markdown report to the results directory."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = _RESULTS_DIR / f"report_{ts}.md"

    header = (
        f"# Eval Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"Generated by `tests/evals/run_evals.py`\n\n"
    )

    report_path.write_text(
        header + "\n\n".join(f"```\n{s}\n```" for s in sections),
        encoding="utf-8",
    )

    print(f"\n  📄 Report written to: {report_path}\n")
    return report_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run eval suite for the multi-agent finance assistant.",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "live", "all"],
        default="offline",
        help="Which eval tier to run (default: offline).",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        default=None,
        help="Write Markdown report to this path (optional).",
    )
    args = parser.parse_args()

    sections: list[str] = []
    offline_failed = False

    # ---- Offline ----
    if args.mode in ("offline", "all"):
        report = _run_pytest("offline")
        section = _print_summary_table(report)
        sections.append(section)

        if report["exit_code"] != 0:
            offline_failed = True

    # ---- Live ----
    if args.mode in ("live", "all"):
        report = _run_pytest("live")
        section = _print_summary_table(report)
        sections.append(section)

        # Additionally print the scored rubric from JSONL
        rubric = _print_live_rubric()
        sections.append(rubric)

    # ---- Markdown output ----
    if args.output and sections:
        _write_markdown_report(sections, output_path=args.output)

    # ---- Exit code ----
    # Only fail on offline test failures (CI gate).
    # Live failures are informational — exit 0.
    if offline_failed:
        print("\n  ❌ OFFLINE EVAL GATE FAILED\n")
        return 1

    if args.mode in ("offline", "all"):
        print("\n  ✅ OFFLINE EVAL GATE PASSED\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
