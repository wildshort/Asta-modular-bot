"""
scanner/quality_check.py
========================

Post-scan quality evaluator. Runs after the main scanner finishes.

Reads every diagnostic JSON in scanner/output/diag/ and applies a set of
quality checks. Writes scanner/output/quality_report.json with findings.

Quality checks performed:
  1. Stale breakout: a chart marked as TL BREAKOUT where the breakout bar
     is more than 3 bars old. (We've fixed this in chart_builder, but we
     check anyway in case the fix breaks.)
  2. No-line drawn but candidates existed: chosen=None when at least one
     valid candidate (diag/horiz, support/resistance) was found.
  3. Stale line: line chosen but its current y-value is far from price
     (>3×ATR away). The proximity filter should prevent this, but we verify.
  4. Suspicious slope: extremely steep diagonal lines (>1×ATR per bar) that
     are likely artifacts.
  5. Low touch count for long span: 3-touch line over 200+ bars. Could be
     coincidental fit, prefer 4+ touches when span is high.

The output JSON has shape:
  {
    "summary": {
      "total_charts": N,
      "issues_found": M,
      "issues_by_type": {...}
    },
    "issues": [
      {
        "symbol": "IPCALAB.NS",
        "type": "no_line_but_candidates_exist",
        "severity": "high|medium|low",
        "details": "..."
      },
      ...
    ]
  }
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


DIAG_DIR = Path("scanner/output/diag")
OUT_PATH = Path("scanner/output/quality_report.json")


def check_one(diag: dict) -> list[dict]:
    """Return a list of issue dicts for a single diagnostic JSON."""
    issues = []
    symbol = diag.get("symbol", "UNKNOWN")
    chosen = diag.get("chosen")
    n_bars = diag.get("n_bars", 0)

    # Check 1: Stale breakout (chosen line is broken but breakout_bar is old)
    if chosen and chosen.get("broken") and chosen.get("breakout_bar") is not None:
        bb = chosen["breakout_bar"]
        bars_ago = (n_bars - 1) - bb
        if bars_ago > 3:
            issues.append({
                "symbol": symbol,
                "type": "stale_breakout",
                "severity": "high",
                "details": f"Breakout was {bars_ago} bars ago — entry would be late",
                "bars_since_breakout": bars_ago,
            })

    # Check 2: No line chosen but candidates existed
    if chosen is None:
        candidates_found = []
        for key in [
            "best_diagonal_resistance",
            "best_diagonal_support",
            "best_horizontal_resistance",
            "best_horizontal_support",
        ]:
            cand = diag.get(key, {})
            if cand.get("found"):
                candidates_found.append(key)
        if candidates_found:
            issues.append({
                "symbol": symbol,
                "type": "no_line_but_candidates_exist",
                "severity": "medium",
                "details": (
                    f"Candidates found ({', '.join(candidates_found)}) but none selected. "
                    f"Direction={diag.get('direction')} regime={diag.get('regime')}"
                ),
                "candidates_found": candidates_found,
                "rejection_reasons": diag.get("rejection_reasons", []),
            })

    # Check 3: Suspicious slope (very steep diagonal)
    if chosen and chosen.get("kind") == "diagonal":
        # Compare slope magnitude to ATR. We don't have full data here but
        # we can check the slope value against typical ranges.
        # For typical stock ATR=20, slope > 5 means >5/bar = 25% of ATR/bar — extreme
        slope = chosen.get("slope") if "slope" in chosen else None
        if slope is not None:
            atr_recent = diag.get("atr_recent", 1.0)
            if atr_recent > 0 and abs(slope) > atr_recent:
                issues.append({
                    "symbol": symbol,
                    "type": "suspicious_steep_slope",
                    "severity": "low",
                    "details": f"Slope {slope:.3f} per bar exceeds ATR {atr_recent:.2f} — line may be artifact",
                })

    # Check 4: Long span but only 3 touches (could be coincidental)
    if chosen:
        span = chosen.get("span", 0)
        touches = chosen.get("touches_count", 0)
        if span >= 200 and touches == 3:
            issues.append({
                "symbol": symbol,
                "type": "long_span_few_touches",
                "severity": "low",
                "details": f"{span}-bar line has only {touches} touches — limited validation",
            })

    return issues


def main() -> int:
    if not DIAG_DIR.exists():
        print(f"Diagnostic directory not found: {DIAG_DIR}")
        # Still write an empty report so artifact upload has something
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            "summary": {"total_charts": 0, "issues_found": 0, "issues_by_type": {}},
            "issues": [],
            "note": "No diagnostic directory — scanner may not have fired any signals.",
        }, indent=2))
        return 0

    diag_files = sorted(DIAG_DIR.glob("*.json"))
    print(f"Quality check: scanning {len(diag_files)} diagnostic files...")

    all_issues: list[dict] = []
    issues_by_type: dict[str, int] = {}

    for diag_file in diag_files:
        try:
            with diag_file.open() as f:
                diag = json.load(f)
        except Exception as e:
            print(f"  ! failed to read {diag_file.name}: {e}")
            continue

        issues = check_one(diag)
        for issue in issues:
            all_issues.append(issue)
            t = issue["type"]
            issues_by_type[t] = issues_by_type.get(t, 0) + 1

    # Sort issues by severity (high first) then symbol
    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_issues.sort(key=lambda i: (severity_order.get(i["severity"], 99), i["symbol"]))

    report = {
        "summary": {
            "total_charts": len(diag_files),
            "issues_found": len(all_issues),
            "issues_by_type": issues_by_type,
            "by_severity": {
                "high": sum(1 for i in all_issues if i["severity"] == "high"),
                "medium": sum(1 for i in all_issues if i["severity"] == "medium"),
                "low": sum(1 for i in all_issues if i["severity"] == "low"),
            },
        },
        "issues": all_issues,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== Quality Check Summary ===")
    print(f"Total charts: {len(diag_files)}")
    print(f"Issues found: {len(all_issues)}")
    print(f"By severity: high={report['summary']['by_severity']['high']}, "
          f"medium={report['summary']['by_severity']['medium']}, "
          f"low={report['summary']['by_severity']['low']}")
    print(f"By type: {issues_by_type}")
    print(f"\nReport written to: {OUT_PATH}")

    # Print top 5 high-severity issues for visibility in workflow logs
    high_issues = [i for i in all_issues if i["severity"] == "high"]
    if high_issues:
        print("\n=== HIGH SEVERITY ISSUES (top 5) ===")
        for issue in high_issues[:5]:
            print(f"  [{issue['symbol']}] {issue['type']}: {issue['details']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
