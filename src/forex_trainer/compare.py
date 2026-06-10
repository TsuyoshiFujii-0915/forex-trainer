"""Comparison CLI: forex-compare <runs_root>.

Tabulates metrics.json across all evaluated runs, sorted by Sharpe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def collect_runs(runs_root: Path) -> list[dict[str, Any]]:
    """Collect metrics from all evaluated runs under a root directory.

    Args:
        runs_root: Directory laid out as <experiment>/<timestamp>/.

    Returns:
        One row per evaluated run, sorted by annualized Sharpe descending.
    """
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parent
        rows.append(
            {
                "experiment": run_dir.parent.name,
                "run": run_dir.name,
                **metrics,
            }
        )
    rows.sort(key=lambda row: row["sharpe_annualized"], reverse=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 when at least one evaluated run was found,
        1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="forex-compare", description="Compare evaluated runs by their metrics."
    )
    parser.add_argument(
        "runs_root",
        type=str,
        nargs="?",
        default="runs",
        help="Runs root directory (default: runs, the repo convention).",
    )
    args = parser.parse_args(argv)
    rows = collect_runs(Path(args.runs_root))
    if not rows:
        print(f"no evaluated runs found under {args.runs_root}", file=sys.stderr)
        return 1

    header = (
        f"{'experiment':<24} {'run':<22} {'cum_logret':>11} {'sharpe':>8} "
        f"{'max_dd':>7} {'cost_ratio':>10} {'steps':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['experiment']:<24} {row['run']:<22} "
            f"{row['cumulative_log_return']:>+11.5f} {row['sharpe_annualized']:>8.2f} "
            f"{row['max_drawdown']:>7.4f} {row['total_cost_ratio']:>10.5f} "
            f"{row['steps']:>6d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
