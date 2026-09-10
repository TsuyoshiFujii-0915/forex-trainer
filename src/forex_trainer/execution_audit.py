"""Read-only audit of sealed Issue #16 evaluation periods."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from forex_env.data.file_provider import FileDataProvider

from .artifact_provenance import git_commits, sha256_file
from .spread_decomposition import POLICIES, PORTFOLIO_ARTIFACTS, load_seal
from .supervised_portfolio import FOLDS


def audit_period(
    fold: str,
    raw_index: pd.DatetimeIndex,
    steps: pd.DataFrame,
    warmup: int,
    window_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit all three controls against the complete raw market calendar.

    Args:
        fold: Calendar evaluation year.
        raw_index: Actual provider labels, including its inclusive end date.
        steps: Sealed transitions for all three policies in this fold.
        warmup: Number of feature construction rows removed by the environment.
        window_size: Number of rows in each observation, including the decision.

    Returns:
        One period ledger row and every bar excluded as a decision by warmup.

    Raises:
        ValueError: If history or any control is incomplete or misaligned.
    """
    context = f"fold {fold}"
    first = warmup + window_size - 1
    if (
        warmup < 1 or window_size < 1 or len(raw_index) < first + 2
        or raw_index.tz is None or raw_index.hasnans
        or not raw_index.is_unique or not raw_index.is_monotonic_increasing
    ):
        raise ValueError(f"{context}: invalid or insufficient timezone-aware market history")
    if set(steps["policy"]) != set(POLICIES):
        raise ValueError(f"{context}: expected all three policies {POLICIES}")
    decisions = [value.isoformat() for value in raw_index[first:-1]]
    targets = [value.isoformat() for value in raw_index[first + 1:]]
    for policy in POLICIES:
        trace = steps.loc[steps["policy"] == policy]
        if (
            trace["decision_timestamp"].tolist() != decisions
            or trace["target_timestamp"].tolist() != targets
        ):
            raise ValueError(f"{context} {policy}: decision/target calendar mismatch")
    seconds = (raw_index[-1] - raw_index[first]).total_seconds()
    years = seconds / (365.25 * 86400)
    row: dict[str, Any] = {
        "fold": fold,
        "declared_start_inclusive": f"{fold}-01-01",
        "declared_end_inclusive": f"{int(fold) + 1}-01-01",
        "timezone": str(raw_index.tz),
        "raw_first": raw_index[0].isoformat(),
        "raw_last": raw_index[-1].isoformat(),
        "raw_bars": len(raw_index),
        "feature_warmup_bars": warmup,
        "observation_window_bars": window_size,
        "additional_predecision_bars": window_size - 1,
        "excluded_decision_bars": first,
        "first_observation_start": raw_index[warmup].isoformat(),
        "first_decision": decisions[0],
        "last_decision": decisions[-1],
        "first_target": targets[0],
        "last_target": targets[-1],
        "decision_count_per_policy": len(decisions),
        "pair_targets_per_policy": len(decisions) * 9,
        "equity_observations_per_policy": len(decisions) + 1,
        "targets_in_next_year": sum(t.year > int(fold) for t in raw_index[first + 1:]),
        "elapsed_seconds": seconds,
        "elapsed_days": seconds / 86400,
        "elapsed_years": years,
        "steps_per_year": len(decisions) / years,
    }
    excluded = pd.DataFrame([
        {"fold": fold, "raw_position": position, "timestamp": timestamp.isoformat(),
         "reason": "feature_warmup" if position < warmup else "observation_history"}
        for position, timestamp in enumerate(raw_index[:first])
    ])
    return row, excluded


def run_audit(
    source_dir: Path, source_provenance_sha256: str, data_path: Path, output_dir: Path,
) -> None:
    """Write a new period ledger from pinned artifacts without policy replay.

    Args:
        source_dir: Explicit Issue #16 artifact directory.
        source_provenance_sha256: Expected hash of its provenance.json.
        data_path: Local copy of the sealed historical carry cache.
        output_dir: New directory; existing outputs are never overwritten.

    Raises:
        FileExistsError: If the output directory already exists.
        ValueError: If a seal, calendar, config, or annualization disagrees.
    """
    if output_dir.exists():
        raise FileExistsError(f"Audit output already exists: {output_dir}")
    seal = load_seal(source_dir, source_provenance_sha256, PORTFOLIO_ARTIFACTS)
    if set(seal["fold_sources"]) != set(FOLDS):
        raise ValueError(f"{source_dir}: expected exactly 17 folds, 2009..2025")
    data_hash = sha256_file(data_path)
    steps = pd.read_csv(source_dir / "steps.csv", dtype={"fold": str})
    metrics = pd.read_csv(source_dir / "fold_metrics.csv", dtype={"fold": str}, float_precision="round_trip")
    for name, frame in (("steps", steps), ("fold_metrics", metrics)):
        if set(frame["fold"]) != set(FOLDS) or set(frame["policy"]) != set(POLICIES):
            raise ValueError(f"{source_dir}/{name}.csv: fold/policy set mismatch")
    rows: list[dict[str, Any]] = []
    exclusions: list[pd.DataFrame] = []
    for fold in FOLDS:
        source = seal["fold_sources"][fold]
        if source["source"]["data_identity"]["sha256"] != data_hash:
            raise ValueError(f"fold {fold}: data hash mismatch: {data_path}")
        raw = source["resolved_eval_env"]
        env, features, data = raw["environment"], raw["features"], raw["data"]
        if (
            data["provider"] != "file" or data["timeframe"] != "1d"
            or data["start_date"] != f"{fold}-01-01"
            or data["end_date"] != f"{int(fold) + 1}-01-01"
            or env["random_start"] or env["window_size"] != 32
            or features["volatility_window"] != 32
            or len(env["currency_pairs"]) != 9
        ):
            raise ValueError(f"fold {fold}: unsupported historical period contract")
        market = FileDataProvider(str(data_path)).get_data(
            tuple(env["currency_pairs"]), data["start_date"], data["end_date"], data["timeframe"],
        )
        row, excluded = audit_period(
            fold, market.index, steps.loc[steps["fold"] == fold],
            features["volatility_window"], env["window_size"],
        )
        for policy in POLICIES:
            metric = metrics.loc[(metrics["fold"] == fold) & (metrics["policy"] == policy)]
            if len(metric) != 1:
                raise ValueError(f"fold {fold} {policy}: expected exactly one metrics row")
            m = metric.iloc[0]
            if (
                m["steps"] != row["decision_count_per_policy"]
                or m["eval_start"] != row["first_decision"]
                or m["eval_end"] != row["last_target"]
                or m["terminated_by_margin_call"]
            ):
                raise ValueError(f"fold {fold} {policy}: sealed metric period mismatch")
            for prefix, cumulative in (("net", "cumulative_log_return"), ("gross", "gross_cumulative_log_return")):
                expected = math.expm1(float(m[cumulative]) / row["elapsed_years"])
                if not math.isclose(expected, float(m[f"annualized_{prefix}_return"]), rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"fold {fold} {policy}: sealed {prefix} annualization mismatch")
        rows.append(row)
        exclusions.append(excluded)
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output_dir / "fold_periods.csv", index=False)
    pd.concat(exclusions, ignore_index=True).to_csv(output_dir / "excluded_bars.csv", index=False)
    provenance = {
        "artifact_version": 1,
        "kind": "read_only_period_audit",
        "source_provenance_sha256": source_provenance_sha256,
        "source_artifact_sha256": seal["generated_artifact_sha256"],
        "source_evaluation": seal["evaluation"],
        "source_fold_contracts": seal["fold_sources"],
        "data_sha256": data_hash,
        "audit_git": git_commits(),
        "audit_module_sha256": sha256_file(Path(__file__)),
        "annualization_tolerance": {"relative": 1e-10, "absolute": 1e-12},
        "generated_artifact_sha256": {
            name: sha256_file(output_dir / name)
            for name in ("fold_periods.csv", "excluded_bars.csv")
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def main() -> None:
    """Run the explicit-input audit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-provenance-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.source_dir, args.source_provenance_sha256, args.data, args.output_dir)


if __name__ == "__main__":
    main()
