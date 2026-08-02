"""Reproducible cross-sectional reversal benchmark CLI (ADR-0012)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import forex_env
from forex_env.errors import ConfigError, DataError, FeatureError

from .config import (
    ResidualConfig,
    TrainerConfigError,
    load_experiment_config,
    resolve_env_raw,
)
from .env_factory import build_single_env
from .evaluate import compute_metrics, walk_eval_range
from .run_dir import (
    capture_data_provenance,
    capture_package_versions,
    capture_repository_provenance,
    resolve_file_data_path,
)

_RULE_CONTRACT = "cross-sectional-reversal-v1"
_TRAINER_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_REPO_ROOT = Path(forex_env.__file__).resolve().parents[2]


def _validate_rule_contract(
    config_path: Path,
    config: Any,
    feature: str,
    top_k: int,
    base_size: float,
) -> None:
    """Validate that one experiment can express the requested rule.

    Args:
        config_path: Originating experiment path for diagnostics.
        config: Parsed ExperimentConfig.
        feature: Selected feature ranked across pairs.
        top_k: Number of pairs assigned to each rank tail.
        base_size: Absolute target weight assigned to selected pairs.

    Raises:
        TrainerConfigError: If the rule contract is invalid for the config.
    """
    if not feature:
        raise TrainerConfigError("Rule feature must be a non-empty string.")
    selected = tuple(config.env["features"]["selected"])
    if feature not in selected:
        raise TrainerConfigError(
            f"Rule feature '{feature}' is not selected by {config_path}; "
            f"selected: {list(selected)}."
        )
    if config.env["features"]["normalize"] is not False:
        raise TrainerConfigError(
            f"Rule benchmark requires env.features.normalize: false in {config_path}."
        )
    if top_k < 1:
        raise TrainerConfigError(f"Rule top_k must be >= 1, got {top_k}.")
    pairs = tuple(config.env["environment"]["currency_pairs"])
    if 2 * top_k > len(pairs):
        raise TrainerConfigError(
            f"Rule top_k={top_k} creates overlapping tails for {len(pairs)} "
            f"pairs in {config_path}."
        )
    if not math.isfinite(base_size) or not 0.0 < base_size <= 1.0:
        raise TrainerConfigError(
            f"Rule base_size must be finite and in (0, 1], got {base_size!r}."
        )


def _evaluate_rule_config(
    config_path: Path, feature: str, top_k: int, base_size: float
) -> dict[str, Any]:
    """Evaluate the reversal rule on one experiment's held-out range.

    Args:
        config_path: Experiment YAML path.
        feature: Selected feature ranked across pairs.
        top_k: Number of pairs assigned to each rank tail.
        base_size: Absolute target weight assigned to selected pairs.

    Returns:
        Fold record containing experiment identity, range, and metrics.
    """
    config, _ = load_experiment_config(config_path)
    _validate_rule_contract(config_path, config, feature, top_k, base_size)
    resolved_eval = resolve_file_data_path(
        resolve_env_raw(config.env, config.eval_range, for_eval=True), Path.cwd()
    )
    data_provenance = capture_data_provenance(resolved_eval)
    residual = ResidualConfig(
        feature=feature,
        top_k=top_k,
        base_size=base_size,
        scale=0.0,
    )
    env = build_single_env(
        resolved_eval,
        config.custom_feature_names,
        config.custom_cross_feature_names,
        seed=0,
        decision_interval=config.run.decision_interval,
        residual=residual,
    )

    def predict(
        observation: Mapping[str, Any], episode_start: np.ndarray
    ) -> np.ndarray:
        """Return a zero residual, leaving the configured base rule unchanged."""
        del observation, episode_start
        return np.zeros(env.action_space.shape, dtype=np.float32)

    try:
        walk = walk_eval_range(env, predict)
    finally:
        env.close()
    return {
        "experiment": config.experiment,
        "config": str(config_path),
        "eval_range": {"start": config.eval_range.start, "end": config.eval_range.end},
        "data_provenance": data_provenance,
        "metrics": compute_metrics(*walk),
    }


def _aggregate_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one-year walk-forward folds using their mean log return.

    Args:
        folds: Fold records returned by `_evaluate_rule_config`.

    Returns:
        Aggregate fold count, mean log return, compounded return, and wins.
    """
    log_returns = [
        float(fold["metrics"]["cumulative_log_return"]) for fold in folds
    ]
    mean_log_return = float(np.mean(log_returns))
    return {
        "folds": len(folds),
        "mean_cumulative_log_return": mean_log_return,
        "annualized_return": math.expm1(mean_log_return),
        "positive_folds": sum(value > 0.0 for value in log_returns),
    }


def run_rule_benchmark(
    config_paths: tuple[Path, ...], feature: str, top_k: int, base_size: float
) -> dict[str, Any]:
    """Run the same explicit reversal rule across experiment configs.

    Args:
        config_paths: Experiment YAML paths defining held-out folds.
        feature: Selected feature ranked across pairs.
        top_k: Number of pairs assigned to each rank tail.
        base_size: Absolute target weight assigned to selected pairs.

    Returns:
        JSON-serializable benchmark report.

    Raises:
        TrainerConfigError: If no config paths are supplied.
    """
    if not config_paths:
        raise TrainerConfigError("At least one experiment config is required.")
    folds = [
        _evaluate_rule_config(path, feature, top_k, base_size)
        for path in config_paths
    ]
    return {
        "contract": _RULE_CONTRACT,
        "parameters": {
            "feature": feature,
            "top_k": top_k,
            "base_size": base_size,
        },
        "folds": folds,
        "aggregate": _aggregate_fold_metrics(folds),
        "current_evaluation": {
            "repositories": {
                "forex_trainer": capture_repository_provenance(_TRAINER_REPO_ROOT),
                "forex_env": capture_repository_provenance(_ENV_REPO_ROOT),
            },
            "versions": capture_package_versions(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the rule benchmark CLI.

    Args:
        argv: Explicit CLI arguments; None delegates to argparse/sys.argv.

    Returns:
        Process exit code, zero on success and one on an explicit error.
    """
    parser = argparse.ArgumentParser(
        prog="forex-rule-eval",
        description="Evaluate an explicit reversal rule on experiment folds.",
    )
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--base-size", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_rule_benchmark(
            tuple(args.configs), args.feature, args.top_k, args.base_size
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (TrainerConfigError, ConfigError, DataError, FeatureError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
