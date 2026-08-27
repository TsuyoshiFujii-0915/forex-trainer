"""Seed-ensemble evaluation CLI: forex-ensemble-eval (ADR-0007).

Loads the validation-selected models of several runs trained against the same
eval env, walks that env once, and forwards the arithmetic mean of the
members' deterministic actions at every decision. Metrics use the same
contract as forex-eval; artifacts land in runs/<experiment>_ens<N>/<ts>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from forex_env.errors import ConfigError, DataError, FeatureError

from .algorithms import ALGO_REGISTRY, resolve_device
from .artifact_provenance import (
    evaluation_runtime_provenance,
    sha256_file,
)
from .config import TrainerConfigError, parse_experiment_config
from .env_factory import build_single_env
from .evaluate import compute_metrics, walk_eval_range
from .run_dir import create_run_dir


def _load_member(
    run_dir: Path,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Load one member's config, model, and resolved eval env.

    Args:
        run_dir: Run directory produced by forex-train.

    Returns:
        Typed config, model, eval env, raw config, meta, and actual model device.

    Raises:
        TrainerConfigError: If required artifacts are missing.
    """
    snapshot_path = run_dir / "config_snapshot.yaml"
    model_path = run_dir / "model_final.zip"
    eval_env_path = run_dir / "env_eval.yaml"
    meta_path = run_dir / "meta.json"
    for required in (snapshot_path, model_path, eval_env_path, meta_path):
        if not required.is_file():
            raise TrainerConfigError(
                f"Run directory is missing {required.name}: {run_dir}"
            )
    raw_config = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    config = parse_experiment_config(raw_config)
    resolved_eval = yaml.safe_load(eval_env_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise TrainerConfigError(f"Member meta root must be a mapping: {meta_path}")
    spec = ALGO_REGISTRY[config.algorithm.name]
    evaluation_device = resolve_device(config.run.device)
    model = spec.algo_class.load(model_path, device=evaluation_device)
    return config, model, resolved_eval, raw_config, meta, evaluation_device


def run_ensemble_evaluation(
    run_dirs: list[Path], runs_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Evaluate the action-mean ensemble of several trained runs.

    Args:
        run_dirs: Member run directories; all must share an identical
            resolved eval env and decision interval.
        runs_root: Root directory under which the ensemble run directory is
            created.

    Returns:
        Tuple of (ensemble run directory, metrics dict).

    Raises:
        TrainerConfigError: If no members are given or members disagree on
            the eval env or decision interval.
    """
    if not run_dirs:
        raise TrainerConfigError("At least one member run directory is required.")
    members = [_load_member(run_dir) for run_dir in run_dirs]
    reference_config, _, reference_eval, reference_raw, _, evaluation_device = members[
        0
    ]
    for run_dir, (
        config,
        _,
        resolved_eval,
        _,
        _,
        member_device,
    ) in zip(run_dirs[1:], members[1:]):
        if resolved_eval != reference_eval:
            raise TrainerConfigError(
                f"Member {run_dir} has a different resolved eval env than "
                f"{run_dirs[0]}; ensemble members must share one eval env."
            )
        if config.run.decision_interval != reference_config.run.decision_interval:
            raise TrainerConfigError(
                f"Member {run_dir} has decision_interval "
                f"{config.run.decision_interval}, expected "
                f"{reference_config.run.decision_interval}."
            )
        if config.run.residual != reference_config.run.residual:
            raise TrainerConfigError(
                f"Member {run_dir} has a different residual action scheme."
            )
        if config.run.rank_allocation != reference_config.run.rank_allocation:
            raise TrainerConfigError(
                f"Member {run_dir} has a different rank allocation scheme."
            )
        if member_device != evaluation_device:
            raise TrainerConfigError(
                f"Member {run_dir} resolves evaluation device {member_device}, "
                f"expected {evaluation_device}."
            )

    env = build_single_env(
        reference_eval,
        reference_config.custom_feature_names,
        reference_config.custom_cross_feature_names,
        seed=0,
        decision_interval=reference_config.run.decision_interval,
        residual=reference_config.run.residual,
        rank_allocation=reference_config.run.rank_allocation,
    )
    states: list[Any] = [None] * len(members)

    def predict(observation: dict[str, Any], episode_start: np.ndarray) -> np.ndarray:
        actions = []
        for index, (_, model, _, _, _, _) in enumerate(members):
            action, states[index] = model.predict(
                observation,
                state=states[index],
                episode_start=episode_start,
                deterministic=True,
            )
            actions.append(np.asarray(action, dtype=np.float64))
        return np.mean(np.stack(actions, axis=0), axis=0).astype(np.float32)

    walk = walk_eval_range(env, predict)
    env.close()
    metrics = compute_metrics(*walk)
    equities, timestamps = walk[1], walk[2]

    experiment = f"{reference_config.experiment}_ens{len(members)}"
    ensemble_dir = create_run_dir(runs_root, experiment)
    (ensemble_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"timestamp": timestamps, "equity_jpy": equities}).to_csv(
        ensemble_dir / "equity_curve.csv", index=False
    )
    (ensemble_dir / "env_eval.yaml").write_text(
        yaml.safe_dump(dict(reference_eval), sort_keys=False), encoding="utf-8"
    )
    member_records = []
    for run_dir, (config, _, _, _, meta, _) in zip(run_dirs, members):
        for field in ("experiment", "seed"):
            if field not in meta:
                raise TrainerConfigError(
                    f"Member meta {run_dir / 'meta.json'} lacks {field}."
                )
        if meta["experiment"] != config.experiment or meta["seed"] != config.run.seed:
            raise TrainerConfigError(
                f"Member identity mismatch between config and meta: {run_dir}"
            )
        member_records.append(
            {
                "run_dir": str(run_dir.resolve()),
                "experiment": config.experiment,
                "seed": config.run.seed,
                "model_path": "model_final.zip",
                "model_sha256": sha256_file(run_dir / "model_final.zip"),
                "config_snapshot_sha256": sha256_file(run_dir / "config_snapshot.yaml"),
                "meta_sha256": sha256_file(run_dir / "meta.json"),
            }
        )
    manifest = {
        "manifest_version": 2,
        "experiment": experiment,
        "policy": "action_mean",
        "model_selection": "validation_best",
        "decision_interval": reference_config.run.decision_interval,
        "members": member_records,
        "evaluation": {
            **evaluation_runtime_provenance(
                evaluation_device,
                reference_raw,
                run_dirs[0] / "config_snapshot.yaml",
            ),
            "metrics_sha256": sha256_file(ensemble_dir / "metrics.json"),
            "env_eval_sha256": sha256_file(ensemble_dir / "env_eval.yaml"),
        },
    }
    (ensemble_dir / "ensemble.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return ensemble_dir, metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 on success, 1 on errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-ensemble-eval",
        description="Evaluate the action-mean ensemble of several runs.",
    )
    parser.add_argument(
        "--runs", type=str, nargs="+", required=True, help="Member run directories."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory for the ensemble artifacts (default: runs, the "
        "repo convention).",
    )
    args = parser.parse_args(argv)
    try:
        ensemble_dir, metrics = run_ensemble_evaluation(
            [Path(run) for run in args.runs], Path(args.runs_root)
        )
    except (TrainerConfigError, ConfigError, DataError, FeatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ensemble run: {ensemble_dir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
