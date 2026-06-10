"""Training CLI: forex-train --config <experiment.yaml>."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from forex_env.errors import ConfigError, DataError, FeatureError

from .algorithms import build_model, resolve_device
from .config import (
    TrainerConfigError,
    load_experiment_config,
    parse_experiment_config,
    resolve_env_raw,
)
from .env_factory import build_vec_env
from .run_dir import create_run_dir, write_run_metadata


def run_training(config_path: Path, runs_root: Path, seed_override: int | None) -> Path:
    """Run one training experiment and return its run directory.

    Args:
        config_path: Path to the experiment YAML.
        runs_root: Root directory under which the run directory is created.
        seed_override: If given, replaces run.seed (CLI convenience for seed
            sweeps without editing the YAML).

    Returns:
        Path to the run directory containing all artifacts.
    """
    config, raw = load_experiment_config(config_path)
    if seed_override is not None:
        raw = copy.deepcopy(raw)
        raw["run"]["seed"] = seed_override
        config = parse_experiment_config(raw)

    resolved_train = resolve_env_raw(config.env, config.train_range, for_eval=False)
    resolved_eval = resolve_env_raw(config.env, config.eval_range, for_eval=True)
    device = resolve_device(config.run.device)

    run_dir = create_run_dir(runs_root, config.experiment)
    write_run_metadata(run_dir, config, raw, resolved_train, resolved_eval, device)

    vec_env = build_vec_env(
        resolved_train,
        config.custom_feature_names,
        config.run.n_envs,
        config.run.vec_env,
        config.run.seed,
    )
    try:
        model = build_model(config, vec_env, device, run_dir / "tensorboard")
        model.learn(total_timesteps=config.run.total_timesteps)
        model.save(run_dir / "model_final")
    finally:
        vec_env.close()
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: CLI arguments; None lets argparse read sys.argv (testability).

    Returns:
        Process exit code: 0 on success, 1 on configuration/data errors.
    """
    parser = argparse.ArgumentParser(
        prog="forex-train", description="Train an RL agent per an experiment YAML."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Experiment YAML path."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory for run artifacts (default: runs, the repo convention).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override run.seed for seed sweeps without editing the YAML.",
    )
    args = parser.parse_args(argv)
    try:
        run_dir = run_training(Path(args.config), Path(args.runs_root), args.seed)
        print(f"run completed: {run_dir}")
        return 0
    except (TrainerConfigError, ConfigError, DataError, FeatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
