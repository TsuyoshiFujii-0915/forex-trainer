"""Explicit current-environment attestation for legacy checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import forex_env
import numpy as np
import pandas as pd
import yaml
from forex_env.errors import ConfigError, DataError, FeatureError

from .algorithms import ALGO_REGISTRY, resolve_device
from .config import TrainerConfigError, parse_experiment_config
from .env_factory import build_single_env
from .evaluate import compute_metrics, walk_eval_range
from .run_dir import (
    capture_data_provenance,
    capture_package_versions,
    capture_repository_provenance,
    resolve_file_data_path,
)

_ATTESTATION_CONTRACT = "legacy-checkpoint-current-environment-v1"
_TRAINER_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_REPO_ROOT = Path(forex_env.__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    """Hash one attestation input file.

    Args:
        path: Input file.

    Returns:
        Lowercase SHA256 hexadecimal digest.

    Raises:
        TrainerConfigError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise TrainerConfigError(f"Failed to hash legacy input {path}: {exc}") from exc
    return digest.hexdigest()


def _load_legacy_member(run_dir: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Load one legacy member and preserve its unverifiable metadata.

    Args:
        run_dir: Legacy run directory.

    Returns:
        Parsed config, loaded model, absolute eval environment, and manifest.

    Raises:
        TrainerConfigError: If artifacts are missing, malformed, or modern.
    """
    paths = {
        "config": run_dir / "config_snapshot.yaml",
        "environment": run_dir / "env_eval.yaml",
        "model": run_dir / "model_final.zip",
        "meta": run_dir / "meta.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise TrainerConfigError(f"Legacy run is missing {path.name}: {run_dir}")
    try:
        raw_config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
        resolved_eval = yaml.safe_load(
            paths["environment"].read_text(encoding="utf-8")
        )
        legacy_meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TrainerConfigError(f"Failed to read legacy run {run_dir}: {exc}") from exc
    if not isinstance(raw_config, dict) or not isinstance(resolved_eval, dict):
        raise TrainerConfigError(f"Legacy run YAML is malformed: {run_dir}")
    if not isinstance(legacy_meta, dict):
        raise TrainerConfigError(f"Legacy run metadata is malformed: {run_dir}")
    if "run_provenance_contract_version" in legacy_meta:
        raise TrainerConfigError(
            f"Run {run_dir} has modern provenance; use the normal verified evaluator."
        )
    config = parse_experiment_config(raw_config)
    absolute_eval = resolve_file_data_path(resolved_eval, Path.cwd())
    spec = ALGO_REGISTRY[config.algorithm.name]
    model = spec.algo_class.load(
        paths["model"], device=resolve_device(config.run.device)
    )
    manifest = {
        "run": str(run_dir.resolve()),
        "model_sha256": _sha256_file(paths["model"]),
        "config_sha256": _sha256_file(paths["config"]),
        "environment_sha256": _sha256_file(paths["environment"]),
        "legacy_meta_sha256": _sha256_file(paths["meta"]),
        "legacy_git": legacy_meta.get("git"),
    }
    return config, model, absolute_eval, manifest


def run_legacy_ensemble_attestation(
    run_dirs: tuple[Path, ...], output_dir: Path
) -> dict[str, Any]:
    """Evaluate legacy checkpoints under a fully attested current environment.

    Args:
        run_dirs: Legacy run members combined by arithmetic action mean.
        output_dir: New immutable attestation directory.

    Returns:
        Current-environment evaluation metrics.

    Raises:
        TrainerConfigError: If inputs are absent/incompatible or output exists.
    """
    if not run_dirs:
        raise TrainerConfigError("At least one legacy run is required.")
    if output_dir.exists():
        raise TrainerConfigError(f"Attestation output already exists: {output_dir}")
    members = [_load_legacy_member(run_dir) for run_dir in run_dirs]
    reference_config, _, reference_eval, _ = members[0]
    for run_dir, (config, _, resolved_eval, _) in zip(run_dirs[1:], members[1:]):
        if resolved_eval != reference_eval:
            raise TrainerConfigError(
                f"Legacy member {run_dir} has a different evaluation environment."
            )
        if config.run.decision_interval != reference_config.run.decision_interval:
            raise TrainerConfigError(
                f"Legacy member {run_dir} has a different decision_interval."
            )
        if config.run.residual != reference_config.run.residual:
            raise TrainerConfigError(
                f"Legacy member {run_dir} has a different residual action contract."
            )

    data_provenance = capture_data_provenance(reference_eval)
    repository_provenance = {
        "forex_trainer": capture_repository_provenance(_TRAINER_REPO_ROOT),
        "forex_env": capture_repository_provenance(_ENV_REPO_ROOT),
    }
    package_versions = capture_package_versions()
    env = build_single_env(
        reference_eval,
        reference_config.custom_feature_names,
        reference_config.custom_cross_feature_names,
        seed=0,
        decision_interval=reference_config.run.decision_interval,
        residual=reference_config.run.residual,
    )
    states: list[Any] = [None] * len(members)

    def predict(observation: dict[str, Any], episode_start: np.ndarray) -> np.ndarray:
        """Average deterministic actions from every attested checkpoint."""
        actions: list[np.ndarray] = []
        for index, (_, model, _, _) in enumerate(members):
            action, states[index] = model.predict(
                observation,
                state=states[index],
                episode_start=episode_start,
                deterministic=True,
            )
            actions.append(np.asarray(action, dtype=np.float64))
        return np.mean(np.stack(actions, axis=0), axis=0).astype(np.float32)

    try:
        walk = walk_eval_range(env, predict)
    finally:
        env.close()
    metrics = compute_metrics(*walk)
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"timestamp": walk[2], "equity_jpy": walk[1]}).to_csv(
        output_dir / "equity_curve.csv", index=False
    )
    (output_dir / "env_eval.yaml").write_text(
        yaml.safe_dump(reference_eval, sort_keys=False), encoding="utf-8"
    )
    attestation = {
        "contract": _ATTESTATION_CONTRACT,
        "training_provenance": "unverifiable",
        "members": [member[3] for member in members],
        "current_evaluation": {
            "data": data_provenance,
            "repositories": repository_provenance,
            "versions": package_versions,
        },
        "metrics": metrics,
    }
    (output_dir / "attestation.json").write_text(
        json.dumps(attestation, indent=2), encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Run legacy current-environment attestation.

    Args:
        argv: Explicit CLI arguments; None delegates to argparse/sys.argv.

    Returns:
        Process exit code, zero on success and one on explicit failure.
    """
    parser = argparse.ArgumentParser(
        prog="forex-legacy-attest",
        description="Evaluate legacy checkpoints under an attested current environment.",
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        metrics = run_legacy_ensemble_attestation(tuple(args.runs), args.output)
    except (TrainerConfigError, ConfigError, DataError, FeatureError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
