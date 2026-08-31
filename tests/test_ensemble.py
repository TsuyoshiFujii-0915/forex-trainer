"""Tests for seed-ensemble evaluation (ADR-0007)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from helpers import make_experiment_raw, write_experiment_yaml

from forex_trainer.config import TrainerConfigError
from forex_trainer.ensemble import run_ensemble_evaluation
from forex_trainer.evaluate import run_evaluation
from forex_trainer.train import run_training


def _train(tmp_path: Path, seed: int, name: str = "ens_target") -> Path:
    """Train one tiny run and return its run directory.

    Args:
        tmp_path: Test-scoped directory.
        seed: Seed override for the run.
        name: Experiment name.

    Returns:
        Run directory path.
    """
    raw = make_experiment_raw()
    raw["experiment"] = name
    config_path = write_experiment_yaml(tmp_path, raw, name=f"{name}_{seed}.yaml")
    return run_training(config_path, tmp_path / "runs", seed_override=seed)


def test_single_member_ensemble_matches_plain_evaluation(tmp_path: Path) -> None:
    """An ensemble of one model reproduces forex-eval exactly."""
    run_dir = _train(tmp_path, seed=1)
    plain = run_evaluation(run_dir)
    ensemble_dir, metrics = run_ensemble_evaluation([run_dir], tmp_path / "runs")
    assert metrics["steps"] == plain["steps"]
    assert metrics["cumulative_log_return"] == pytest.approx(
        plain["cumulative_log_return"], abs=1e-12
    )
    assert (ensemble_dir / "metrics.json").is_file()
    assert (ensemble_dir / "equity_curve.csv").is_file()
    manifest = json.loads((ensemble_dir / "ensemble.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 2
    assert manifest["policy"] == "action_mean"
    assert manifest["model_selection"] == "validation_best"
    assert len(manifest["members"]) == 1
    assert manifest["members"][0]["seed"] == 1
    assert len(manifest["members"][0]["model_sha256"]) == 64
    assert len(manifest["members"][0]["config_snapshot_sha256"]) == 64
    assert len(manifest["members"][0]["meta_sha256"]) == 64
    assert manifest["evaluation"]["resolved_device"] == "cpu"
    assert set(manifest["evaluation"]["git"]) == {"forex_trainer", "forex_env"}
    assert "torch" in manifest["evaluation"]["versions"]
    assert len(manifest["evaluation"]["metrics_sha256"]) == 64
    assert len(manifest["evaluation"]["env_eval_sha256"]) == 64


def test_two_member_ensemble_walks_shared_env(tmp_path: Path) -> None:
    """A two-member ensemble produces the full metrics contract."""
    run_a = _train(tmp_path, seed=1)
    run_b = _train(tmp_path, seed=2)
    ensemble_dir, metrics = run_ensemble_evaluation([run_a, run_b], tmp_path / "runs")
    plain = run_evaluation(run_a)
    assert metrics["steps"] == plain["steps"]
    assert math.isfinite(metrics["cumulative_log_return"])
    assert math.isfinite(metrics["gross_cumulative_log_return"])
    assert "ens2" in str(ensemble_dir)


def test_mismatched_eval_envs_rejected(tmp_path: Path) -> None:
    """Members trained against different eval envs must be rejected."""
    run_a = _train(tmp_path, seed=1)
    raw = make_experiment_raw()
    raw["experiment"] = "ens_other"
    raw["eval_range"] = {"start": "2020-02-20", "end": "2020-03-01"}
    config_path = write_experiment_yaml(tmp_path, raw, name="other.yaml")
    run_b = run_training(config_path, tmp_path / "runs", seed_override=1)
    with pytest.raises(TrainerConfigError, match="eval env"):
        run_ensemble_evaluation([run_a, run_b], tmp_path / "runs")


def test_evaluators_reject_eval_env_drift_from_config_snapshot(tmp_path: Path) -> None:
    """A sealed eval env cannot differ semantically from its config snapshot."""
    run_dir = _train(tmp_path, seed=1)
    eval_env_path = run_dir / "env_eval.yaml"
    eval_env = yaml.safe_load(eval_env_path.read_text(encoding="utf-8"))
    eval_env["transaction_costs"]["commission_rate"] = 0.25
    eval_env_path.write_text(yaml.safe_dump(eval_env), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="config snapshot"):
        run_evaluation(run_dir)
    with pytest.raises(TrainerConfigError, match="config snapshot"):
        run_ensemble_evaluation([run_dir], tmp_path / "runs")


def test_evaluators_reject_legacy_training_provenance(tmp_path: Path) -> None:
    """Evaluation cannot turn an incomplete legacy run into a v2 artifact."""
    run_dir = _train(tmp_path, seed=1)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["requested_device"]
    del meta["data_identity"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="Training provenance"):
        run_evaluation(run_dir)
    with pytest.raises(TrainerConfigError, match="Training provenance"):
        run_ensemble_evaluation([run_dir], tmp_path / "runs")
