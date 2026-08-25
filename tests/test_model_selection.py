"""User-facing model-selection artifact and evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forex_trainer.config import TrainerConfigError
from forex_trainer.evaluate import run_evaluation
from forex_trainer.model_selection import (
    ModelArtifact,
    evaluate_model_artifacts,
    load_late_checkpoint_artifacts,
)
from forex_trainer.train import run_training
from helpers import make_experiment_raw, write_experiment_yaml


def _train(tmp_path: Path, seed: int = 3) -> Path:
    """Train one small experiment for model-selection tests.

    Args:
        tmp_path: Test-scoped directory.
        seed: Training seed.

    Returns:
        Created run directory.
    """
    raw = make_experiment_raw()
    config_path = write_experiment_yaml(tmp_path, raw)
    return run_training(config_path, tmp_path / "runs", seed_override=seed)


def test_training_persists_declared_late_checkpoints(tmp_path: Path) -> None:
    """Training records exactly five late validation-time checkpoints."""
    run_dir = _train(tmp_path)
    manifest = json.loads(
        (run_dir / "late_checkpoints.json").read_text(encoding="utf-8")
    )
    assert manifest["rule"] == "final_five_validation_walks"
    assert manifest["nominal_timesteps"] == [52, 56, 58, 62, 64]
    assert len(manifest["checkpoints"]) == 5
    assert [item["nominal_timestep"] for item in manifest["checkpoints"]] == [
        52,
        56,
        58,
        62,
        64,
    ]
    checkpoint_paths = sorted((run_dir / "late_checkpoints").glob("*.zip"))
    assert len(checkpoint_paths) == 5
    assert {path.name for path in checkpoint_paths} == {
        item["path"] for item in manifest["checkpoints"]
    }


def test_last_model_is_evaluated_without_validation_best(tmp_path: Path) -> None:
    """An explicitly selected last model never falls back to model_final."""
    run_dir = _train(tmp_path)
    (run_dir / "model_final.zip").unlink()
    artifact = ModelArtifact.from_run_model(run_dir, "model_last.zip")
    output_dir = tmp_path / "last_eval"
    metrics = evaluate_model_artifacts([artifact], output_dir, "last")
    assert metrics["steps"] > 0
    evaluation = json.loads(
        (output_dir / "model_selection.json").read_text(encoding="utf-8")
    )
    assert evaluation["scheme"] == "last"
    assert evaluation["members"][0]["model_path"].endswith("model_last.zip")


def test_validation_best_is_evaluated_without_last(tmp_path: Path) -> None:
    """An explicitly selected validation-best model does not require last."""
    run_dir = _train(tmp_path)
    (run_dir / "model_last.zip").unlink()
    plain = run_evaluation(run_dir)
    artifact = ModelArtifact.from_run_model(run_dir, "model_final.zip")
    metrics = evaluate_model_artifacts(
        [artifact], tmp_path / "best_eval", "validation_best"
    )
    assert metrics["cumulative_log_return"] == pytest.approx(
        plain["cumulative_log_return"], abs=1e-12
    )


def test_one_checkpoint_ensemble_matches_explicit_checkpoint_eval(
    tmp_path: Path,
) -> None:
    """One checkpoint through action averaging preserves its exact behavior."""
    run_dir = _train(tmp_path)
    checkpoint = load_late_checkpoint_artifacts(run_dir)[0]
    first = evaluate_model_artifacts(
        [checkpoint], tmp_path / "checkpoint_a", "late_checkpoint_ensemble"
    )
    second = evaluate_model_artifacts(
        [checkpoint], tmp_path / "checkpoint_b", "late_checkpoint_ensemble"
    )
    assert second["cumulative_log_return"] == pytest.approx(
        first["cumulative_log_return"], abs=1e-12
    )
    assert second["gross_cumulative_log_return"] == pytest.approx(
        first["gross_cumulative_log_return"], abs=1e-12
    )


def test_missing_declared_checkpoint_fails_explicitly(tmp_path: Path) -> None:
    """A partial checkpoint ensemble is rejected instead of silently shrinking."""
    run_dir = _train(tmp_path)
    manifest = json.loads(
        (run_dir / "late_checkpoints.json").read_text(encoding="utf-8")
    )
    missing = run_dir / "late_checkpoints" / manifest["checkpoints"][2]["path"]
    missing.unlink()
    with pytest.raises(TrainerConfigError, match=str(missing)):
        load_late_checkpoint_artifacts(run_dir)
