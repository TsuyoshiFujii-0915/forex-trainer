"""Tests for explicit legacy-checkpoint revalidation attestations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forex_trainer.config import TrainerConfigError
from forex_trainer.legacy_revalidation import run_legacy_ensemble_attestation
from forex_trainer.train import run_training
from helpers import make_experiment_raw, write_experiment_yaml


def _legacy_run(tmp_path: Path, seed: int) -> Path:
    """Train a tiny model and remove only its modern provenance contract."""
    raw = make_experiment_raw()
    raw["run"]["total_timesteps"] = 32
    raw["run"]["n_envs"] = 1
    config_path = write_experiment_yaml(tmp_path, raw)
    run_dir = run_training(config_path, tmp_path / "runs", seed_override=seed)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("run_provenance_contract_version")
    meta.pop("data_provenance")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def test_legacy_attestation_records_unknown_training_and_current_inputs(
    tmp_path: Path,
) -> None:
    """Legacy models are measured without pretending their training was reproducible."""
    members = (_legacy_run(tmp_path / "a", 1), _legacy_run(tmp_path / "b", 2))
    output_dir = tmp_path / "attestation"

    metrics = run_legacy_ensemble_attestation(members, output_dir)

    manifest = json.loads(
        (output_dir / "attestation.json").read_text(encoding="utf-8")
    )
    assert manifest["contract"] == "legacy-checkpoint-current-environment-v1"
    assert manifest["training_provenance"] == "unverifiable"
    assert len(manifest["members"]) == 2
    assert all(len(member["model_sha256"]) == 64 for member in manifest["members"])
    assert manifest["current_evaluation"]["data"]["sha256"]
    assert set(manifest["current_evaluation"]["repositories"]) == {
        "forex_trainer",
        "forex_env",
    }
    assert metrics["steps"] > 0
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "equity_curve.csv").is_file()


def test_legacy_attestation_rejects_modern_runs(tmp_path: Path) -> None:
    """A provenance-complete run must use the normal verified evaluator."""
    raw = make_experiment_raw()
    raw["run"]["total_timesteps"] = 32
    raw["run"]["n_envs"] = 1
    run_dir = run_training(
        write_experiment_yaml(tmp_path, raw), tmp_path / "runs", seed_override=1
    )

    with pytest.raises(TrainerConfigError, match="normal verified evaluator"):
        run_legacy_ensemble_attestation((run_dir,), tmp_path / "attestation")


def test_legacy_attestation_never_overwrites_output(tmp_path: Path) -> None:
    """Attestation artifacts are immutable once their output path exists."""
    member = _legacy_run(tmp_path, 1)
    output_dir = tmp_path / "attestation"
    output_dir.mkdir()

    with pytest.raises(TrainerConfigError, match="already exists"):
        run_legacy_ensemble_attestation((member,), output_dir)
