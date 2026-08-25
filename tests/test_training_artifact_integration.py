"""Training artifact contract integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from forex_trainer.train import run_training
from helpers import make_experiment_raw, write_experiment_yaml


def test_training_writes_accounting_and_late_checkpoint_artifacts(
    tmp_path: Path,
) -> None:
    """One training run must satisfy both accounting and selection contracts."""
    raw = make_experiment_raw()
    config_path = write_experiment_yaml(tmp_path, raw)

    run_dir = run_training(config_path, tmp_path / "runs", seed_override=None)

    training_stats = json.loads(
        (run_dir / "training_stats.json").read_text(encoding="utf-8")
    )
    late_manifest = json.loads(
        (run_dir / "late_checkpoints.json").read_text(encoding="utf-8")
    )
    assert training_stats["actual_environment_steps"] == 64
    assert training_stats["completed_episodes"] == 2
    assert late_manifest["nominal_timesteps"] == [52, 56, 58, 62, 64]
    assert len(late_manifest["checkpoints"]) == 5
    assert len(list((run_dir / "late_checkpoints").glob("*.zip"))) == 5
