"""User-visible training-accounting artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

from forex_trainer.train import run_training
from helpers import make_experiment_raw, write_experiment_yaml


def test_training_records_actual_sampling_and_selection_counts(tmp_path: Path) -> None:
    """Every completed run records actual steps, episodes, and validation reuse."""
    raw = make_experiment_raw()
    config_path = write_experiment_yaml(tmp_path, raw)

    run_dir = run_training(config_path, tmp_path / "runs", seed_override=None)

    stats = json.loads((run_dir / "training_stats.json").read_text(encoding="utf-8"))
    assert stats["requested_environment_steps"] == 64
    assert stats["actual_environment_steps"] == 64
    assert stats["completed_episodes"] == 2
    assert stats["validation_walks"] >= 1
    assert 0 < stats["selected_checkpoint_step"] <= 64
