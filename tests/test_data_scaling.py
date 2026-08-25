"""Controlled data-scaling study behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from forex_env.data.file_provider import save_ohlcv_parquet
from forex_env.data.synthetic import SyntheticDataProvider

from forex_trainer.config import TrainerConfigError
from forex_trainer.data_scaling import (
    effective_sample_size,
    load_scaling_study,
    rollout_accounting,
    run_data_scaling_study,
)
from helpers import make_experiment_raw, write_experiment_yaml


def _write_study(path: Path, fold_name: str) -> None:
    """Write a minimal strict scaling-study definition.

    Args:
        path: Study YAML destination.
        fold_name: Fold path relative to the study YAML.
    """
    path.write_text(
        yaml.safe_dump(
            {
                "name": "tiny_data_scaling",
                "fold_configs": [fold_name],
                "audit_fold_configs": [fold_name],
                "seeds": [7, 11],
                "history_years": [1],
                "device": "cpu",
                "workers": 1,
                "bootstrap_samples": 100,
                "bootstrap_seed": 23,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_rollout_accounting_distinguishes_requested_from_executed_steps() -> None:
    """The longf lower-bound budget is expanded to complete PPO rollouts."""
    accounting = rollout_accounting(
        requested_steps=300_000,
        n_steps=1_024,
        n_envs=8,
        episode_max_steps=128,
        batch_size=1_024,
        n_epochs=10,
    )
    assert accounting.actual_steps == 303_104
    assert accounting.rollouts == 37
    assert accounting.episode_equivalents == 2_368
    assert accounting.optimizer_minibatch_steps == 2_960
    assert accounting.sample_presentations == 3_031_040


def test_effective_sample_size_detects_serial_information_reuse() -> None:
    """A slowly changing series has far less information than its row count."""
    independent = np.tile(np.array([-1.0, 1.0]), 200)
    persistent = np.repeat(np.array([-1.0, 1.0]), 200)
    independent_ess = effective_sample_size(independent, max_lag=100)
    persistent_ess = effective_sample_size(persistent, max_lag=100)
    assert independent_ess == pytest.approx(len(independent))
    assert persistent_ess < len(persistent) / 10


def test_study_schema_rejects_unknown_keys(tmp_path: Path) -> None:
    """A study cannot silently accept an unimplemented experimental control."""
    raw = make_experiment_raw()
    write_experiment_yaml(tmp_path, raw, "fold.yaml")
    study_path = tmp_path / "study.yaml"
    _write_study(study_path, "fold.yaml")
    study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    study["fallback"] = "forbidden"
    study_path.write_text(yaml.safe_dump(study), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="exactly"):
        load_scaling_study(study_path)


def test_scaling_study_runs_fixed_and_expanding_histories(tmp_path: Path) -> None:
    """The committed matrix trains every seed and emits auditable curves."""
    cache_path = tmp_path / "market.parquet"
    data = SyntheticDataProvider(seed=19).get_data(
        ("JPY/USD",), "2019-01-01", "2020-03-01", "1h"
    )
    save_ohlcv_parquet(data, "1h", "2019-01-01", "2020-03-01", cache_path)

    raw = make_experiment_raw()
    raw["env"]["data"] = {
        "provider": "file",
        "timeframe": "1h",
        "path": str(cache_path),
    }
    raw["train_range"] = {"start": "2019-01-01", "end": "2020-02-01"}
    fold_path = write_experiment_yaml(tmp_path, raw, "fold.yaml")
    study_path = tmp_path / "study.yaml"
    _write_study(study_path, fold_path.name)

    study_dir = tmp_path / "study-output"
    report = run_data_scaling_study(
        study_path, tmp_path / "runs", study_dir
    )

    assert set(report["conditions"]) == {"1y", "expanding"}
    source_runs = json.loads(
        (study_dir / "source_runs.json").read_text(encoding="utf-8")
    )
    assert len(source_runs) == 4
    assert all(Path(row["run_dir"]).is_dir() for row in source_runs.values())
    for name in (
        "data_audit.csv",
        "scaling_results.csv",
        "report.json",
        "report.md",
    ):
        assert (study_dir / name).is_file()
