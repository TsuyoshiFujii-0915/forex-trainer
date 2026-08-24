"""Checkpoint-selection study orchestration and report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from forex_trainer.config import TrainerConfigError
from forex_trainer.selection_study import (
    FoldResult,
    build_study_report,
    load_study_config,
    run_selection_study,
)
from helpers import make_experiment_raw, write_experiment_yaml


def _metrics(net: float, gross: float, sharpe: float, drawdown: float) -> dict[str, object]:
    """Create the metric subset consumed by the study report.

    Args:
        net: Annualized net return.
        gross: Annualized gross return.
        sharpe: Annualized Sharpe.
        drawdown: Maximum drawdown.

    Returns:
        Metric mapping.
    """
    return {
        "annualized_net_return": net,
        "annualized_gross_return": gross,
        "sharpe_annualized": sharpe,
        "max_drawdown": drawdown,
    }


def test_report_contains_eras_wins_and_paired_fold_differences() -> None:
    """The report exposes every acceptance-criterion comparison."""
    results = [
        FoldResult("2018", "validation_best", (42, 43, 44), _metrics(0.02, 0.03, 0.4, 0.2)),
        FoldResult("2018", "last", (42, 43, 44), _metrics(0.03, 0.05, 0.5, 0.1)),
        FoldResult("2018", "late_checkpoint_ensemble", (42, 43, 44), _metrics(0.01, 0.04, 0.3, 0.3)),
        FoldResult("2019", "validation_best", (42, 43, 44), _metrics(-0.01, 0.00, -0.2, 0.4)),
        FoldResult("2019", "last", (42, 43, 44), _metrics(0.01, 0.02, 0.1, 0.2)),
        FoldResult("2019", "late_checkpoint_ensemble", (42, 43, 44), _metrics(-0.02, -0.01, -0.3, 0.5)),
    ]
    report = build_study_report(results)
    assert report["schemes"]["validation_best"]["overall"]["winning_folds"] == 1
    assert report["schemes"]["last"]["eras"]["2009-2018"]["folds"] == 1
    assert report["schemes"]["last"]["eras"]["2019-2025"]["folds"] == 1
    paired = report["paired_differences_vs_validation_best"]["last"]
    assert paired["folds"]["2018"]["annualized_net_return"] == pytest.approx(0.01)
    assert paired["folds"]["2019"]["annualized_net_return"] == pytest.approx(0.02)
    assert paired["mean"]["max_drawdown"] == pytest.approx(-0.15)


def test_study_config_rejects_duplicate_fold_years(tmp_path: Path) -> None:
    """A study cannot compare two configs for one evaluation year."""
    raw = make_experiment_raw()
    first = write_experiment_yaml(tmp_path, raw, "a.yaml")
    raw["experiment"] = "duplicate"
    second = write_experiment_yaml(tmp_path, raw, "b.yaml")
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        yaml.safe_dump(
            {"name": "duplicate_folds", "fold_configs": [first.name, second.name], "seeds": [1, 2]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrainerConfigError, match="2020"):
        load_study_config(study_path)


def test_study_runs_all_schemes_from_one_fold_seed_matrix(tmp_path: Path) -> None:
    """One training matrix produces all three comparable scheme evaluations."""
    raw = make_experiment_raw()
    config_path = write_experiment_yaml(tmp_path, raw, "fold.yaml")
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        yaml.safe_dump(
            {"name": "tiny_selection", "fold_configs": [config_path.name], "seeds": [7, 11]}
        ),
        encoding="utf-8",
    )
    study_dir, report = run_selection_study(study_path, tmp_path / "runs")
    rows = json.loads((study_dir / "fold_results.json").read_text(encoding="utf-8"))
    assert {row["scheme"] for row in rows} == {
        "validation_best",
        "last",
        "late_checkpoint_ensemble",
    }
    assert {tuple(row["seeds"]) for row in rows} == {(7, 11)}
    manifests = {
        row["scheme"]: json.loads(
            (study_dir / row["artifact_dir"] / "model_selection.json").read_text(
                encoding="utf-8"
            )
        )
        for row in rows
    }
    assert len(manifests["validation_best"]["members"]) == 2
    assert len(manifests["last"]["members"]) == 2
    assert len(manifests["late_checkpoint_ensemble"]["members"]) == 10
    assert set(report["schemes"]) == {
        "validation_best",
        "last",
        "late_checkpoint_ensemble",
    }
    assert (study_dir / "report.md").is_file()
