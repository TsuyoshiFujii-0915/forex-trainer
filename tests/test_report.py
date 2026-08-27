"""Aggregate research report behavior tests (Issue #6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from forex_trainer.config import TrainerConfigError
from forex_trainer.report import main, run_research_report


def _write_run(
    root: Path,
    configuration: str,
    fold: int,
    seed: int,
    net_log_return: float,
    device: str,
    data_path: Path,
) -> Path:
    """Write one complete evaluated training-run artifact.

    Args:
        root: Artifact root.
        configuration: Named campaign configuration.
        fold: Evaluation-year label.
        seed: Training seed.
        net_log_return: Evaluation-period cumulative net log return.
        device: Recorded training device.
        data_path: File-backed market data artifact.

    Returns:
        Created run directory.
    """
    run_dir = root / configuration / f"fold{fold}-seed{seed}"
    run_dir.mkdir(parents=True)
    config = {
        "experiment": f"{configuration}-{fold}-{seed}",
        "env": {
            "data": {
                "provider": "file",
                "path": str(data_path),
                "timeframe": "1d",
            },
            "environment": {"currency_pairs": ["JPY/USD"]},
            "features": {"selected": ["log_return"]},
            "transaction_costs": {"commission_rate": 0.0},
        },
        "train_range": {"start": f"{fold - 3}-01-01", "end": f"{fold - 1}-07-01"},
        "val_range": {"start": f"{fold - 1}-07-01", "end": f"{fold}-01-01"},
        "eval_range": {"start": f"{fold}-01-01", "end": f"{fold + 1}-01-01"},
        "algorithm": {"name": "ppo", "hyperparams": {"learning_rate": 0.001}},
        "network": {"name": configuration, "kwargs": {}},
        "run": {
            "total_timesteps": 100,
            "seed": seed,
            "device": "auto",
            "n_envs": 1,
            "vec_env": "dummy",
            "decision_interval": 1,
            "residual": "none",
        },
    }
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "experiment": config["experiment"],
                "seed": seed,
                "requested_device": "auto",
                "device": device,
                "algorithm": "ppo",
                "network": configuration,
                "git": {"forex_trainer": "trainer-sha", "forex_env": "env-sha"},
                "versions": {"torch": "test", "stable-baselines3": "test"},
                "data_identity": {
                    "provider": "file",
                    "path": str(data_path.resolve()),
                    "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "cumulative_log_return": net_log_return,
                "gross_cumulative_log_return": net_log_return + 0.02,
                "sharpe_annualized": net_log_return * 10.0,
                "max_drawdown": 0.20 - net_log_return / 10.0,
                "eval_start": f"{fold}-01-01T00:00:00+00:00",
                "eval_end": f"{fold + 1}-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    model_path = run_dir / "model_final.zip"
    model_path.write_bytes(f"{configuration}-{fold}-{seed}".encode())
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "model_selection": "validation_best",
                "manifest_version": 1,
                "model_path": "model_final.zip",
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_campaign(
    path: Path,
    runs: dict[str, list[Path]],
    model_selection: dict[str, str] | None = None,
) -> Path:
    """Write a strict campaign manifest for the supplied artifacts.

    Args:
        path: Manifest destination.
        runs: Run directories grouped by configuration name.
        model_selection: Per-configuration selection scheme override.

    Returns:
        Manifest path.
    """
    selections = model_selection or {name: "validation_best" for name in runs}
    manifest: dict[str, Any] = {
        "name": "aggregate_contract",
        "configurations": {
            name: {
                "model_selection": selections[name],
                "runs": [str(run.relative_to(path.parent)) for run in directories],
            }
            for name, directories in runs.items()
        },
        "comparisons": [{"baseline": "baseline", "candidate": "candidate"}],
        "eras": {
            "early": {"start": 2018, "end": 2018},
            "recent": {"start": 2019, "end": 2019},
        },
        "bootstrap_samples": 500,
        "bootstrap_seed": 17,
        "moving_block_length": 2,
        "trial_count": 12,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _complete_campaign(tmp_path: Path) -> Path:
    """Create a two-configuration, two-fold, two-seed campaign.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Campaign manifest path.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_path = tmp_path / "market.parquet"
    data_path.write_bytes(b"fixed market data identity")
    values = {
        "baseline": {
            (2018, 1): 0.10,
            (2018, 2): 0.20,
            (2019, 1): -0.20,
            (2019, 2): -0.10,
        },
        "candidate": {
            (2018, 1): 0.20,
            (2018, 2): 0.30,
            (2019, 1): -0.10,
            (2019, 2): 0.00,
        },
    }
    runs = {
        name: [
            _write_run(tmp_path / "runs", name, fold, seed, value, "cpu", data_path)
            for (fold, seed), value in fold_values.items()
        ]
        for name, fold_values in values.items()
    }
    return _write_campaign(tmp_path / "campaign.yaml", runs)


def test_report_aggregates_fold_seed_era_and_paired_uncertainty(tmp_path: Path) -> None:
    """One command emits the complete fold-aligned evidence contract."""
    campaign_path = _complete_campaign(tmp_path)

    output_dir, report = run_research_report(campaign_path, tmp_path / "report")

    assert report["campaign"] == "aggregate_contract"
    assert report["trial_count"] == 12
    assert report["configuration_count"] == 2
    baseline = report["configurations"]["baseline"]
    assert baseline["observation_count"] == 4
    assert baseline["fold_count"] == 2
    assert baseline["seed_count"] == 2
    assert baseline["overall"]["winning_folds"] == 1
    assert set(baseline["folds"]) == {"2018", "2019"}
    assert set(baseline["seeds"]) == {"1", "2"}
    assert baseline["eras"]["early"]["fold_count"] == 1
    paired = report["comparisons"][0]
    assert paired["baseline"] == "baseline"
    assert paired["candidate"] == "candidate"
    assert paired["fold_count"] == 2
    assert paired["seed_fold_pair_count"] == 4
    assert paired["folds"]["2018"]["annualized_net_return"] > 0.0
    assert paired["mean_differences"]["sharpe_annualized"] == pytest.approx(1.0)
    assert paired["uncertainty"]["annualized_net_return"]["fold_bootstrap_95_low"] > 0.0
    assert report["selection_bias"]["status"] == "not_estimated"
    assert report["uncertainty_assumptions"]["sampling_unit"] == "evaluation_fold"
    for filename in (
        "campaign_snapshot.yaml",
        "observations.csv",
        "provenance.json",
        "report.json",
        "report.md",
    ):
        assert (output_dir / filename).is_file()


def test_report_is_reproducible_for_a_fixed_bootstrap_seed(tmp_path: Path) -> None:
    """Repeated generation from the same artifacts yields identical evidence."""
    campaign_path = _complete_campaign(tmp_path)

    _, first = run_research_report(campaign_path, tmp_path / "first")
    _, second = run_research_report(campaign_path, tmp_path / "second")

    assert first == second


def test_report_rejects_an_incomplete_paired_fold_seed_matrix(tmp_path: Path) -> None:
    """Pairing never silently intersects mismatched observations."""
    campaign_path = _complete_campaign(tmp_path)
    raw = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    raw["configurations"]["candidate"]["runs"].pop()
    campaign_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="fold/seed matrix"):
        run_research_report(campaign_path, tmp_path / "report")


def test_report_rejects_mismatched_comparison_provenance(tmp_path: Path) -> None:
    """A paired comparison fails when device or model-selection conditions differ."""
    campaign_path = _complete_campaign(tmp_path)
    candidate_run = next((tmp_path / "runs" / "candidate").iterdir())
    meta_path = candidate_run / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["device"] = "cuda"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="device"):
        run_research_report(campaign_path, tmp_path / "device-report")

    campaign_path = _complete_campaign(tmp_path / "model-selection")
    candidate_run = next(
        (tmp_path / "model-selection" / "runs" / "candidate").iterdir()
    )
    evaluation_path = candidate_run / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["model_selection"] = "last"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(TrainerConfigError, match="model_selection"):
        run_research_report(campaign_path, tmp_path / "selection-report")


def test_report_records_but_does_not_reject_git_differences(tmp_path: Path) -> None:
    """Implementation SHA is visible treatment provenance, not an implicit blocker."""
    campaign_path = _complete_campaign(tmp_path)
    for candidate_run in (tmp_path / "runs" / "candidate").iterdir():
        meta_path = candidate_run / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["git"]["forex_trainer"] = "candidate-sha"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    _, report = run_research_report(campaign_path, tmp_path / "report")

    differences = report["comparisons"][0]["provenance_differences"]
    assert differences["git"]["baseline"]["forex_trainer"] == "trainer-sha"
    assert differences["git"]["candidate"]["forex_trainer"] == "candidate-sha"


def test_report_rejects_protocol_drift_inside_one_configuration(tmp_path: Path) -> None:
    """Fold artifacts in one named configuration must share one protocol."""
    campaign_path = _complete_campaign(tmp_path)
    baseline_runs = sorted((tmp_path / "runs" / "baseline").iterdir())
    config_path = baseline_runs[-1] / "config_snapshot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["algorithm"]["hyperparams"]["learning_rate"] = 0.123
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="protocol"):
        run_research_report(campaign_path, tmp_path / "report")


def test_forex_report_cli_writes_artifacts_and_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exposes successful output and actionable validation errors."""
    campaign_path = _complete_campaign(tmp_path)
    output_dir = tmp_path / "cli-report"

    assert (
        main(["--campaign", str(campaign_path), "--output-dir", str(output_dir)]) == 0
    )
    assert str(output_dir.resolve()) in capsys.readouterr().out

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("name: missing-fields\n", encoding="utf-8")
    assert (
        main(["--campaign", str(malformed), "--output-dir", str(tmp_path / "bad")]) == 1
    )
    assert "error:" in capsys.readouterr().err
