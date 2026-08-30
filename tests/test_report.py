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
        "train_range": {"start": f"{fold - 3}-07-01", "end": f"{fold - 1}-07-01"},
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
    env_eval_path = run_dir / "env_eval.yaml"
    env_eval_path.write_text(
        yaml.safe_dump(
            {
                **config["env"],
                "environment": {
                    **config["env"]["environment"],
                    "random_start": False,
                    "episode_max_steps": 1_000_000,
                },
                "data": {
                    **config["env"]["data"],
                    "start_date": config["eval_range"]["start"],
                    "end_date": config["eval_range"]["end"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
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
                "manifest_version": 2,
                "model_selection": "validation_best",
                "model_path": "model_final.zip",
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                "config_snapshot_sha256": hashlib.sha256(
                    (run_dir / "config_snapshot.yaml").read_bytes()
                ).hexdigest(),
                "env_eval_sha256": hashlib.sha256(
                    env_eval_path.read_bytes()
                ).hexdigest(),
                "meta_sha256": hashlib.sha256(
                    (run_dir / "meta.json").read_bytes()
                ).hexdigest(),
                "resolved_device": device,
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
                "result_kind": "seed",
                "range_policy": {"kind": "rolling", "train_years": 2},
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


def _reseal_config_snapshot(run_dir: Path) -> None:
    """Update a fixture's config digest after creating a distinct valid artifact.

    Args:
        run_dir: Evaluated training-run fixture directory.
    """
    evaluation_path = run_dir / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["config_snapshot_sha256"] = hashlib.sha256(
        (run_dir / "config_snapshot.yaml").read_bytes()
    ).hexdigest()
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")


def _reseal_meta(run_dir: Path) -> None:
    """Update a fixture evaluation manifest after creating distinct training meta.

    Args:
        run_dir: Evaluated training-run fixture directory.
    """
    evaluation_path = run_dir / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["meta_sha256"] = hashlib.sha256(
        (run_dir / "meta.json").read_bytes()
    ).hexdigest()
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")


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


def _write_ensemble_artifact(
    root: Path,
    configuration: str,
    fold: int,
    members: list[Path],
    net_log_return: float,
) -> Path:
    """Write one action-mean ensemble artifact with immutable provenance.

    Args:
        root: Ensemble artifact root.
        configuration: Campaign configuration name.
        fold: Evaluation year.
        members: Source training-run directories.
        net_log_return: Ensemble policy cumulative net log return.

    Returns:
        Created ensemble directory.
    """
    ensemble_dir = root / configuration / f"fold{fold}-ens{len(members)}"
    ensemble_dir.mkdir(parents=True)
    metrics_path = ensemble_dir / "metrics.json"
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
    ensemble_env_path = ensemble_dir / "env_eval.yaml"
    ensemble_env_path.write_bytes((members[0] / "env_eval.yaml").read_bytes())
    first_meta = json.loads((members[0] / "meta.json").read_text(encoding="utf-8"))
    manifest = {
        "manifest_version": 2,
        "experiment": f"{configuration}-{fold}-ens{len(members)}",
        "policy": "action_mean",
        "model_selection": "validation_best",
        "decision_interval": 1,
        "members": [
            {
                "run_dir": str(member.resolve()),
                "experiment": json.loads(
                    (member / "meta.json").read_text(encoding="utf-8")
                )["experiment"],
                "seed": json.loads((member / "meta.json").read_text(encoding="utf-8"))[
                    "seed"
                ],
                "model_path": "model_final.zip",
                "model_sha256": hashlib.sha256(
                    (member / "model_final.zip").read_bytes()
                ).hexdigest(),
                "config_snapshot_sha256": hashlib.sha256(
                    (member / "config_snapshot.yaml").read_bytes()
                ).hexdigest(),
                "meta_sha256": hashlib.sha256(
                    (member / "meta.json").read_bytes()
                ).hexdigest(),
            }
            for member in members
        ],
        "evaluation": {
            "resolved_device": "cpu",
            "git": {"forex_trainer": "trainer-sha", "forex_env": "env-sha"},
            "versions": {"torch": "test", "stable-baselines3": "test"},
            "data_identity": first_meta["data_identity"],
            "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "env_eval_sha256": hashlib.sha256(
                ensemble_env_path.read_bytes()
            ).hexdigest(),
        },
    }
    (ensemble_dir / "ensemble.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ensemble_dir


def _ensemble_campaign(tmp_path: Path) -> Path:
    """Create two ensemble configurations over two aligned folds.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Ensemble campaign manifest.
    """
    seed_campaign = _complete_campaign(tmp_path)
    seed_raw = yaml.safe_load(seed_campaign.read_text(encoding="utf-8"))
    ensemble_runs: dict[str, list[Path]] = {}
    values = {
        "baseline": {2018: 0.18, 2019: -0.12},
        "candidate": {2018: 0.28, 2019: -0.02},
    }
    for configuration, fold_values in values.items():
        source_runs = [
            (seed_campaign.parent / value).resolve()
            for value in seed_raw["configurations"][configuration]["runs"]
        ]
        ensemble_runs[configuration] = []
        for fold, net_log_return in fold_values.items():
            members = [run for run in source_runs if f"fold{fold}-" in run.name]
            ensemble_runs[configuration].append(
                _write_ensemble_artifact(
                    tmp_path / "ensembles",
                    configuration,
                    fold,
                    members,
                    net_log_return,
                )
            )
    campaign_path = _write_campaign(tmp_path / "ensemble-campaign.yaml", ensemble_runs)
    raw = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    for configuration in raw["configurations"].values():
        configuration["result_kind"] = "ensemble"
    campaign_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return campaign_path


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
    provenance = json.loads(
        (output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["baseline"]["seeds"] == [1, 2]
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
    evaluation_path = candidate_run / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["resolved_device"] = "cuda"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

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


def test_report_rejects_evaluator_git_differences(
    tmp_path: Path,
) -> None:
    """Paired metric comparisons require one evaluator implementation contract."""
    campaign_path = _complete_campaign(tmp_path)
    for candidate_run in (tmp_path / "runs" / "candidate").iterdir():
        evaluation_path = candidate_run / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["git"]["forex_trainer"] = "candidate-sha"
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="evaluation_git"):
        run_research_report(campaign_path, tmp_path / "report")


def test_report_records_training_git_as_treatment_provenance(tmp_path: Path) -> None:
    """Training implementation differences remain valid visible treatments."""
    campaign_path = _complete_campaign(tmp_path)
    for candidate_run in (tmp_path / "runs" / "candidate").iterdir():
        meta_path = candidate_run / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["git"]["forex_trainer"] = "candidate-training-sha"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        _reseal_meta(candidate_run)

    _, report = run_research_report(campaign_path, tmp_path / "report")

    differences = report["comparisons"][0]["provenance_differences"]
    assert differences["training_git"]["baseline"]["forex_trainer"] == "trainer-sha"
    assert (
        differences["training_git"]["candidate"]["forex_trainer"]
        == "candidate-training-sha"
    )


def test_report_rejects_standard_training_meta_changed_after_evaluation(
    tmp_path: Path,
) -> None:
    """The standard manifest detects training metadata edited after evaluation."""
    campaign_path = _complete_campaign(tmp_path)
    candidate_run = next((tmp_path / "runs" / "candidate").iterdir())
    meta_path = candidate_run / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["git"]["forex_trainer"] = "post-evaluation-edit"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="Training meta changed"):
        run_research_report(campaign_path, tmp_path / "report")


def test_report_rejects_eval_env_semantic_drift_even_when_resealed(
    tmp_path: Path,
) -> None:
    """Artifact hashes cannot legitimize an eval env that contradicts config."""
    campaign_path = _complete_campaign(tmp_path)
    candidate_run = next((tmp_path / "runs" / "candidate").iterdir())
    eval_env_path = candidate_run / "env_eval.yaml"
    eval_env = yaml.safe_load(eval_env_path.read_text(encoding="utf-8"))
    eval_env["transaction_costs"]["commission_rate"] = 0.25
    eval_env_path.write_text(yaml.safe_dump(eval_env), encoding="utf-8")
    evaluation_path = candidate_run / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["env_eval_sha256"] = hashlib.sha256(
        eval_env_path.read_bytes()
    ).hexdigest()
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(TrainerConfigError, match="config snapshot"):
        run_research_report(campaign_path, tmp_path / "report")


def test_report_rejects_protocol_drift_inside_one_configuration(tmp_path: Path) -> None:
    """Fold artifacts in one named configuration must share one protocol."""
    campaign_path = _complete_campaign(tmp_path)
    baseline_runs = sorted((tmp_path / "runs" / "baseline").iterdir())
    config_path = baseline_runs[-1] / "config_snapshot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["algorithm"]["hyperparams"]["learning_rate"] = 0.123
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _reseal_config_snapshot(baseline_runs[-1])

    with pytest.raises(TrainerConfigError, match="protocol"):
        run_research_report(campaign_path, tmp_path / "report")


def test_report_rejects_mixed_history_lengths_inside_configuration(
    tmp_path: Path,
) -> None:
    """A rolling configuration cannot silently mix different history lengths."""
    campaign_path = _complete_campaign(tmp_path)
    baseline_run = max((tmp_path / "runs" / "baseline").iterdir())
    config_path = baseline_run / "config_snapshot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fold = int(config["eval_range"]["start"][:4])
    config["train_range"]["start"] = f"{fold - 4}-07-01"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _reseal_config_snapshot(baseline_run)

    with pytest.raises(TrainerConfigError, match="range"):
        run_research_report(campaign_path, tmp_path / "report")


def test_report_records_history_length_as_treatment_provenance(tmp_path: Path) -> None:
    """Different rolling history geometry remains visible between configurations."""
    campaign_path = _complete_campaign(tmp_path)
    for candidate_run in (tmp_path / "runs" / "candidate").iterdir():
        config_path = candidate_run / "config_snapshot.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        fold = int(config["eval_range"]["start"][:4])
        config["train_range"]["start"] = f"{fold - 5}-07-01"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        _reseal_config_snapshot(candidate_run)
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["configurations"]["candidate"]["range_policy"]["train_years"] = 4
    campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
    )

    _, report = run_research_report(campaign_path, tmp_path / "report")

    difference = report["comparisons"][0]["provenance_differences"]["range_identity"]
    assert difference["baseline"]["train_years"] == 2
    assert difference["candidate"]["train_years"] == 4


def test_report_records_expanding_history_as_distinct_treatment(tmp_path: Path) -> None:
    """A fixed-start expanding window is not conflated with a rolling window."""
    campaign_path = _complete_campaign(tmp_path)
    expanding_start = "2010-07-01"
    for candidate_run in (tmp_path / "runs" / "candidate").iterdir():
        config_path = candidate_run / "config_snapshot.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["train_range"]["start"] = expanding_start
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        _reseal_config_snapshot(candidate_run)
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["configurations"]["candidate"]["range_policy"] = {
        "kind": "expanding",
        "train_start": expanding_start,
    }
    campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
    )

    _, report = run_research_report(campaign_path, tmp_path / "report")

    difference = report["comparisons"][0]["provenance_differences"]["range_identity"]
    assert difference["baseline"] == {"kind": "rolling", "train_years": 2}
    assert difference["candidate"] == {
        "kind": "expanding",
        "train_start": expanding_start,
    }


def test_report_aggregates_action_mean_ensemble_fold_observations(
    tmp_path: Path,
) -> None:
    """Ensemble policy metrics are reported directly rather than seed-mean proxies."""
    campaign_path = _ensemble_campaign(tmp_path)

    _, report = run_research_report(campaign_path, tmp_path / "ensemble-report")

    baseline = report["configurations"]["baseline"]
    assert baseline["result_kind"] == "ensemble"
    assert baseline["ensemble_member_count"] == 2
    assert baseline["member_seeds"] == [1, 2]
    assert "seed_count" not in baseline
    paired = report["comparisons"][0]
    assert paired["pairing_unit"] == "fold"
    assert paired["fold_count"] == 2
    assert paired["mean_differences"]["sharpe_annualized"] == pytest.approx(1.0)


def test_report_rejects_changed_ensemble_member_model(tmp_path: Path) -> None:
    """An ensemble observation is invalid after any member model changes."""
    campaign_path = _ensemble_campaign(tmp_path)
    member_model = next((tmp_path / "runs" / "baseline").iterdir()) / "model_final.zip"
    member_model.write_bytes(b"changed after ensemble evaluation")

    with pytest.raises(TrainerConfigError, match="model"):
        run_research_report(campaign_path, tmp_path / "ensemble-report")


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
