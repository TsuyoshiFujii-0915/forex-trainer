"""End-to-end pipeline tests: train -> artifacts -> evaluate -> compare.

These exercise every registered algorithm and network with tiny budgets on
synthetic data, plus the file-provider path and seed determinism.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from forex_env.fetch import main as fetch_main
from helpers import make_experiment_raw, write_experiment_yaml

from forex_trainer.algorithms import ALGO_REGISTRY
from forex_trainer.compare import main as compare_main
from forex_trainer.evaluate import run_evaluation
from forex_trainer.networks import NETWORK_REGISTRY
from forex_trainer.train import main as train_main
from forex_trainer.train import run_training

_ONPOLICY_HYPERPARAMS = {"n_steps": 16, "batch_size": 16}
_OFFPOLICY_HYPERPARAMS = {
    "learning_starts": 16,
    "batch_size": 32,
    "buffer_size": 1000,
    "train_freq": 1,
    "gradient_steps": 1,
}
_ALGO_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "ppo": _ONPOLICY_HYPERPARAMS,
    "recurrent_ppo": _ONPOLICY_HYPERPARAMS,
    "sac": _OFFPOLICY_HYPERPARAMS,
    "td3": _OFFPOLICY_HYPERPARAMS,
    "tqc": _OFFPOLICY_HYPERPARAMS,
}


def _run_smoke(tmp_path: Path, raw: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Train + evaluate one tiny experiment and assert the artifact contract.

    Args:
        tmp_path: Test-scoped directory for config and runs.
        raw: Raw experiment dictionary.

    Returns:
        Tuple of (run directory, metrics dict).
    """
    config_path = write_experiment_yaml(tmp_path, raw)
    run_dir = run_training(config_path, tmp_path / "runs", seed_override=None)

    assert (run_dir / "model_final.zip").is_file()
    assert (run_dir / "model_last.zip").is_file()
    assert (run_dir / "evaluations.npz").is_file()
    assert (run_dir / "config_snapshot.yaml").is_file()
    assert (run_dir / "env_train.yaml").is_file()
    assert (run_dir / "env_val.yaml").is_file()
    assert (run_dir / "env_eval.yaml").is_file()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["git"]) == {"forex_trainer", "forex_env"}
    assert any((run_dir / "tensorboard").iterdir())

    metrics = run_evaluation(run_dir)
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert metrics["steps"] > 10
    assert math.isfinite(metrics["cumulative_log_return"])
    assert math.isfinite(metrics["gross_cumulative_log_return"])
    # Costs are non-negative, so pre-cost returns bound net returns from above.
    assert (
        metrics["gross_cumulative_log_return"]
        >= metrics["cumulative_log_return"] - 1e-12
    )
    assert 0.0 <= metrics["max_drawdown"] < 1.0
    assert metrics["mean_weight_turnover"] >= 0.0
    assert metrics["total_weight_turnover"] >= metrics["mean_weight_turnover"]
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "equity_curve.csv").is_file()
    assert evaluation["model_selection"] == "validation_best"
    assert evaluation["model_path"] == "model_final.zip"
    assert len(evaluation["model_sha256"]) == 64
    assert len(evaluation["metrics_sha256"]) == 64
    assert len(evaluation["config_snapshot_sha256"]) == 64
    assert len(evaluation["env_eval_sha256"]) == 64
    assert len(evaluation["meta_sha256"]) == 64
    assert evaluation["resolved_device"] == "cpu"
    assert set(evaluation["git"]) == {"forex_trainer", "forex_env"}
    assert "torch" in evaluation["versions"]
    assert evaluation["data_identity"] == meta["data_identity"]
    return run_dir, metrics


@pytest.mark.parametrize("algo", sorted(ALGO_REGISTRY))
def test_every_algorithm_trains_and_evaluates(tmp_path: Path, algo: str) -> None:
    """The full pipeline works for every registered algorithm (mlp network)."""
    raw = make_experiment_raw()
    raw["experiment"] = f"algo_{algo}"
    raw["algorithm"] = {"name": algo, "hyperparams": dict(_ALGO_HYPERPARAMS[algo])}
    raw["run"]["total_timesteps"] = 48
    raw["run"]["n_envs"] = 1
    _run_smoke(tmp_path, raw)


@pytest.mark.parametrize("network", sorted(NETWORK_REGISTRY))
def test_every_network_trains_and_evaluates(tmp_path: Path, network: str) -> None:
    """The full pipeline works for every registered network (ppo algorithm)."""
    raw = make_experiment_raw()
    raw["experiment"] = f"net_{network}"
    raw["network"] = {"name": network, "kwargs": {"features_dim": 32}}
    _run_smoke(tmp_path, raw)


def test_same_seed_reproduces_identical_results(tmp_path: Path) -> None:
    """Two runs with identical config and seed yield identical eval metrics."""
    raw = make_experiment_raw()
    _, metrics_a = _run_smoke(tmp_path / "a", raw)
    _, metrics_b = _run_smoke(tmp_path / "b", raw)
    assert metrics_a["cumulative_log_return"] == pytest.approx(
        metrics_b["cumulative_log_return"], abs=1e-12
    )
    assert metrics_a["steps"] == metrics_b["steps"]


def test_rank_allocation_trains_and_evaluates(tmp_path: Path) -> None:
    """Structured scores work through training, selection, and evaluation."""
    raw = make_experiment_raw()
    raw["experiment"] = "rank_allocation_smoke"
    raw["env"]["environment"]["currency_pairs"] = ["JPY/USD", "JPY/EUR"]
    raw["env"]["transaction_costs"]["spreads"] = {
        "JPY/USD": 0.0001,
        "JPY/EUR": 0.0001,
    }
    raw["run"]["rank_allocation"] = {"top_k": 1, "gross_exposure": 1.0}
    _, metrics = _run_smoke(tmp_path, raw)
    assert metrics["mean_gross_leverage"] == pytest.approx(1.0, rel=0.05)


def test_learned_leverage_mode_still_evaluates(tmp_path: Path) -> None:
    """Turnover reporting preserves the existing full direct-action mode."""
    raw = make_experiment_raw()
    raw["experiment"] = "learned_leverage_smoke"
    raw["env"]["environment"]["allow_action_leverage"] = True
    _, metrics = _run_smoke(tmp_path, raw)
    assert metrics["mean_weight_turnover"] >= 0.0


def test_file_provider_pipeline(tmp_path: Path) -> None:
    """fetch -> parquet cache -> train/eval through the file provider."""
    fetch_config = {
        "environment": {
            "seed": 11,
            "initial_balance_jpy": 1_000_000.0,
            "episode_max_steps": 32,
            "window_size": 8,
            "max_leverage": 5.0,
            "margin_call_threshold": 0.2,
            "allow_action_leverage": False,
            "random_start": False,
            "currency_pairs": ["JPY/USD"],
        },
        "data": {
            "provider": "synthetic",
            "start_date": "2020-01-01",
            "end_date": "2020-03-01",
            "timeframe": "1h",
        },
        "features": {"volatility_window": 8, "normalize": True, "selected": []},
        "transaction_costs": {
            "commission_rate": 0.0,
            "overnight_rate": 0.0,
            "carry_mode": "none",
            "spreads": {"JPY/USD": 0.0},
        },
    }
    fetch_path = tmp_path / "fetch.yaml"
    fetch_path.write_text(yaml.safe_dump(fetch_config), encoding="utf-8")
    cache_path = tmp_path / "cache.parquet"
    assert fetch_main(["--config", str(fetch_path), "--output", str(cache_path)]) == 0

    raw = make_experiment_raw()
    raw["experiment"] = "file_provider_smoke"
    raw["env"]["data"] = {
        "provider": "file",
        "timeframe": "1h",
        "path": str(cache_path),
    }
    _run_smoke(tmp_path, raw)


def test_compare_lists_evaluated_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """forex-compare tabulates evaluated runs and fails cleanly when empty."""
    assert compare_main([str(tmp_path / "runs")]) == 1
    capsys.readouterr()

    raw = make_experiment_raw()
    raw["experiment"] = "compare_target"
    _run_smoke(tmp_path, raw)
    assert compare_main([str(tmp_path / "runs")]) == 0
    output = capsys.readouterr().out
    assert "compare_target" in output
    assert "sharpe" in output


def test_train_cli_reports_config_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The train CLI exits 1 with a diagnostic on invalid configs."""
    raw = make_experiment_raw()
    raw["algorithm"]["name"] = "nonexistent"
    config_path = write_experiment_yaml(tmp_path, raw)
    exit_code = train_main(
        ["--config", str(config_path), "--runs-root", str(tmp_path / "runs")]
    )
    assert exit_code == 1
    assert "nonexistent" in capsys.readouterr().err


def test_seed_override_changes_run(tmp_path: Path) -> None:
    """--seed overrides run.seed and is recorded in the run metadata."""
    raw = make_experiment_raw()
    config_path = write_experiment_yaml(tmp_path, raw)
    run_dir = run_training(config_path, tmp_path / "runs", seed_override=99)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 99
