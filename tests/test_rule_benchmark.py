"""Tests for the reproducible cross-sectional reversal benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from forex_trainer.config import TrainerConfigError
from forex_trainer.rule_benchmark import main, run_rule_benchmark
from helpers import make_experiment_raw, write_experiment_yaml

_PAIRS = ["JPY/USD", "JPY/EUR", "JPY/GBP", "JPY/AUD"]


def _rule_config(tmp_path: Path) -> Path:
    """Write a deterministic synthetic config suitable for rule evaluation."""
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = list(_PAIRS)
    raw["env"]["transaction_costs"]["spreads"] = {
        pair: 0.0 for pair in _PAIRS
    }
    raw["env"]["features"]["normalize"] = False
    raw["env"]["features"]["selected"] = ["log_return", "mom24"]
    return write_experiment_yaml(tmp_path, raw)


def test_rule_benchmark_evaluates_and_aggregates_configs(tmp_path: Path) -> None:
    """The committed benchmark reproduces a zero-residual reversal rule."""
    config_path = _rule_config(tmp_path)

    report = run_rule_benchmark(
        (config_path,), feature="mom24", top_k=1, base_size=0.8
    )

    assert report["contract"] == "cross-sectional-reversal-v1"
    assert report["parameters"] == {
        "feature": "mom24",
        "top_k": 1,
        "base_size": 0.8,
    }
    assert len(report["folds"]) == 1
    cumulative = report["folds"][0]["metrics"]["cumulative_log_return"]
    assert math.isfinite(cumulative)
    assert report["aggregate"]["annualized_return"] == pytest.approx(
        math.expm1(cumulative)
    )


def test_rule_benchmark_rejects_per_symbol_normalization(tmp_path: Path) -> None:
    """Per-symbol normalization must not silently change rank-rule semantics."""
    config_path = _rule_config(tmp_path)
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = list(_PAIRS)
    raw["env"]["transaction_costs"]["spreads"] = {
        pair: 0.0 for pair in _PAIRS
    }
    raw["env"]["features"]["selected"] = ["log_return", "mom24"]
    config_path = write_experiment_yaml(tmp_path, raw, "normalized.yaml")

    with pytest.raises(TrainerConfigError, match="normalize: false"):
        run_rule_benchmark(
            (config_path,), feature="mom24", top_k=1, base_size=0.8
        )


def test_rule_benchmark_cli_writes_explicit_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI persists the exact rule contract and fold-level metrics."""
    config_path = _rule_config(tmp_path)
    output_path = tmp_path / "rule-report.json"

    exit_code = main(
        [
            "--configs",
            str(config_path),
            "--feature",
            "mom24",
            "--top-k",
            "1",
            "--base-size",
            "0.8",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["folds"]
    assert str(output_path) in capsys.readouterr().out
