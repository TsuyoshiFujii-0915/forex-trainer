"""Shared helpers for the forex-trainer test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_experiment_raw() -> dict[str, Any]:
    """Build a tiny, fast experiment config for tests (synthetic data).

    Returns:
        Raw experiment dictionary; tests mutate it per scenario.
    """
    return {
        "experiment": "test_exp",
        "env": {
            "environment": {
                "seed": 11,
                "initial_balance_jpy": 1_000_000.0,
                "episode_max_steps": 32,
                "window_size": 8,
                "max_leverage": 5.0,
                "margin_call_threshold": 0.2,
                "allow_action_leverage": False,
                "random_start": True,
                "currency_pairs": ["JPY/USD"],
            },
            "data": {"provider": "synthetic", "timeframe": "1h"},
            "features": {
                "volatility_window": 8,
                "normalize": True,
                "selected": ["log_return", "volatility", "rsi14"],
            },
            "transaction_costs": {
                "commission_rate": 0.0001,
                "overnight_rate": 0.0,
                "carry_mode": "none",
                "spreads": {"JPY/USD": 0.0001},
            },
        },
        "train_range": {"start": "2020-01-01", "end": "2020-02-01"},
        "val_range": {"start": "2020-02-01", "end": "2020-02-15"},
        "eval_range": {"start": "2020-02-15", "end": "2020-03-01"},
        "algorithm": {"name": "ppo", "hyperparams": {"n_steps": 16, "batch_size": 16}},
        "network": {"name": "mlp", "kwargs": {"features_dim": 32, "hidden_dim": 32}},
        "run": {
            "total_timesteps": 64,
            "seed": 3,
            "device": "cpu",
            "n_envs": 2,
            "vec_env": "dummy",
            "decision_interval": 1,
            "residual": "none",
            "rank_allocation": "none",
            "apply_hold_gate": "none",
        },
    }


def write_experiment_yaml(
    tmp_path: Path, raw: dict[str, Any], name: str = "exp.yaml"
) -> Path:
    """Write an experiment dict as YAML and return its path.

    Args:
        tmp_path: Directory for the file.
        raw: Raw experiment dictionary.
        name: File name.

    Returns:
        Path to the written YAML file.
    """
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path
