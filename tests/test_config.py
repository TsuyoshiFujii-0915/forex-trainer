"""Tests for fail-fast experiment configuration."""

from __future__ import annotations

import pytest
from forex_env.errors import ConfigError

from forex_trainer.config import (
    TrainerConfigError,
    load_experiment_config,
    parse_experiment_config,
    resolve_env_raw,
)
from helpers import make_experiment_raw, write_experiment_yaml


def test_valid_config_parses(tmp_path) -> None:
    """A complete experiment config loads into typed dataclasses."""
    path = write_experiment_yaml(tmp_path, make_experiment_raw())
    config, raw = load_experiment_config(path)
    assert config.experiment == "test_exp"
    assert config.algorithm.name == "ppo"
    assert config.network.name == "mlp"
    assert config.custom_feature_names == ("rsi14",)
    assert raw["experiment"] == "test_exp"


def test_missing_file_raises() -> None:
    """A nonexistent config path raises with the path in the message."""
    with pytest.raises(TrainerConfigError, match="missing.yaml"):
        load_experiment_config("missing.yaml")


def test_overlapping_ranges_rejected() -> None:
    """eval_range starting before train_range.end is a leak and must fail."""
    raw = make_experiment_raw()
    raw["eval_range"] = {"start": "2020-01-15", "end": "2020-03-01"}
    with pytest.raises(TrainerConfigError, match="leak"):
        parse_experiment_config(raw)


def test_unknown_algorithm_rejected() -> None:
    """Unregistered algorithm names fail with the name in the message."""
    raw = make_experiment_raw()
    raw["algorithm"]["name"] = "dreamer"
    with pytest.raises(TrainerConfigError, match="dreamer"):
        parse_experiment_config(raw)


def test_unknown_network_rejected() -> None:
    """Unregistered network names fail with the name in the message."""
    raw = make_experiment_raw()
    raw["network"]["name"] = "gnn"
    with pytest.raises(TrainerConfigError, match="gnn"):
        parse_experiment_config(raw)


def test_unknown_feature_rejected() -> None:
    """Feature names outside base + registry fail with the name listed."""
    raw = make_experiment_raw()
    raw["env"]["features"]["selected"] = ["log_return", "alpha42"]
    with pytest.raises(TrainerConfigError, match="alpha42"):
        parse_experiment_config(raw)


def test_env_data_dates_forbidden() -> None:
    """Dates inside env.data conflict with range injection and must fail."""
    raw = make_experiment_raw()
    raw["env"]["data"]["start_date"] = "2019-01-01"
    with pytest.raises(TrainerConfigError, match="start_date"):
        parse_experiment_config(raw)


def test_invalid_experiment_name_rejected() -> None:
    """Experiment names are used as directory names and are restricted."""
    raw = make_experiment_raw()
    raw["experiment"] = "bad name/with slash"
    with pytest.raises(TrainerConfigError, match="experiment"):
        parse_experiment_config(raw)


def test_unknown_top_level_key_rejected() -> None:
    """Typos at the top level must be rejected."""
    raw = make_experiment_raw()
    raw["networkk"] = {"name": "mlp", "kwargs": {}}
    with pytest.raises(TrainerConfigError, match="networkk"):
        parse_experiment_config(raw)


def test_embedded_env_block_is_fully_validated() -> None:
    """Invalid env values surface via forex-env's own validation at load."""
    raw = make_experiment_raw()
    raw["env"]["environment"]["margin_call_threshold"] = 1.5
    with pytest.raises(ConfigError, match="margin_call_threshold"):
        parse_experiment_config(raw)


def test_resolve_env_raw_injects_dates_and_eval_overrides() -> None:
    """Range injection and eval overrides produce the expected env config."""
    raw = make_experiment_raw()
    config = parse_experiment_config(raw)
    train_env = resolve_env_raw(config.env, config.train_range, for_eval=False)
    assert train_env["data"]["start_date"] == "2020-01-01"
    assert train_env["environment"]["random_start"] is True

    eval_env = resolve_env_raw(config.env, config.eval_range, for_eval=True)
    assert eval_env["data"]["start_date"] == "2020-02-01"
    assert eval_env["environment"]["random_start"] is False
    assert eval_env["environment"]["episode_max_steps"] == 1_000_000
    # The original block must stay untouched (deep copy).
    assert "start_date" not in config.env["data"]
