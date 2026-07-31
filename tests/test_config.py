"""Tests for fail-fast experiment configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from forex_env.errors import ConfigError
from helpers import make_experiment_raw, write_experiment_yaml

from forex_trainer.config import (
    TrainerConfigError,
    load_experiment_config,
    parse_experiment_config,
    resolve_env_raw,
)


def test_valid_config_parses(tmp_path) -> None:
    """A complete experiment config loads into typed dataclasses."""
    path = write_experiment_yaml(tmp_path, make_experiment_raw())
    config, raw = load_experiment_config(path)
    assert config.experiment == "test_exp"
    assert config.algorithm.name == "ppo"
    assert config.network.name == "mlp"
    assert config.custom_feature_names == ("rsi14",)
    assert raw["experiment"] == "test_exp"


def test_readme_experiment_yaml_parses() -> None:
    """The canonical public experiment example is a complete valid config."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    experiment_section = readme.split("## Experiment YAML", maxsplit=1)[1]
    yaml_source = experiment_section.split("```yaml", maxsplit=1)[1].split(
        "```", maxsplit=1
    )[0]
    raw = yaml.safe_load(yaml_source)

    config = parse_experiment_config(raw)

    assert config.experiment == "ppo_mlp_daily"


def test_missing_file_raises() -> None:
    """A nonexistent config path raises with the path in the message."""
    with pytest.raises(TrainerConfigError, match="missing.yaml"):
        load_experiment_config("missing.yaml")


def test_overlapping_ranges_rejected() -> None:
    """eval_range starting before val_range.end is a leak and must fail."""
    raw = make_experiment_raw()
    raw["eval_range"] = {"start": "2020-01-15", "end": "2020-03-01"}
    with pytest.raises(TrainerConfigError, match="leak"):
        parse_experiment_config(raw)


def test_val_range_overlapping_train_rejected() -> None:
    """val_range starting before train_range.end is a leak and must fail."""
    raw = make_experiment_raw()
    raw["val_range"] = {"start": "2020-01-15", "end": "2020-02-15"}
    with pytest.raises(TrainerConfigError, match="leak"):
        parse_experiment_config(raw)


def test_missing_val_range_rejected() -> None:
    """val_range is required for model selection (ADR-0005)."""
    raw = make_experiment_raw()
    del raw["val_range"]
    with pytest.raises(TrainerConfigError, match="val_range"):
        parse_experiment_config(raw)


def test_missing_decision_interval_rejected() -> None:
    """run.decision_interval is required (ADR-0004)."""
    raw = make_experiment_raw()
    del raw["run"]["decision_interval"]
    with pytest.raises(TrainerConfigError, match="decision_interval"):
        parse_experiment_config(raw)


def test_non_positive_decision_interval_rejected() -> None:
    """run.decision_interval must be a positive integer."""
    raw = make_experiment_raw()
    raw["run"]["decision_interval"] = 0
    with pytest.raises(TrainerConfigError, match="decision_interval"):
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
    assert eval_env["data"]["start_date"] == "2020-02-15"
    assert eval_env["environment"]["random_start"] is False
    assert eval_env["environment"]["episode_max_steps"] == 1_000_000
    # The original block must stay untouched (deep copy).
    assert "start_date" not in config.env["data"]


def test_missing_residual_rejected() -> None:
    """run.residual is required (ADR-0009)."""
    raw = make_experiment_raw()
    del raw["run"]["residual"]
    with pytest.raises(TrainerConfigError, match="residual"):
        parse_experiment_config(raw)


def test_residual_unknown_feature_rejected() -> None:
    """residual.feature must be one of the selected features."""
    raw = make_experiment_raw()
    raw["run"]["residual"] = {
        "feature": "mom24",
        "top_k": 2,
        "base_size": 0.8,
        "scale": 0.3,
    }
    with pytest.raises(TrainerConfigError, match="mom24"):
        parse_experiment_config(raw)


def test_residual_mapping_accepted() -> None:
    """A valid residual mapping parses into the typed config."""
    raw = make_experiment_raw()
    raw["env"]["features"]["normalize"] = False
    raw["env"]["features"]["selected"] = ["log_return", "volatility", "mom24"]
    raw["run"]["residual"] = {
        "feature": "mom24",
        "top_k": 2,
        "base_size": 0.8,
        "scale": 0.3,
    }
    config = parse_experiment_config(raw)
    assert config.run.residual is not None
    assert config.run.residual.feature == "mom24"
