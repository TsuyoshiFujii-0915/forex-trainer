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


def test_missing_rank_allocation_rejected() -> None:
    """run.rank_allocation is required for reproducible action semantics."""
    raw = make_experiment_raw()
    del raw["run"]["rank_allocation"]
    with pytest.raises(TrainerConfigError, match="rank_allocation"):
        parse_experiment_config(raw)


def test_rank_allocation_mapping_accepted() -> None:
    """A valid sparse rank allocation parses into the typed config."""
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = [
        "JPY/USD",
        "JPY/EUR",
        "JPY/GBP",
        "JPY/AUD",
    ]
    raw["env"]["transaction_costs"]["spreads"] = {
        pair: 0.0 for pair in raw["env"]["environment"]["currency_pairs"]
    }
    raw["run"]["rank_allocation"] = {"top_k": 2, "gross_exposure": 2.0}
    config = parse_experiment_config(raw)
    assert config.run.rank_allocation is not None
    assert config.run.rank_allocation.top_k == 2
    assert config.run.rank_allocation.gross_exposure == 2.0


@pytest.mark.parametrize(
    ("top_k", "gross_exposure", "max_leverage", "message"),
    [
        (0, 1.0, 4.0, "top_k"),
        (1, 0.0, 4.0, "gross_exposure"),
        (1, float("nan"), 4.0, "finite"),
        (1, 2.1, 4.0, "weight"),
        (2, 2.0, 1.0, "max_leverage"),
    ],
)
def test_invalid_rank_allocation_rejected(
    top_k: int, gross_exposure: float, max_leverage: float, message: str
) -> None:
    """Rank settings that cannot preserve fixed gross exposure fail early."""
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = [
        "JPY/USD",
        "JPY/EUR",
        "JPY/GBP",
        "JPY/AUD",
    ]
    raw["env"]["transaction_costs"]["spreads"] = {
        pair: 0.0 for pair in raw["env"]["environment"]["currency_pairs"]
    }
    raw["env"]["environment"]["max_leverage"] = max_leverage
    raw["run"]["rank_allocation"] = {
        "top_k": top_k,
        "gross_exposure": gross_exposure,
    }
    with pytest.raises(TrainerConfigError, match=message):
        parse_experiment_config(raw)


def test_rank_allocation_rejects_overlapping_tails() -> None:
    """Long and short tails must fit without selecting a pair twice."""
    raw = make_experiment_raw()
    raw["run"]["rank_allocation"] = {"top_k": 1, "gross_exposure": 1.0}
    with pytest.raises(TrainerConfigError, match="currency_pairs"):
        parse_experiment_config(raw)


def test_rank_allocation_rejects_agent_controlled_leverage() -> None:
    """Fixed gross allocation requires leverage to remain pinned."""
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = ["JPY/USD", "JPY/EUR"]
    raw["env"]["transaction_costs"]["spreads"] = {
        "JPY/USD": 0.0,
        "JPY/EUR": 0.0,
    }
    raw["env"]["environment"]["allow_action_leverage"] = True
    raw["run"]["rank_allocation"] = {"top_k": 1, "gross_exposure": 1.0}
    with pytest.raises(TrainerConfigError, match="allow_action_leverage"):
        parse_experiment_config(raw)


def test_rank_allocation_and_residual_are_mutually_exclusive() -> None:
    """Two incompatible action transformations cannot be enabled together."""
    raw = make_experiment_raw()
    raw["env"]["environment"]["currency_pairs"] = ["JPY/USD", "JPY/EUR"]
    raw["env"]["transaction_costs"]["spreads"] = {
        "JPY/USD": 0.0,
        "JPY/EUR": 0.0,
    }
    raw["env"]["features"]["normalize"] = False
    raw["env"]["features"]["selected"] = ["log_return", "volatility", "mom24"]
    raw["run"]["residual"] = {
        "feature": "mom24",
        "top_k": 1,
        "base_size": 0.8,
        "scale": 0.3,
    }
    raw["run"]["rank_allocation"] = {"top_k": 1, "gross_exposure": 1.0}
    with pytest.raises(TrainerConfigError, match="cannot both"):
        parse_experiment_config(raw)
