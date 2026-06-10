"""Tests for the feature registry: causality and env compatibility.

Every registered feature is automatically checked for lookahead (extending
the dataset into the future must not change past values) and for finiteness
inside the environment after warmup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from forex_env import ForexEnv, parse_config
from forex_env.data.synthetic import SyntheticDataProvider
from forex_env.features import BASE_FEATURE_NAMES

from forex_trainer.config import parse_experiment_config, resolve_env_raw
from forex_trainer.features import FEATURE_REGISTRY
from helpers import make_experiment_raw

_PREFIX_ROWS = 300


def _per_pair_frame(end_date: str) -> pd.DataFrame:
    """Build a per-pair OHLCV frame from synthetic data.

    Args:
        end_date: End date of the generated range.

    Returns:
        Plain-column OHLCV DataFrame for one pair.
    """
    data = SyntheticDataProvider(seed=5).get_data(
        ("JPY/USD",), "2020-01-01", end_date, "1h"
    )
    return data.xs("JPY/USD", axis=1, level=0)


@pytest.mark.parametrize("name", sorted(FEATURE_REGISTRY))
def test_feature_has_no_lookahead(name: str) -> None:
    """Past feature values must not change when future data is appended.

    This catches any accidental use of future rows (e.g. shift(-1) or
    centered windows) in a registered feature.
    """
    long_frame = _per_pair_frame("2020-03-01")
    short_frame = long_frame.iloc[:_PREFIX_ROWS]
    function = FEATURE_REGISTRY[name]
    on_long = function(long_frame).iloc[:_PREFIX_ROWS]
    on_short = function(short_frame)
    pd.testing.assert_series_equal(on_long, on_short, check_names=False)


@pytest.mark.parametrize("name", sorted(FEATURE_REGISTRY))
def test_feature_is_finite_inside_env(name: str) -> None:
    """Each registered feature must survive the env's warmup finiteness check."""
    raw = make_experiment_raw()
    raw["env"]["features"]["selected"] = ["log_return", "volatility", name]
    config = parse_experiment_config(raw)
    env_raw = resolve_env_raw(config.env, config.train_range, for_eval=False)
    env = ForexEnv(
        parse_config(env_raw), custom_features={name: FEATURE_REGISTRY[name]}
    )
    obs, _ = env.reset(seed=0)
    assert obs["market"].shape[2] == 3
    assert np.isfinite(obs["market"]).all()


def test_registry_does_not_shadow_base_features() -> None:
    """Registry names must not collide with the env's base features."""
    assert not set(FEATURE_REGISTRY) & set(BASE_FEATURE_NAMES)
