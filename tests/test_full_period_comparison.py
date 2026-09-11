"""Comparison must bind market inputs and non-cost environment settings."""

from pathlib import Path

import pytest

from forex_trainer.full_period import build_period_env, evaluate_policy, prepare_period, require_comparable
from test_full_period import config_raw, constant_policy, market_calendar


@pytest.mark.parametrize("change", ["market", "risk"])
def test_comparison_rejects_different_measurement_inputs(tmp_path: Path, change: str) -> None:
    market, calendar = market_calendar("2023-11-30", 70)
    raw = config_raw()["env"]
    results = []
    for index in range(2):
        plan = prepare_period(market, calendar, calendar["sessions"][63]["session_close"], "2024-04-01T00:00:00Z", tuple(calendar["symbols"]), tmp_path / "raw.parquet")
        env, windows = build_period_env(raw, plan, tmp_path / f"slice{index}.parquet")
        results.append(evaluate_policy(env, windows, plan, constant_policy))
        env.close()
        if change == "market":
            market.iloc[64:, :] *= 1.001
        else:
            raw["environment"]["margin_call_threshold"] = 0.3
    with pytest.raises(ValueError, match="market|environment"):
        require_comparable(results)
