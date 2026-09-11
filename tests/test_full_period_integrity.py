"""Additional fail-fast and terminal-result acceptance tests for Issue #28."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forex_trainer.full_period import (
    FrozenRidge, build_period_env, evaluate_policy, prepare_period, run_period_campaign,
)
from test_full_period import (
    config_raw, constant_policy, market_calendar, ridge_record, sealed_fixture,
)


@pytest.mark.parametrize("name", ["42/model_final.zip", "42/meta.json", "calendar.json", "models.json"])
def test_tampered_frozen_files_abort_before_writing(sealed_fixture: Path, tmp_path: Path, name: str) -> None:
    path = sealed_fixture.parent / name
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\ncorrupt")
        with pytest.raises(ValueError, match="hash mismatch"):
            run_period_campaign(sealed_fixture, tmp_path / "result")
        assert not (tmp_path / "result").exists()
    finally:
        path.write_bytes(original)


def test_bankrupt_account_is_saved_as_terminal_not_a_data_error(tmp_path: Path) -> None:
    market, calendar = market_calendar("2023-11-30", 70)
    raw = config_raw()["env"]
    first_pair = calendar["symbols"][0]
    raw["transaction_costs"]["spreads"][first_pair] = 0.0
    market.loc[market.index[64]:, (first_pair, "CarryAnnual")] = 0.0
    for field in ("Open", "High", "Low", "Close"):
        market.loc[market.index[64]:, (first_pair, field)] *= 100.0
    plan = prepare_period(market, calendar, calendar["sessions"][63]["session_close"], "2024-04-01T00:00:00Z", tuple(calendar["symbols"]), tmp_path / "source.parquet")
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")

    def short_policy(observation: dict[str, np.ndarray], window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lose more than the initial equity on a real positive price move."""
        scores, action = constant_policy(observation, window)
        return -scores, -action

    result = evaluate_policy(env, windows, plan, short_policy)
    env.close()
    assert result["status"] == "incomplete_margin_call"
    assert result["metrics"]["terminated_by_margin_call"] is True
    assert result["trace"][-1]["equity_jpy"] < 0
    assert result["metrics"]["annualized_gross_return"] is None
    assert result["metrics"]["undefined_reason"] == "non_positive_terminal_equity"
    json.dumps(result, allow_nan=False)


def test_frozen_ridge_matches_original_double_precision_prediction(tmp_path: Path) -> None:
    from forex_trainer.supervised_ranking import RidgeModel, Standardizer, apply_standardizer
    record = ridge_record()
    window = np.arange(9 * 32 * 8, dtype=np.float64).reshape(9, 32, 8) * 0.001 + 1e-10
    ridge = FrozenRidge(record, tmp_path / "models.json")
    standardizer = Standardizer(np.array(record["train_mean"]), np.array(record["train_scale"]))
    model = RidgeModel(np.array(record["coefficients"]), record["intercept"], record["selected_alpha"])
    expected = model.predict(apply_standardizer(window.reshape(1, 9, 256), standardizer))[0]
    np.testing.assert_array_equal(ridge.predict(window), expected)


def test_all_accounting_components_reconcile_and_dst_uses_utc_days(tmp_path: Path) -> None:
    market, calendar = market_calendar("2023-11-30", 80)
    raw = config_raw()["env"]
    plan = prepare_period(market, calendar, calendar["sessions"][63]["session_close"], "2024-04-01T00:00:00Z", tuple(calendar["symbols"]), tmp_path / "source.parquet")
    env, windows = build_period_env(raw, plan, tmp_path / "slice.parquet")
    result = evaluate_policy(env, windows, plan, constant_policy)
    env.close()
    previous_exposure = np.zeros(9)
    for row in result["trace"]:
        decision, target = pd.Timestamp(row["decision_timestamp"]), pd.Timestamp(row["target_timestamp"])
        days = (target - decision).total_seconds() / 86400
        weights = np.array(row["target_weights"])
        exposure = weights * row["equity_before"]
        relatives = np.array([plan.market.loc[target, (symbol, "Close")] / plan.market.loc[decision, (symbol, "Close")] for symbol in plan.symbols])
        turnover = np.abs(exposure - previous_exposure)
        spread = sum(turnover[i] * raw["transaction_costs"]["spreads"][symbol] for i, symbol in enumerate(plan.symbols))
        assert row["spread_jpy"] == pytest.approx(spread)
        assert row["overnight_jpy"] == pytest.approx(np.abs(exposure).sum() * 2e-5 * days)
        pnl = float(exposure @ (relatives - 1))
        assert row["equity_jpy"] == pytest.approx(row["equity_before"] + pnl + row["financing_jpy"] - row["cost_jpy"])
        previous_exposure = exposure * relatives


def test_ppo_trace_receives_its_own_previous_assets(sealed_fixture: Path, tmp_path: Path) -> None:
    result = run_period_campaign(sealed_fixture, tmp_path / "result")
    trace = result["folds"]["2024"]["ppo_ens3"]["trace"]
    assert np.any(np.array(trace[1]["assets_before"]))
    np.testing.assert_array_equal(np.array(trace[1]["assets_before"])[:, 0], np.array(trace[0]["target_weights"], dtype=np.float32))
