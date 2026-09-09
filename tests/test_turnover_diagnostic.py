"""Behavioral tests for frozen turnover attribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from forex_env.accounting import PortfolioAccount
from forex_env.config import TransactionCostsConfig

from forex_trainer.supervised_portfolio import fixed_portfolio_weights
from forex_trainer.turnover_diagnostic import diagnose_fold


def _case() -> dict[str, Any]:
    symbols = tuple(f"pair{i}" for i in range(9))
    costs = TransactionCostsConfig(commission_rate=.0001, overnight_rate=0.,
                                  carry_mode="signed", spreads={p: .0002 for p in symbols})
    scores = np.array([np.arange(9), np.arange(9), np.arange(8, -1, -1)], dtype=float)
    weights = fixed_portfolio_weights(scores).astype(float)
    times = pd.date_range("2020-01-01", periods=4, tz="UTC")
    account = PortfolioAccount(1000., symbols, costs)
    predictions: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    for t in range(3):
        before = account.equity_jpy
        spread, commission = account.rebalance(before * weights[t])
        relatives = 1 + np.arange(9) * .001
        account.mark_to_market(relatives, 1.)
        price = weights[t] * (relatives - 1)
        cost = spread + commission
        gross = float(price.sum())
        after = account.equity_jpy
        ranks = np.argsort(np.argsort(scores[t], kind="stable"), kind="stable") + 1
        for p, symbol in enumerate(symbols):
            predictions.append({"decision_timestamp": times[t].isoformat(), "target_timestamp": times[t+1].isoformat(),
                                "pair": symbol, "supervised_score": scores[t, p]})
            pairs.append({"decision_timestamp": times[t].isoformat(), "pair": symbol, "predicted_rank": ranks[p],
                          "target_weight": weights[t, p], "price_simple_contribution": price[p],
                          "carry_simple_contribution": 0., "gross_simple_contribution": price[p]})
        steps.append({"decision_timestamp": times[t].isoformat(), "target_timestamp": times[t+1].isoformat(),
                      "equity_before": before, "equity_after": after, "cost_jpy": cost,
                      "weight_turnover": float(np.abs(weights[t] - (weights[t-1] if t else 0)).sum())})
        accounting.append({"decision_timestamp": times[t].isoformat(), "target_timestamp": times[t+1].isoformat(),
                           "price_simple_return": gross, "carry_simple_return": 0., "cost_ratio": cost/before,
                           "net_simple_return": after/before-1, "gross_log_return": np.log1p(gross),
                           "net_log_return": np.log(after/before)})
    return {"predictions": pd.DataFrame(predictions), "pairs": pd.DataFrame(pairs),
            "steps": pd.DataFrame(steps), "accounting": pd.DataFrame(accounting),
            "symbols": symbols, "policy": "supervised", "costs": costs}


def test_actual_trades_separate_initial_entry_drift_and_reversals() -> None:
    pairs, decisions, spells = diagnose_fold(**_case())
    assert decisions.loc[0, "weight_turnover"] == pytest.approx(3.2)
    assert decisions.loc[1, "weight_turnover"] == 0
    assert decisions.loc[1, "traded_notional_jpy"] > 0
    assert decisions.loc[1, "membership_exit_rate"] == 0
    assert decisions.loc[2, "membership_exit_rate"] == 1
    assert decisions.loc[2, "reversals"] == 4
    assert decisions.loc[1, "score_correlation"] == pytest.approx(1)
    assert decisions.loc[2, "rank_correlation"] == pytest.approx(-1)
    assert decisions.loc[2, "bottom_boundary_gap"] == 1
    for t, state in enumerate(("initial_entry", "retained", "reversal")):
        active = pairs.loc[(pairs["decision_index"] == t) & (pairs["traded_notional_jpy"] > 0)]
        assert set(active["state"]) == {state}
        assert active["cost_jpy"].sum() == pytest.approx(decisions.loc[t, "cost_jpy"])
    assert np.max(np.abs(decisions["cost_residual_ratio"])) < 1e-12
    assert sorted(spells.loc[~spells["right_censored"], "duration_decisions"]) == [2] * 4
    assert sorted(spells.loc[spells["right_censored"], "duration_decisions"]) == [1] * 4
    assert spells["duration_decisions"].sum() == 12
    np.testing.assert_allclose(pairs["signed_trade_jpy"], pairs["signed_target_change_jpy"] + pairs["signed_drift_jpy"])


def test_fold_calls_reset_holdings_and_censor_spells() -> None:
    first = diagnose_fold(**_case())
    second = diagnose_fold(**_case())
    pd.testing.assert_frame_equal(first[2], second[2])
    assert second[1].loc[0, "traded_notional_jpy"] == pytest.approx(3200.)
    assert pd.isna(second[1].loc[0, "membership_exit_rate"])


@pytest.mark.parametrize("table", ["predictions", "pairs", "steps", "accounting"])
@pytest.mark.parametrize("operation", ["missing", "duplicate", "reorder"])
def test_alignment_errors_are_not_dropped(table: str, operation: str) -> None:
    case = _case()
    frame = case[table]
    if operation == "missing":
        case[table] = frame.iloc[1:]
    elif operation == "duplicate":
        case[table] = pd.concat([frame, frame.iloc[:1]])
    else:
        case[table] = frame.iloc[::-1]
    with pytest.raises(ValueError, match="supervised"):
        diagnose_fold(**case)


@pytest.mark.parametrize("table,column", [("steps", "cost_jpy"), ("steps", "equity_before"),
    ("pairs", "price_simple_contribution"), ("pairs", "target_weight"),
    ("accounting", "cost_ratio"), ("accounting", "net_simple_return")])
def test_accounting_corruption_fails(table: str, column: str) -> None:
    case = _case()
    case[table].loc[0, column] += 1
    with pytest.raises(ValueError, match="supervised"):
        diagnose_fold(**case)


def test_target_horizon_change_fails() -> None:
    case = _case()
    case["predictions"].loc[0, "target_timestamp"] = "2020-01-04T00:00:00+00:00"
    with pytest.raises(ValueError, match="supervised"):
        diagnose_fold(**case)


def test_constant_scores_are_reported_as_undefined_not_zero() -> None:
    case = _case()
    case["predictions"].loc[:8, "supervised_score"] = 0.
    _, decisions, _ = diagnose_fold(**case)
    assert pd.isna(decisions.loc[1, "score_correlation"])
    assert decisions.loc[1, "score_correlation_status"] == "constant_input"
