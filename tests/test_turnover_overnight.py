"""Distinguish environment holding fees from transaction costs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from forex_env.accounting import PortfolioAccount

from test_turnover_diagnostic import _case
from forex_trainer.turnover_diagnostic import diagnose_fold


def test_nonzero_overnight_is_separate_from_traded_notional_and_signed_carry() -> None:
    case = _case()
    case["costs"] = replace(case["costs"], overnight_rate=.0002)
    account = PortfolioAccount(1000., case["symbols"], case["costs"])
    for t in range(3):
        weights = case["pairs"].iloc[t*9:(t+1)*9].target_weight.to_numpy()
        before = account.equity_jpy
        spread, commission = account.rebalance(before * weights)
        _, overnight = account.mark_to_market(1 + np.arange(9) * .001, 1.)
        after = account.equity_jpy
        cost = spread + commission + overnight
        case["steps"].loc[t, ["equity_before", "equity_after", "cost_jpy"]] = [before, after, cost]
        case["accounting"].loc[t, ["cost_ratio", "net_simple_return", "net_log_return"]] = [cost/before, after/before-1, np.log(after/before)]
    pairs, decisions, _ = diagnose_fold(**case)
    assert decisions.loc[0, "overnight_jpy"] == pytest.approx(.64)
    assert decisions.loc[1, "weight_turnover"] == 0
    assert decisions.loc[1, "overnight_jpy"] > decisions.loc[1, "trading_cost_jpy"]
    np.testing.assert_allclose(decisions.cost_jpy, decisions.trading_cost_jpy + decisions.overnight_jpy)
    np.testing.assert_allclose(decisions.trading_cost_jpy, decisions.spread_jpy + decisions.commission_jpy)
    np.testing.assert_allclose(pairs.cost_jpy, pairs.trading_cost_jpy + pairs.overnight_jpy)
    assert decisions.carry_simple_return.sum() == 0
