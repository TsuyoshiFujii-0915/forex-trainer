"""Decision and risk contracts without executing synthetic accounts."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forex_trainer.common_allocation import (
    Costs, Forecast, State, allocate, horizon_days, utility, proportional_cap,
    load_forecasts, select_candidate, Candidate,
)
from forex_trainer.common_basket import PAIRS


def state(weights: np.ndarray) -> State:
    prices = np.arange(1., 10.)
    return State('2024-03-29T00:00:00+00:00', PAIRS, 1_000_000.,
                 weights * 1_000_000. / prices, prices, np.zeros(9),
                 Costs(np.full(9, .001), .0002, .00001))


def forecast(value: float, horizon: int) -> Forecast:
    return Forecast('synthetic', horizon, '2024-03-29T00:00:00+00:00',
                    '2024-01-01T00:00:00+00:00', '2023-12-29T00:00:00+00:00',
                    '2023-06-30T23:00:00+00:00', value,
                    'basket_h_business_day_price_simple_return', PAIRS,
                    np.eye(9) * .009, 'a' * 64, 'b' * 64, 'c' * 64)


@pytest.mark.parametrize('value,u', [(1., 1.), (-1., -1.), (0., 0.)])
def test_fixed_uses_sign_equal_notional_and_shared_forecast(value: float, u: float) -> None:
    current = state(np.zeros(9))
    f = forecast(value, 1)
    fixed = allocate(current, f, 'fixed')
    cost = allocate(current, f, 'cost')
    np.testing.assert_allclose(fixed.quantities * current.prices / current.equity, u / 9)
    assert fixed.forecast_identity == cost.forecast_identity
    assert fixed.forecast_identity['bundle_sha256'] == 'a' * 64
    assert fixed.mode == 'basket_target'


def test_stationary_partial_close_and_hold() -> None:
    current = replace(state(np.zeros(9)), costs=Costs(np.zeros(9), 0., 0.))
    partial = allocate(current, forecast(.005, 1), 'cost')
    assert partial.u == pytest.approx(.5)
    invested = replace(current, quantities=np.ones(9) * current.equity / 9 / current.prices)
    close = allocate(invested, forecast(0., 1), 'cost')
    np.testing.assert_array_equal(close.quantities, np.zeros(9))
    drift = state(np.arange(1., 10.) * .001)
    held = allocate(drift, forecast(.0005, 1), 'cost')
    assert held.mode == 'hold_quantity'
    np.testing.assert_array_equal(held.quantities, drift.quantities)
    assert held.utility['entry_cost'] == 0


def test_solver_beats_dense_grid_with_drift_short_costs_and_carry() -> None:
    current = replace(state(np.linspace(-.06, .12, 9)), carry=np.linspace(-.05, .08, 9))
    f = forecast(-.004, 5)
    decision = allocate(current, f, 'cost')
    grid = [utility(current, f, np.full(9, u / 9))['total'] for u in np.linspace(-1, 1, 2001)]
    grid.append(utility(current, f, current.weights)['total'])
    assert decision.utility['total'] >= max(grid) - 1e-12
    assert decision.solver['constraint_residual'] <= 1e-10
    assert {-1., 0., 1.}.issubset(set(decision.solver['kinks']))
    assert set(9 * current.weights).issubset(set(decision.solver['kinks']))


def test_horizon_uses_actual_utc_dst_and_tail_is_not_shortened() -> None:
    assert horizon_days(forecast(.01, 1)) == pytest.approx(71 / 24)
    assert horizon_days(forecast(.01, 5)) == pytest.approx(167 / 24)
    current = replace(state(np.zeros(9)), carry=np.full(9, .0365))
    f = forecast(.02, 5)
    parts = utility(current, f, np.full(9, -1 / 9))
    assert parts['predicted_price'] == pytest.approx(-.02)
    assert parts['signed_financing'] == pytest.approx(.0365 * 167 / 24 / 365)
    assert parts['holding_markup'] == pytest.approx(.00001 * 167 / 24)
    assert parts['entry_cost'] == pytest.approx(.0012)
    assert parts['risk_penalty'] == pytest.approx(.005)
    assert parts['total'] == pytest.approx(sum([parts['predicted_price'], parts['signed_financing'],
                                              -parts['holding_markup'], -parts['entry_cost'], -parts['risk_penalty']]))


def test_ties_use_global_best_then_trade_gross_and_signed_u() -> None:
    candidates = [Candidate('basket_target', 1., np.ones(9), 1., 3., 1.),
                  Candidate('basket_target', -1., -np.ones(9), 1. - 5e-13, 2., 1.),
                  Candidate('basket_target', .5, np.ones(9), 1., 2., .5),
                  Candidate('basket_target', -.5, -np.ones(9), 1., 2., .5)]
    assert select_candidate(candidates).u == -.5
    for order in (candidates[::-1], candidates[1:] + candidates[:1]):
        assert select_candidate(order).u == -.5
    with pytest.raises(ValueError, match='candidate'):
        select_candidate([])
    with pytest.raises(ValueError, match='finite'):
        select_candidate([replace(candidates[0], total=np.nan)])


@pytest.mark.parametrize('kind', ['forecast', 'covariance', 'asof', 'covariance_asof', 'pair',
                                  'units', 'decision', 'horizon', 'price', 'carry', 'fee', 'equity'])
def test_invalid_decision_inputs_raise_with_origin(kind: str) -> None:
    current, f = state(np.zeros(9)), forecast(.01, 1)
    with pytest.raises(ValueError):
        if kind == 'forecast':
            f = replace(f, value=np.nan)
        elif kind == 'covariance':
            f = replace(f, covariance=-np.eye(9))
        elif kind == 'asof':
            f = replace(f, parameter_as_of='2025-01-01T00:00:00Z')
        elif kind == 'covariance_asof':
            f = replace(f, covariance_as_of='2025-01-01T00:00:00Z')
        elif kind == 'pair':
            f = replace(f, pairs=PAIRS[::-1])
        elif kind == 'units':
            f = replace(f, units='daily_log_return')
        elif kind == 'decision':
            current = replace(current, at='2024-04-01T23:00:00Z')
        elif kind == 'horizon':
            f = replace(f, horizon=2)
        elif kind == 'price':
            current = replace(current, prices=np.zeros(9))
        elif kind == 'carry':
            current = replace(current, carry=np.full(9, np.inf))
        elif kind == 'fee':
            current = replace(current, costs=Costs(np.full(9, -.1), 0., 0.))
        else:
            current = replace(current, equity=0.)
        allocate(current, f, 'cost')


def test_risk_reduction_includes_its_own_cost_and_has_no_fallback() -> None:
    current = state(np.array([1.2, -1.2, 1.2, -1.2, 1.2, 0., 0., 0., 0.]))
    result = proportional_cap(current, current.quantities)
    turnover = np.abs((result - current.quantities) * current.prices)
    after = current.equity - turnover @ (current.costs.spreads + current.costs.commission)
    weights = result * current.prices / after
    assert np.abs(weights).sum() == pytest.approx(5., abs=1e-10)
    assert np.max(np.abs(weights)) <= 1 + 1e-10
    np.testing.assert_allclose(result[:5] / current.quantities[:5], result[0] / current.quantities[0])
    impossible = replace(current, costs=Costs(np.full(9, 2.), 0., 0.))
    with pytest.raises(ValueError, match='feasible'):
        proportional_cap(impossible, impossible.quantities)


def test_sealed_forecasts_are_loaded_without_diagnostic_future_targets() -> None:
    bundle = Path('docs/research/results/issue41/bundle')
    rows = load_forecasts(bundle)
    assert len(rows) == 7138
    f = rows['2009', 5, next(key[2] for key in rows if key[:2] == ('2009', 5))]
    assert f.units == 'basket_h_business_day_price_simple_return'
    assert f.horizon == 5
    assert not hasattr(f, 'diagnostic_target')
    assert not f.covariance.flags.writeable
