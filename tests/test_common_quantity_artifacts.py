"""Audit saved synthetic accounts without consuming another fixture attempt."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.common_quantity_study import verify_fixture

DIRECTORY = Path('docs/research/results/issue42/fixture')


@pytest.fixture(scope='module')
def accounts() -> dict[str, Any]:
    verify_fixture(DIRECTORY)
    return {path.stem: json.loads(path.read_text()) for path in (DIRECTORY / 'accounts').glob('*.json')}


def test_fixture_budget_and_same_runtime(accounts: dict[str, Any]) -> None:
    assert len(accounts) == 13
    assert len({a['measurement_id'] for a in accounts.values()}) == 1
    assert len({a['runtime_sha256'] for a in accounts.values()}) == 1
    report = json.loads((DIRECTORY / 'manifest.json').read_text())
    assert report['fixture_accounts'] == 13
    assert report['market_accounts'] == report['fit_calls'] == report['ppo_training'] == 0


def test_every_trade_and_cost_reconciles(accounts: dict[str, Any]) -> None:
    for account in accounts.values():
        previous = np.zeros(9)
        equity = 1_000_000.
        for row in account['trace']:
            assert row['equity_start'] == pytest.approx(equity)
            for trade in row['trades']:
                q = np.array(trade['quantities_after'])
                notionals = np.abs(q - previous) * np.array(row['prices'])
                np.testing.assert_allclose(trade['traded_notional'], notionals, atol=1e-8)
                assert trade['spread'] == pytest.approx(notionals @ np.array(account['costs']['spreads']))
                assert trade['commission'] == pytest.approx(notionals.sum() * account['costs']['commission'])
                equity -= trade['spread'] + trade['commission']
                previous = q
            opening = previous * np.array(row['prices'])
            closing = previous * np.array(row['next_prices'])
            assert row['price_pnl'] == pytest.approx((closing - opening).sum(), abs=1e-8)
            assert row['financing'] == pytest.approx(closing @ -np.array(row['carry']) * row['elapsed_days'] / 365)
            assert row['markup'] == pytest.approx(np.abs(opening).sum() * account['costs']['markup'] * row['elapsed_days'])
            equity += row['price_pnl'] + row['financing'] - row['markup']
            assert row['equity_end'] == pytest.approx(equity)
            assert row['constraint_residual'] <= 1e-10
        assert account['final_equity'] == pytest.approx(equity)
        np.testing.assert_allclose(account['final_quantities'], previous)


def test_literal_hold_partial_close_short_and_fixed_drift(accounts: dict[str, Any]) -> None:
    rows = accounts['hold']['trace']
    assert rows[1]['literal_hold'] is True
    assert rows[1]['actual_traded_notional'] == 0
    assert rows[1]['price_pnl'] != 0 and rows[1]['markup'] > 0
    assert 0 < rows[2]['actual_traded_notional'] < rows[0]['actual_traded_notional']
    np.testing.assert_array_equal(rows[3]['quantities'], np.zeros(9))
    assert np.all(np.array(rows[4]['quantities']) < 0)
    assert rows[4]['financing'] > 0
    fixed = accounts['fixed-h1']['trace']
    assert fixed[1]['actual_traded_notional'] > 0
    assert fixed[1]['literal_hold'] is False
    assert rows[0]['elapsed_days'] == pytest.approx(71 / 24)


def test_risk_override_and_terminal_are_explicit(accounts: dict[str, Any]) -> None:
    risk = accounts['risk']['trace'][1]
    assert risk['risk_override'] is True
    assert risk['actual_traded_notional'] > 0
    assert any(t['reason'] == 'risk_drift' for t in risk['trades'])
    assert accounts['margin']['status'] == 'strategy_terminal'
    assert accounts['margin']['terminal_reason'] == 'margin_threshold'
    assert accounts['bankruptcy']['status'] == 'strategy_terminal'
    assert accounts['bankruptcy']['net_log_return'] is None
    assert accounts['bankruptcy']['terminal_reason'] == 'nonpositive_equity'
    assert accounts['bankruptcy']['remaining_decisions'] > 0


def test_shared_forecasts_and_stress_accounts(accounts: dict[str, Any]) -> None:
    for h in (1, 5):
        fixed, cost = accounts[f'fixed-h{h}'], accounts[f'cost-h{h}']
        assert [r['decision']['forecast_identity'] for r in fixed['trace']] == [r['decision']['forecast_identity'] for r in cost['trace']]
        assert all(r['decision']['forecast_identity']['horizon'] == h for r in fixed['trace'])
        assert fixed['status'] == cost['status'] == 'complete'
    normal, spread, markup = (accounts[n] for n in ('fixed-h1', 'F1', 'F2'))
    assert spread['trace'][0]['trades'][0]['spread'] == pytest.approx(2 * normal['trace'][0]['trades'][0]['spread'])
    assert markup['trace'][0]['markup'] == pytest.approx(2 * normal['trace'][0]['markup'])
    assert all(a['trace'][0]['equity_start'] == 1_000_000 for a in (normal, spread, markup))


def test_frozen_control_interfaces_use_independent_account_assets(accounts: dict[str, Any]) -> None:
    for name in ('canonical', 'ridge', 'ppo_ens3'):
        account = accounts[name]
        assert account['status'] == 'complete'
        np.testing.assert_array_equal(account['trace'][0]['assets_before'], np.zeros((9, 3)))
        assert account['policy_source']['kind'] == 'synthetic_untrained_interface_fixture'
        assert account['trace'][0]['actual_traded_notional'] > 0
        assert account['trace'][1]['assets_before'] != account['trace'][0]['assets_before']
