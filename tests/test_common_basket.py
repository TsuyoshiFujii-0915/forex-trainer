"""Behavioral contract for the Issue 41 common-basket forecaster."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from forex_trainer.common_basket import (
    ALPHAS, FEATURES, PAIRS, InputError, build_panel, training_rows,
    standardize, fit_candidates, choose_alpha, forecast_rows, diagnostics,
)
from forex_trainer.common_basket_study import (
    FitLedger, load_contract, require_activation, verify_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/research/issue41_common_basket.json'


def market_fixture() -> pd.DataFrame:
    """Create deterministic nonconstant prices and carry without external data."""
    index = pd.bdate_range('2005-01-03', periods=800, tz='Europe/London')
    t = np.arange(len(index), dtype=float)
    values: dict[tuple[str, str], np.ndarray] = {}
    for j, pair in enumerate(PAIRS):
        close = np.exp(0.0003 * t + 0.03 * np.sin(t / (9 + j)) + j / 10)
        for field in ('Open', 'High', 'Low', 'Close'):
            values[pair, field] = close
        values[pair, 'Volume'] = np.ones(len(t))
        values[pair, 'CarryAnnual'] = 0.01 + 0.005 * np.cos(t / (27 + j))
    return pd.DataFrame(values, index=index)


def test_features_are_current_pair_means_and_future_invariant() -> None:
    market = market_fixture()
    panel = build_panel(market)
    assert panel.features.shape == (800, 5)
    i = 150
    expected = []
    for pair in PAIRS:
        close = market[pair, 'Close']
        log = np.log(close / close.shift(1))
        expected.append([log.iloc[i], log.rolling(32).std().iloc[i],
                         close.iloc[i] / close.iloc[i-19:i+1].mean() - 1,
                         np.log(close.iloc[i] / close.iloc[i-24]),
                         market[pair, 'CarryAnnual'].iloc[i]])
    np.testing.assert_allclose(panel.features[i], np.mean(expected, axis=0))
    changed = market.copy()
    changed.iloc[600:] *= 3
    future = build_panel(changed)
    np.testing.assert_allclose(panel.features[63:600], future.features[63:600])
    for h in (1, 5):
        rows = training_rows(panel, date(2005, 1, 3), panel.dates[520], h)
        other = training_rows(future, date(2005, 1, 3), panel.dates[520], h)
        np.testing.assert_array_equal(rows.features, other.features)
        np.testing.assert_array_equal(rows.pair_returns, other.pair_returns)
        np.testing.assert_allclose(rows.targets, rows.pair_returns.mean(axis=1))
        expected_target = np.mean(panel.prices[63+h] / panel.prices[63] - 1)
        assert rows.targets[0] == pytest.approx(expected_target)
        assert max(rows.target_dates) < panel.dates[520]


def test_gap_history_purge_and_business_days_are_explicit() -> None:
    market = market_fixture().drop(market_fixture().index[350])
    panel = build_panel(market)
    rows = training_rows(panel, panel.dates[0], panel.dates[520], 5)
    assert rows.counts['history_excluded'] == 126
    assert rows.counts['label_gap_excluded'] == 5
    assert rows.counts['boundary_purged'] == 5
    assert len(rows.targets) == 384
    assert len(rows.excluded) == 136
    assert rows.target_dates[0] == panel.dates[68]
    assert rows.dates[0] == panel.dates[63]
    assert all((end-start).days >= 5 for start, end in zip(rows.dates, rows.target_dates))


@pytest.mark.parametrize('kind', ['pair_order', 'duplicate', 'unsorted', 'midday', 'weekend', 'nonfinite', 'price'])
def test_bad_input_fails_with_origin(kind: str) -> None:
    market = market_fixture()
    if kind == 'pair_order':
        market = market.loc[:, list(reversed(market.columns))]
    elif kind == 'duplicate':
        market = pd.concat([market, market.iloc[-1:]])
    elif kind == 'unsorted':
        market = market.iloc[::-1]
    elif kind == 'midday':
        market.index += pd.Timedelta(hours=1)
    elif kind == 'weekend':
        market.index = pd.date_range('2005-01-01', periods=800, tz='Europe/London')
    elif kind == 'nonfinite':
        market.loc[market.index[100], (PAIRS[0], 'CarryAnnual')] = np.nan
    else:
        market.loc[market.index[100], (PAIRS[0], 'Close')] = 0
    with pytest.raises(InputError) as error:
        build_panel(market)
    assert error.value.rows
    assert str(error.value)


def test_train_only_standardization_and_constant_stop() -> None:
    panel = build_panel(market_fixture())
    rows = training_rows(panel, panel.dates[0], panel.dates[400], 1)
    mean, scale = standardize(rows.features)
    np.testing.assert_array_equal(mean, rows.features.mean(axis=0))
    np.testing.assert_array_equal(scale, rows.features.std(axis=0, ddof=0))
    constant = rows.features.copy()
    constant[:, 4] = 0.01
    with pytest.raises(InputError, match='carry_annual'):
        standardize(constant)


@pytest.fixture(scope='module')
def fitted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Spend exactly six synthetic fits and reuse their immutable models."""
    panel = build_panel(market_fixture())
    train = training_rows(panel, panel.dates[0], panel.dates[400], 1)
    validation = training_rows(panel, panel.dates[400], panel.dates[520], 1)
    path = tmp_path_factory.mktemp('basket-fits') / 'ledger.jsonl'
    ledger = FitLedger(path, 'fixture', 6)
    candidates = fit_candidates(train, validation, ledger, 'synthetic-h1')
    zero_train = replace(train, targets=np.zeros(len(train.targets)))
    zero_val = replace(validation, targets=np.zeros(len(validation.targets)))
    ties = fit_candidates(zero_train, zero_val, ledger, 'synthetic-zero')
    return dict(panel=panel, train=train, validation=validation,
                candidates=candidates, ties=ties, ledger=ledger)


def test_mean_loss_penalty_intercept_covariance_and_no_refit(fitted: dict[str, Any]) -> None:
    train = fitted['train']
    n = len(train.targets)
    for model in fitted['candidates']:
        z = (train.features - model.mean) / model.scale
        residual = model.predict(train.features) - train.targets
        np.testing.assert_allclose(z.T @ residual + n * model.alpha * model.coefficients,
                                   np.zeros(5), atol=1e-12)
        assert residual.mean() == pytest.approx(0, abs=1e-14)
        covariance = np.cov(train.pair_returns, rowvar=False, ddof=1)
        np.testing.assert_allclose(model.covariance,
                                  0.9 * covariance + 0.1 * np.diag(np.diag(covariance)) + 1e-8 * np.eye(9))
    chosen = choose_alpha(fitted['candidates'])
    assert any(chosen is model for model in fitted['candidates'])
    assert choose_alpha(fitted['ties']).alpha == 1.0
    assert fitted['ledger'].consumed == 6
    with pytest.raises(ValueError, match='budget'):
        fitted['ledger'].reserve('seventh-fit')


def test_tie_tolerance_compares_to_global_minimum(fitted: dict[str, Any]) -> None:
    models = fitted['candidates']
    near = [replace(m, validation_mse=e) for m, e in zip(models, [0, 0.9e-12, 1.8e-12])]
    assert choose_alpha(near).alpha == 0.1


def test_forecasts_keep_unlabelled_tail_and_model_cannot_use_future(fitted: dict[str, Any]) -> None:
    panel = fitted['panel']
    model = choose_alpha(fitted['candidates'])
    decisions = panel.dates[-20:-1]
    rows = forecast_rows(panel, model, decisions, 5, panel.dates[520], '2009', 'sealed-model')
    assert len(rows) == 19
    assert sum(r['diagnostic_target_status'] == 'available' for r in rows) == 15
    assert all(np.isfinite(r['predicted_basket_simple_return']) for r in rows)
    assert len({r['decision_at_utc'] for r in rows}) == len(rows)
    with pytest.raises(InputError, match='history|missing'):
        forecast_rows(panel, model, (panel.dates[20],), 5, panel.dates[0], '2009', 'sealed-model')
    with pytest.raises(InputError, match='cutoff'):
        forecast_rows(panel, model, decisions, 5, panel.dates[-1], '2009', 'sealed-model')


def test_diagnostics_never_invent_correlation() -> None:
    result = diagnostics(np.array([1., 2., 3.]), np.array([0., 0., 0.]))
    assert result['mse_minus_zero'] == 0
    assert result['correlation'] is None
    assert result['correlation_status'] == 'undefined_constant_series'
    assert result['prediction_variance'] == 0


def test_ledger_preserves_failures_and_rejects_reuse(tmp_path: Path) -> None:
    path = tmp_path / 'ledger.jsonl'
    ledger = FitLedger(path, 'regular', 1)
    ledger.reserve('fold-2009-h1-alpha0.01')
    assert ledger.consumed == 1
    with pytest.raises(FileExistsError):
        FitLedger(path, 'regular', 1)
    with pytest.raises(ValueError, match='budget|duplicate'):
        ledger.reserve('fold-2009-h1-alpha0.01')
    assert 'reserved' in path.read_text()


def test_registered_splits_and_activation_do_not_wait_for_rl() -> None:
    config, registration = load_contract(CONFIG)
    assert len(registration['folds']) == 17
    assert registration['folds'][0]['train_range_london'][1] == '2007-01-01'
    assert registration['gates']['rl_data_ready']['status'] == 'blocked'
    with pytest.raises(ValueError, match='merge|activation'):
        require_activation(config, registration, None)


def test_missing_or_tampered_bundle_cannot_reach_allocators(tmp_path: Path) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_bundle(tmp_path)


def test_cli_preflight_records_all_cases_without_fits(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys
    output = tmp_path / 'preflight'
    result = subprocess.run([sys.executable, '-m', 'forex_trainer.common_basket_study',
                             'preflight', '--config', str(CONFIG), '--output', str(output)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'input_ready'
    assert report['fit_calls'] == 0
    assert report['market_inference_calls'] == 0
    assert len(report['cases']) == 34
    assert all(c['train_rows'] >= 252 and c['validation_rows'] >= 60 for c in report['cases'])
    assert len({c['expected_decisions'] for c in report['cases']}) > 1
    assert report['execution_status'] == 'requires_merge_and_runtime_seal'
    again = subprocess.run([sys.executable, '-m', 'forex_trainer.common_basket_study',
                            'preflight', '--config', str(CONFIG), '--output', str(output)],
                           cwd=ROOT, capture_output=True, text=True)
    assert again.returncode != 0
