"""Regression cases for receipt-time execution and sealed source boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forex_trainer.quote_replay import ReplayError, replay
from forex_trainer.quote_replay_policies import frozen_policies
from test_quote_replay import long_product, scenario, short_product


def test_execution_waits_for_quote_receipt(tmp_path: Path) -> None:
    raw = scenario()
    quote = raw['steps'][0]['quotes'][0]
    quote['available_at'] = quote['retrieved_at'] = '2024-03-08T22:00:01.500000Z'
    first = replay(raw, long_product, tmp_path / 'receipt')['trace'][0]
    assert first['fill_timestamp'] == quote['retrieved_at']
    assert first['quote_timestamp'] == quote['timestamp']
    assert first['gap_seconds'] == 3.5


def test_commission_uses_traded_product_price_once(tmp_path: Path) -> None:
    raw = scenario()
    raw['costs']['commission_rate'] = .001
    first = replay(raw, long_product, tmp_path / 'commission')['trace'][0]
    assert first['commission_jpy'] == 505
    assert first['spread_jpy'] == 5000
    assert first['equity_jpy'] == 1044495


def test_margin_terminal_keeps_loss_and_remaining_coverage(tmp_path: Path) -> None:
    raw = scenario()
    raw['steps'][0]['mark']['mid'][0] = 400.
    result = replay(raw, short_product, tmp_path / 'terminal')
    assert result['status'] == 'terminal_margin'
    assert len(result['trace']) == 1
    assert result['last_state']['equity_jpy'] == -505000
    assert result['coverage']['completed_steps'] == 1
    assert result['coverage']['planned_steps'] == 2


@pytest.mark.parametrize('failure', ['record_before_generate', 'late_funding_gap', 'late_funding_holding',
                                    'mark_future_release', 'quote_receipt_order', 'deadline'])
def test_temporal_failures_stop_with_a_traceable_incident(tmp_path: Path, failure: str) -> None:
    import copy
    raw = scenario()
    first = raw['steps'][0]
    if failure == 'record_before_generate':
        first['decision_recorded_at'] = first['session_close']
    elif failure == 'late_funding_gap':
        raw['funding'][0]['start'] = '2024-03-08T22:00:00Z'
    elif failure == 'late_funding_holding':
        raw['funding'][0]['end'] = '2024-03-08T22:00:02Z'
    elif failure == 'mark_future_release':
        first['mark']['available_at'] = first['mark']['retrieved_at'] = '2024-03-12T00:00:00Z'
    elif failure == 'quote_receipt_order':
        later = copy.deepcopy(first['quotes'][0])
        later['timestamp'] = later['available_at'] = later['retrieved_at'] = '2024-03-08T22:00:01.1Z'
        first['quotes'][0]['retrieved_at'] = '2024-03-08T22:00:01.5Z'
        first['quotes'].append(later)
    else:
        first['quotes'][0]['timestamp'] = first['quotes'][0]['available_at'] = first['quotes'][0]['retrieved_at'] = '2024-03-08T22:01:01Z'
    output = tmp_path / failure
    with pytest.raises(ReplayError):
        replay(raw, long_product, output)
    result = json.loads((output / 'result.json').read_text())
    assert result['incident']['origin']
    if failure in {'late_funding_holding', 'mark_future_release'}:
        assert result['last_state']['phase'] == 'fill'
        assert result['last_state']['quantity_base'][0] == 5000


def test_real_sealed_fold_loading_and_tampered_model_refusal(tmp_path: Path) -> None:
    from test_full_period import campaign_fixture
    from forex_trainer.full_period import runtime_identity
    campaign = json.loads(campaign_fixture(tmp_path).read_text())
    runtime = runtime_identity()
    manifest: dict[str, Any] = {'source':campaign['source'], 'fold':'2024',
        'runtime':{'forex_env_sha':runtime['git']['forex_env'], 'versions':runtime['versions'], 'device':'cpu'}}
    path = tmp_path / 'quote-sources.json'
    path.write_text(json.dumps(manifest))
    policies, identity = frozen_policies(path)
    assert set(policies) == {'canonical','ridge','ppo_ens3'}
    assert identity['fold'] == '2024'
    result = replay(scenario(), policies['ppo_ens3'], tmp_path / 'frozen-ppo')
    assert result['status'] == 'complete'
    model = tmp_path / '42/model_final.zip'
    model.write_bytes(model.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='hash'):
        frozen_policies(path)


def test_numeric_overflow_preserves_finite_last_fill(tmp_path: Path) -> None:
    raw = scenario()
    raw['funding'][0]['long_jpy_per_base_day'][0] = 1e308
    output = tmp_path / 'overflow'
    with pytest.raises(ReplayError, match='overflow|finite'):
        replay(raw, long_product, output)
    result = json.loads((output / 'result.json').read_text())
    assert result['last_state']['phase'] == 'fill'
    assert result['last_state']['equity_jpy'] == 995000


def test_quote_received_after_session_close_is_not_executable(tmp_path: Path) -> None:
    raw = scenario()
    raw['calendar']['sessions'][0]['close'] = '2024-03-08T22:00:01.250000Z'
    quote = raw['steps'][0]['quotes'][0]
    quote['available_at'] = quote['retrieved_at'] = '2024-03-08T22:00:01.500000Z'
    with pytest.raises(ReplayError, match='eligible quote'):
        replay(raw, long_product, tmp_path / 'closed-at-receipt')


def test_malformed_cli_input_saves_preflight_incident(tmp_path: Path) -> None:
    from forex_trainer.quote_replay import run_scenario
    raw = scenario()
    del raw['steps']
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(raw))
    output = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='steps'):
        run_scenario(path, 'fixture', None, output)
    incident = json.loads((output / 'incident.json').read_text())
    assert incident['origin'] == str(path)
    assert incident['last_state']['quantity_base'] == [0.] * 9
