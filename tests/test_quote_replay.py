"""Causal quote replay behavior, including independent real PPO inference."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.quote_replay import ReplayError, replay, run_scenario
from forex_trainer.quote_replay_policies import fixture_policies

FIXTURE = Path(__file__).parent / 'fixtures/issue31/scenario.json'


def scenario() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def long_product(observation: dict[str, np.ndarray], window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.zeros((9, 1), dtype=np.float32)
    action[0, 0] = -0.5
    return action[:, 0].astype(float), action


def short_product(observation: dict[str, np.ndarray], window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores, action = long_product(observation, window)
    return -scores, -action


def test_entry_gap_quantity_jpy_pnl_and_final_mark(tmp_path: Path) -> None:
    result = replay(scenario(), long_product, tmp_path / 'account')
    first, second = result['trace']
    assert first['quantity_base'][0] == 5000
    assert first['spread_jpy'] == 5000
    assert first['gap_pnl_jpy'] == 0
    assert first['holding_pnl_jpy'] == 50000
    assert first['equity_jpy'] == 1045000
    assert second['gap_pnl_jpy'] == 10000
    assert second['fill_equity_jpy'] == 1055000
    assert second['quantity_base'][0] == 4709
    assert second['spread_jpy'] == 291
    assert second['equity_jpy'] == 1092381
    assert result['last_state']['quantity_base'][0] == 4709
    assert len(list((tmp_path / 'account').glob('decision-*.json'))) == 2
    assert result['coverage']['elapsed_seconds'] == 96 * 3600 - 3600
    assert first['holding_seconds'] == 71 * 3600 - 3
    assert first['quote_to_jpy'] == [1.] * 9


def test_short_product_is_long_model_coordinate(tmp_path: Path) -> None:
    result = replay(scenario(), short_product, tmp_path / 'short')
    first = result['trace'][0]
    assert first['quantity_base'][0] == -5000
    assert first['fill_prices'][0] == 99
    assert first['holding_pnl_jpy'] == -50000
    assert first['equity_jpy'] == 945000
    assert first['effective_weights'][0] == 0.5


@pytest.mark.parametrize('failure', ['early_decision','late_input','early_quote','stale','missing',
    'holiday','closed','depth','funding','pair','double_spread','double_markup','instrument',
    'minimum','margin','naive','unknown','window_hash','quote_available','order','funding_late'])
def test_invalid_input_has_incident_and_no_fabricated_fill(tmp_path: Path, failure: str) -> None:
    raw = scenario()
    step = raw['steps'][0]
    if failure == 'early_decision':
        step['decision_generated_at'] = '2024-03-08T21:59:57Z'
    elif failure == 'late_input':
        step['inputs'][0]['retrieved_at'] = '2024-03-08T22:00:01Z'
    elif failure == 'early_quote':
        step['quotes'][0]['timestamp'] = step['decision_recorded_at']
    elif failure == 'stale':
        step['quotes'][0]['retrieved_at'] = '2024-03-08T22:00:03Z'
    elif failure == 'missing':
        step['quotes'] = []
    elif failure == 'holiday':
        raw['calendar']['holidays'].append('2024-03-08')
    elif failure == 'closed':
        raw['calendar']['sessions'][0]['close'] = '2024-03-08T21:00:00Z'
    elif failure == 'depth':
        step['quotes'][0]['capacity_base'][0] = 1.
    elif failure == 'funding':
        raw['funding'] = []
    elif failure == 'pair':
        step['quotes'][0]['bid'].pop()
    elif failure == 'double_spread':
        raw['costs']['fixed_spread_debit'] = 0.01
    elif failure == 'double_markup':
        raw['costs']['swap_includes_markup'] = True
        raw['costs']['markup_per_base_day'][0] = 0.01
    elif failure == 'instrument':
        raw['instruments'][0]['quote'] = 'USD'
    elif failure == 'minimum':
        raw['instruments'][0]['minimum_trade'] = 10000.
    elif failure == 'margin':
        raw['instruments'][0]['margin_rate'] = 3.
    elif failure == 'naive':
        step['decision_recorded_at'] = '2024-03-08T22:00:00'
    elif failure == 'unknown':
        raw['partial_fill'] = True
    elif failure == 'window_hash':
        step['window'][0][0][0] += 1.
    elif failure == 'quote_available':
        step['quotes'][0]['available_at'] = '2024-03-08T22:00:02Z'
    elif failure == 'order':
        step['quotes'] = [copy.deepcopy(step['quotes'][0])] * 2
    elif failure == 'funding_late':
        raw['funding'][0]['available_at'] = '2024-03-08T22:00:02Z'
        raw['funding'][0]['retrieved_at'] = '2024-03-08T22:00:02Z'
    output = tmp_path / failure
    with pytest.raises(ReplayError, match='scenario'):
        replay(raw, long_product, output)
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'incident'
    assert not result['trace']
    assert result['last_state']['quantity_base'] == [0.] * 9
    assert result['incident']['origin']


def test_skip_ineligible_quote_then_use_first_even_when_depth_fails(tmp_path: Path) -> None:
    raw = scenario()
    eligible = raw['steps'][0]['quotes'][0]
    early = copy.deepcopy(eligible)
    early['timestamp'] = '2024-03-08T21:59:59Z'
    early['available_at'] = early['retrieved_at'] = early['timestamp']
    later = copy.deepcopy(eligible)
    later['timestamp'] = later['available_at'] = later['retrieved_at'] = '2024-03-08T22:00:02Z'
    raw['steps'][0]['quotes'] = [early, eligible, later]
    result = replay(raw, long_product, tmp_path / 'valid')
    assert result['trace'][0]['fill_timestamp'] == eligible['timestamp']
    assert result['trace'][0]['rejected_quotes'][0]['index'] == 0
    eligible['capacity_base'][0] = 1.
    with pytest.raises(ReplayError, match='capacity'):
        replay(raw, long_product, tmp_path / 'insufficient')


def test_funding_integrates_gap_and_holding_using_product_direction(tmp_path: Path) -> None:
    raw = scenario()
    raw['funding'][0]['long_jpy_per_base_day'][0] = 0.1
    raw['funding'][0]['short_jpy_per_base_day'][0] = -0.2
    raw['costs']['markup_per_base_day'][0] = 0.01
    first = replay(raw, long_product, tmp_path / 'long')['trace'][0]
    days = (71 * 3600 - 3) / 86400
    assert first['holding_financing_jpy'] == pytest.approx(5000 * 0.1 * days)
    assert first['holding_markup_jpy'] == pytest.approx(5000 * 0.01 * days)
    short = replay(raw, short_product, tmp_path / 'short')['trace'][0]
    assert short['holding_financing_jpy'] == pytest.approx(-5000 * 0.2 * days)
    second = replay(raw, long_product, tmp_path / 'again')['trace'][1]
    assert second['gap_financing_jpy'] == pytest.approx(5000 * 0.1 * 3 / 86400)


def test_missing_later_quote_preserves_previous_account_and_saved_decision(tmp_path: Path) -> None:
    raw = scenario()
    raw['steps'][1]['quotes'] = []
    with pytest.raises(ReplayError):
        replay(raw, long_product, tmp_path / 'account')
    result = json.loads((tmp_path / 'account/result.json').read_text())
    assert len(result['trace']) == 1
    assert result['last_state']['equity_jpy'] == 1045000
    assert result['last_state']['quantity_base'][0] == 5000
    assert (tmp_path / 'account/decision-0001.json').is_file()


def test_bad_final_mark_preserves_actual_fill_state(tmp_path: Path) -> None:
    raw = scenario()
    raw['steps'][0]['mark']['timestamp'] = '2024-03-08T21:00:00Z'
    with pytest.raises(ReplayError):
        replay(raw, long_product, tmp_path / 'account')
    result = json.loads((tmp_path / 'account/result.json').read_text())
    assert result['last_state']['quantity_base'][0] == 5000
    assert result['last_state']['equity_jpy'] == 995000
    assert result['last_state']['phase'] == 'fill'


def test_real_ppo_reinference_and_three_independent_accounts(tmp_path: Path) -> None:
    policies, identity = fixture_policies()
    raw = scenario()
    result = {name: replay(raw, policy, tmp_path / name) for name, policy in policies.items()}
    assert set(result) == {'canonical', 'ridge', 'ppo_ens3'}
    assert identity['ppo_kind'] == 'untrained_fixture_only'
    assert len({json.dumps(r['trace'][1]['assets_before']) for r in result.values()}) == 3
    ppo = result['ppo_ens3']
    obs = {'market': np.array(raw['steps'][1]['window'], dtype=np.float32),
           'assets': np.array(ppo['trace'][1]['assets_before'], dtype=np.float32)}
    window = np.array(raw['steps'][1]['window'], dtype=np.float64)
    _, expected = policies['ppo_ens3'](obs, window)
    np.testing.assert_array_equal(ppo['trace'][1]['action'], expected[:, 0])
    obs['assets'][:] = 0
    _, reset_action = policies['ppo_ens3'](obs, window)
    assert not np.array_equal(expected, reset_action)
    assert all(r['coverage'] == ppo['coverage'] for r in result.values())
    for r in result.values():
        assert r['trace'][1]['assets_before'][0][2] == pytest.approx(
            r['trace'][0]['price_pnl_by_pair'][0] / 1000000, abs=1e-7)


def test_explicit_scenario_cli_seals_provenance_and_rejects_overwrite(tmp_path: Path) -> None:
    output = tmp_path / 'campaign'
    result = run_scenario(FIXTURE, 'fixture', None, output)
    assert result['status'] == 'complete'
    provenance = json.loads((output / 'provenance.json').read_text())
    from forex_trainer.artifact_provenance import sha256_file
    assert provenance['input_sha256'] == sha256_file(FIXTURE)
    for name, digest in provenance['artifacts'].items():
        assert sha256_file(output / name) == digest
    assert provenance['economic_validation'] == 'unverified'
    with pytest.raises(ValueError, match='exists'):
        run_scenario(FIXTURE, 'fixture', None, output)
    with pytest.raises(ValueError, match='scenario|evidence'):
        run_scenario(FIXTURE, 'shadow_quote', None, tmp_path / 'bad')
