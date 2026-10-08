"""Review regressions using saved artifacts, without fits, replay, or mocks."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from forex_trainer.artifact_provenance import sha256_file
from forex_trainer.common_allocation import Costs, load_forecasts
from forex_trainer.common_basket_study import digest, encode, read
from forex_trainer.common_quantity import Mark
from forex_trainer.common_quantity_study import audit_account, fixture_marks, verify_fixture

BUNDLE = Path('docs/research/results/issue41/bundle')
FIXTURE = Path('docs/research/results/issue42/fixture')


def reseal(directory: Path, hash_field: str) -> None:
    """Recompute outer hashes on a temporary artifact copy.

    Args:
        directory: Isolated copy, never the registered source.
        hash_field: Manifest content digest field for this artifact format.
    """
    manifest = read(directory / 'manifest.json')
    del manifest[hash_field]
    manifest['artifacts'] = {name: sha256_file(directory / name) for name in manifest['artifacts']}
    manifest[hash_field] = digest(manifest)
    (directory / 'manifest.json').write_bytes(encode(manifest))


@pytest.mark.parametrize('mutation', ['other_horizon', 'duplicate', 'missing', 'model_hash'])
def test_forecast_cells_cannot_redirect_or_overwrite_verified_models(tmp_path: Path, mutation: str) -> None:
    directory = tmp_path / 'bundle'
    shutil.copytree(BUNDLE, directory)
    manifest = read(directory / 'manifest.json')
    cell = next(c for c in manifest['cells'] if c['fold'] == '2009' and c['horizon'] == 1)
    if mutation == 'other_horizon':
        cell['model_path'] = 'model-2009-h5.json'
    elif mutation == 'duplicate':
        manifest['cells'].append(copy.deepcopy(cell))
    elif mutation == 'missing':
        manifest['cells'].remove(cell)
    else:
        cell['model_sha256'] = '0' * 64
    (directory / 'manifest.json').write_bytes(encode(manifest))
    reseal(directory, 'bundle_content_sha256')
    with pytest.raises(ValueError, match='model|cell'):
        load_forecasts(directory)


@pytest.mark.parametrize('mutation', [
    'truncated_complete', 'empty', 'nonpositive_complete', 'margin_complete',
    'false_terminal', 'row_terminal', 'first_decision', 'actual_last_mark',
    'planned_last_mark', 'remaining_decisions', 'time_gap', 'next_time_gap',
    'carry', 'prices', 'input_hash', 'costs', 'log_status', 'liquidation',
    'fixture_inputs',
])
def test_saved_coverage_terminal_and_inputs_are_verified(tmp_path: Path, mutation: str) -> None:
    directory = tmp_path / 'fixture'
    shutil.copytree(FIXTURE, directory)
    name = {'nonpositive_complete': 'bankruptcy', 'margin_complete': 'margin'}.get(mutation, 'hold')
    path = directory / f'accounts/{name}.json'
    account = read(path)
    if mutation == 'truncated_complete':
        account['trace'] = account['trace'][:1]
        last = account['trace'][-1]
        account['final_equity'] = last['equity_end']
        account['final_quantities'] = last['quantities']
        account['net_log_return'] = float(np.log(last['equity_end'] / 1_000_000))
    elif mutation == 'empty':
        account.update(trace=[], final_equity=1_000_000., final_quantities=[0.] * 9, net_log_return=0.)
    elif mutation in ('nonpositive_complete', 'margin_complete'):
        account.update(status='complete', terminal_reason=None, remaining_decisions=0)
        account['trace'][-1]['terminal_reason'] = None
    elif mutation == 'false_terminal':
        account.update(status='strategy_terminal', terminal_reason='margin_threshold')
    elif mutation == 'row_terminal':
        account['trace'][0]['terminal_reason'] = 'margin_threshold'
    elif mutation in ('first_decision', 'actual_last_mark', 'planned_last_mark'):
        account[mutation] = '2030-01-01T00:00:00+00:00'
    elif mutation == 'remaining_decisions':
        account['remaining_decisions'] = 4
    elif mutation == 'time_gap':
        account['trace'][1]['at'] = account['trace'][0]['at']
    elif mutation == 'next_time_gap':
        account['trace'][0]['next_at'] = account['trace'][2]['at']
    elif mutation == 'carry':
        account['trace'][1]['carry'] = list(reversed(account['trace'][0]['carry']))
    elif mutation == 'prices':
        account['trace'][0]['prices'][0] *= 2
    elif mutation == 'input_hash':
        account['input_sha256'] = '0' * 64
    elif mutation == 'costs':
        account['costs']['markup'] *= 2
    elif mutation == 'log_status':
        account['net_log_status'] = 'undefined_nonpositive_equity'
    elif mutation == 'liquidation':
        account['virtual_terminal_liquidation'] = True
    else:
        raw = read(directory / 'fixture.json')
        raw['mark_count'] = 2
        (directory / 'fixture.json').write_bytes(encode(raw))
    path.write_bytes(encode(account))
    reseal(directory, 'content_sha256')
    with pytest.raises(ValueError, match='coverage|terminal|input|accounting|reconciliation|fixture'):
        verify_fixture(directory)


def cost_terminal_ledger() -> tuple[dict[str, Any], tuple[Mark, ...], Costs]:
    """Construct a cash ledger at the margin boundary without executing it.

    Returns:
        Same-instant cost-terminal ledger, planned marks, and explicit costs.
    """
    account = read(FIXTURE / 'accounts/hold.json')
    marks = fixture_marks(read(FIXTURE / 'fixture.json'))
    row = account['trace'][0]
    quantities = np.zeros(9)
    quantities[0] = 100_000 / marks[0].prices[0]
    notional = quantities * marks[0].prices
    costs = Costs(np.full(9, 8.), 0., 0.)
    row.update(next_at=row['at'], next_prices=row['prices'], elapsed_days=0.,
               quantities=quantities.tolist(), price_pnl=0., financing=0., markup=0.,
               equity_end=200_000., terminal_reason='margin_threshold', actual_traded_notional=100_000.)
    row['trades'] = [{'reason': 'legacy_target', 'quantities_before': [0.] * 9,
                      'quantities_after': quantities.tolist(), 'traded_notional': notional.tolist(),
                      'spread': 800_000., 'commission': 0.}]
    account.update(trace=[row], costs=costs.record(), final_equity=200_000., final_quantities=quantities.tolist(),
                   net_log_return=float(np.log(.2)), net_log_status='defined', status='strategy_terminal',
                   terminal_reason='margin_threshold', actual_last_mark=row['at'], remaining_decisions=4)
    return account, marks, costs


def test_cost_only_terminal_is_accepted_at_decision_time_without_replay() -> None:
    account, marks, costs = cost_terminal_ledger()
    audit_account(account, 'cost-only-ledger', marks, costs)


@pytest.mark.parametrize('mutation', ['later_mark', 'continuation'])
def test_first_cost_terminal_cannot_advance_or_continue(mutation: str) -> None:
    account, marks, costs = cost_terminal_ledger()
    if mutation == 'later_mark':
        account['trace'][0]['next_at'] = marks[1].at
        account['actual_last_mark'] = marks[1].at
    else:
        account['trace'].append(copy.deepcopy(account['trace'][0]))
        account['remaining_decisions'] = 3
    with pytest.raises(ValueError, match='coverage|terminal'):
        audit_account(account, 'cost-only-ledger', marks, costs)
