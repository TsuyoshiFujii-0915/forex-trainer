"""Audit existing synthetic account records without replaying a policy."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.common_allocation import Costs

from forex_trainer.common_basket_study import read
from forex_trainer.common_direction_study import Inputs, audit_result, policy_source
from forex_trainer.common_quantity_study import fixture_marks, synthetic_forecasts


def saved_account() -> tuple[dict[str, Any], Inputs, dict[str, Any]]:
    """Adapt a saved numerical fixture to the campaign's identity envelope.

    Returns:
        Existing account cash ledger, synthetic sources, and account key.
    """
    root = Path('docs/research/results/issue42/fixture')
    raw = read(root / 'fixture.json')
    marks = fixture_marks(raw)
    forecast = synthetic_forecasts(raw, marks, 1)
    inputs = Inputs({'forecast_manifest': {'path': 'synthetic', 'sha256': '0' * 64}}, {}, [],
                    {('synthetic', 1, at): f for at, f in forecast.items()},
                    {'synthetic': marks}, {'synthetic': {'kind': 'synthetic'}}, {})
    cell = {'fold': 'synthetic', 'scenario': 'F0', 'policy': 'A1-fixed'}
    account = read(root / 'accounts/fixed-h1.json')
    account['policy_source'] = policy_source(inputs, cell)
    return account, inputs, cell


def test_saved_synthetic_ledger_passes_campaign_audit_without_replay() -> None:
    account, inputs, cell = saved_account()
    audit_result(account, inputs, cell, account['runtime_sha256'], fixture_costs())


@pytest.mark.parametrize('field', ['forecast', 'state', 'hold', 'override', 'exposure', 'source', 'runtime'])
def test_campaign_detects_identity_or_trace_changes(field: str) -> None:
    account, inputs, cell = saved_account()
    changed = copy.deepcopy(account)
    row = changed['trace'][0]
    if field == 'forecast':
        row['decision']['forecast_identity']['model_sha256'] = 'f' * 64
    elif field == 'state':
        row['decision']['state_sha256'] = 'f' * 64
    elif field == 'hold':
        row['literal_hold'] = True
    elif field == 'override':
        row['risk_override'] = True
    elif field == 'exposure':
        row['execution_weights_after_cost'][0] += .1
    elif field == 'source':
        changed['policy_source']['training_lineage_changed'] = True
    else:
        changed['runtime_sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        audit_result(changed, inputs, cell, account['runtime_sha256'], fixture_costs())


def fixture_costs() -> Costs:
    """Read the synthetic fixture's independent registered cost treatment.

    Returns:
        Explicit fixture rates, which differ from market F0 rates.
    """
    raw = read(Path('docs/research/results/issue42/fixture/fixture.json'))
    return Costs(np.array(raw['spreads']), raw['commission'], raw['markup'])
