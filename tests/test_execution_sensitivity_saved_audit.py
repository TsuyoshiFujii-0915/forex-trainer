"""Audit the original failed market trace without policy inference or replay."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forex_trainer.common_basket_study import read
from forex_trainer.common_quantity import Mark
from forex_trainer.common_quantity_study import audit_account
from forex_trainer.execution_sensitivity import audit_proxy, input_identity
from forex_trainer.execution_sensitivity_study import CONFIG, Inputs, load_inputs, scenario_costs

SAVED = Path('docs/research/results/issue45/campaign/accounts')


@pytest.fixture(scope='module')
def original_inputs() -> Inputs:
    """Load the sealed inputs without invoking any policy.

    Returns:
        Original market marks and frozen model identities.
    """
    return load_inputs(CONFIG)


@pytest.fixture
def failed_prefix(original_inputs: Inputs) -> tuple[dict[str, Any], tuple[Mark, ...]]:
    """Restrict the failed trace to its first four actual transitions.

    Args:
        original_inputs: Original sealed market axis.

    Returns:
        A complete four-transition audit fixture ending at the first failure.
    """
    result = read(SAVED / '2009-F0-canonical-next_close.json.gz')
    result['trace'] = result['trace'][:4]
    marks = original_inputs.marks['2009'][:5]
    last = result['trace'][-1]
    result.update(planned_last_mark=marks[-1].at, actual_last_mark=marks[-1].at,
                  remaining_decisions=0, final_equity=last['equity_end'],
                  final_quantities=last['quantities'],
                  net_log_return=float(np.log(last['equity_end'] / 1_000_000.)),
                  input_sha256=input_identity(marks))
    return result, marks


def test_actual_one_ulp_prefix_passes_without_changing_saved_values(
    failed_prefix: tuple[dict[str, Any], tuple[Mark, ...]],
) -> None:
    result, marks = failed_prefix
    before = deepcopy(result)
    assert result['trace'][-1]['equity_at_decision'] == 999390.5382721465
    audit_proxy(result, marks, scenario_costs('F0'), 'original-four-transition-prefix')
    assert result == before


@pytest.mark.parametrize('field, error', [
    ('equity_at_decision', 'decision.equity'), ('price_pnl', 'price_pnl'),
    ('financing', 'financing'), ('markup', 'markup'), ('spread', 'fill.spread'),
    ('decision_hash', 'stale or foreign decision'), ('order_quantity', 'frozen order'),
    ('order_time', 'frozen order'),
])
def test_saved_prefix_rejects_corrupted_cash_identity_and_orders(
    failed_prefix: tuple[dict[str, Any], tuple[Mark, ...]], field: str, error: str,
) -> None:
    result, marks = failed_prefix
    row = result['trace'][-1]
    if field in ('equity_at_decision', 'price_pnl', 'financing', 'markup'):
        row[field] += .01
    elif field == 'spread':
        row['trades'][-1]['spread'] += .01
        row['orders'][0]['trades'][-1]['spread'] += .01
    elif field == 'decision_hash':
        row['decision']['state_sha256'] = '0' * 64
    elif field == 'order_quantity':
        row['orders'][0]['quantities'][0] += 1.
    else:
        row['orders'][0]['expected_fill_at'] = marks[-1].at.replace('23:00', '22:00')
    with pytest.raises(ValueError, match=error):
        audit_proxy(result, marks, scenario_costs('F0'), 'corrupted-prefix')


@pytest.mark.parametrize('execution', ['same_close', 'next_close'])
def test_all_171_saved_transitions_pass_without_reexecution(original_inputs: Inputs, execution: str) -> None:
    path = SAVED / f'2009-F0-canonical-{execution}.json.gz'
    original_bytes = path.read_bytes()
    result = read(path)
    assert len(result['trace']) == 171 and result['remaining_decisions'] == 0
    if execution == 'same_close':
        audit_account(result, str(path), original_inputs.marks['2009'], scenario_costs('F0'))
    else:
        audit_proxy(result, original_inputs.marks['2009'], scenario_costs('F0'), str(path))
    assert path.read_bytes() == original_bytes
