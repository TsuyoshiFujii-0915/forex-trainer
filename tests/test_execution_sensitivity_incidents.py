"""Preserve completed fills when subsequent risk enforcement fails."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from forex_trainer.common_allocation import Costs, Decision, State
from forex_trainer.common_quantity import QuantityExecutionError, instruction
from forex_trainer.execution_sensitivity import replay_next_close
from test_execution_sensitivity import marks


def test_infeasible_post_fill_risk_keeps_the_full_fill_and_cost() -> None:
    axis = marks([100., 380.], date(2025, 1, 6))

    def leveraged(state: State, observation: dict[str, np.ndarray], window: np.ndarray) -> Decision:
        return instruction(state, np.full(9, 2_000.), 'partial')

    with pytest.raises(QuantityExecutionError, match='no feasible proportional quantity') as caught:
        replay_next_close(axis, Costs(np.full(9, .1), 0., 0.), leveraged, 'a' * 64, {})
    evidence = caught.value.evidence
    assert evidence['trace'] == []
    assert evidence['at'] == axis[1].at
    assert evidence['pending_order']['quantities'] == [2_000.] * 9
    fill = evidence['failed_fill']
    assert fill['trades'][0]['reason'] == 'frozen_next_close'
    assert fill['trades'][0]['spread'] == pytest.approx(684_000.)
    assert fill['trades'][0]['quantities_after'] == [2_000.] * 9
    assert fill['trades'][0]['at'] == axis[1].at
    assert len(fill['trades']) == 1
