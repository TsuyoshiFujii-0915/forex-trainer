"""The shared one-step engine preserves the registered replay accounting."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np

from forex_trainer.quote_replay import execute_quote, initial_state, replay
from test_quote_replay import long_product, scenario


def test_shared_step_matches_existing_replay(tmp_path: Path) -> None:
    raw = scenario()
    expected = replay(raw, long_product, tmp_path / 'legacy')
    state = initial_state()
    state.update(timestamp=raw['initial_mark']['timestamp'], mid=raw['initial_mark']['mid'])
    fills: list[dict[str, Any]] = []

    def save_fill(value: dict[str, Any]) -> None:
        fills.append(value)

    for index, step in enumerate(raw['steps']):
        weights = np.zeros(9)
        weights[0] = -.5
        result = execute_quote(raw, step, state, weights, save_fill)
        assert result['status'] == 'complete'
        for key, value in result['trace'].items():
            assert value == expected['trace'][index][key]
    assert state == expected['last_state']
    assert len(fills) == 2
