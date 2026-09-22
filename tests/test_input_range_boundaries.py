"""Successor coverage retains original flat waiting time at fold boundaries."""
from __future__ import annotations

import pandas as pd

from forex_trainer.input_recovery import select_contiguous_range
from test_full_period import market_calendar


def test_complete_coverage_keeps_requested_start_before_first_close() -> None:
    market, calendar = market_calendar('2023-10-02', 150)
    start = '2024-01-01T00:00:00Z'
    end = (pd.Timestamp(calendar['sessions'][-1]['session_close']) + pd.Timedelta(hours=1)).isoformat()
    result = select_contiguous_range(market, calendar, start, end)
    assert result['measurement_start'] == pd.Timestamp(start).isoformat()
    assert result['measurement_end'] == pd.Timestamp(end).tz_convert('UTC').isoformat()
