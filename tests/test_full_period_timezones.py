"""Named timezone cache compatibility across offset changes."""

from pathlib import Path

import pandas as pd

from forex_trainer.full_period import prepare_period
from test_full_period import market_calendar


def test_named_zone_labels_match_iso_offsets_across_dst(tmp_path: Path) -> None:
    market, calendar = market_calendar("2024-08-01", 100)
    market.index = market.index.tz_localize("Europe/London")
    calendar["timezone"] = "Europe/London"
    for label, row in zip(market.index, calendar["sessions"]):
        row["bar_label"] = label.isoformat()
        row["session_close"] = label.isoformat()
        row["session_open"] = (label - pd.Timedelta(days=1)).isoformat()
    plan = prepare_period(market, calendar, "2024-11-01T00:00:00Z", "2024-12-01T00:00:00Z", tuple(calendar["symbols"]), tmp_path / "london.parquet")
    assert plan.timestamps[0] == "2024-11-01T00:00:00+00:00"
    assert len(plan.history_labels) == 63
    assert len(plan.timestamps) == 21
