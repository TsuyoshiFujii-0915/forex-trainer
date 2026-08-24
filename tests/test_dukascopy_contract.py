"""Current forex-env currency-pair contract tests for the Dukascopy fetcher."""

from __future__ import annotations

import pytest
from forex_env.errors import DataError

from forex_trainer.dukascopy import to_dukascopy_instrument


def test_dukascopy_fetcher_rejects_non_jpy_denomination() -> None:
    """The JPY-specific price scale cannot be applied to another base currency."""
    with pytest.raises(DataError, match="JPY-denominated"):
        to_dukascopy_instrument("USD/EUR")
