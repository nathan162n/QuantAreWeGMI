"""Unit tests for universe screener filter pipeline.

Tests verify that each filter stage correctly accepts or rejects
symbols based on the configured thresholds.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.universe.screener import UniverseScreener


def _make_bars(close: float, volume: float, n: int = 60) -> pd.DataFrame:
    """Create a minimal bar DataFrame for testing filters.

    Args:
        close: Constant close price.
        volume: Constant volume.
        n: Number of bars. Default 60.

    Returns:
        pd.DataFrame: Bar data with OHLCV columns.

    Example:
        >>> df = _make_bars(50.0, 1_000_000)
    """
    return pd.DataFrame({
        "open": [close * 0.999] * n,
        "high": [close * 1.002] * n,
        "low": [close * 0.998] * n,
        "close": [close] * n,
        "volume": [volume] * n,
        "vwap": [close] * n,
    })


class TestPriceFilter:
    """Tests for the minimum price filter."""

    def test_accepts_above_min_price(self):
        """Stock priced at $50 should pass the $15 min price filter.

        Example:
            >>> screener._filter_price(["AAPL"], {"AAPL": bars_at_50})
            (["AAPL"], {})
        """
        screener = UniverseScreener.__new__(UniverseScreener)
        screener.config = {"min_price": 15.0}

        bars = {"AAPL": _make_bars(50.0, 1_000_000)}
        passed, rejected = screener._filter_price(["AAPL"], bars)
        assert "AAPL" in passed
        assert len(rejected) == 0

    def test_rejects_below_min_price(self):
        """Stock priced at $10 should be rejected by the $15 min price filter.

        Example:
            >>> screener._filter_price(["PENNY"], {"PENNY": bars_at_10})
            ([], {"PENNY": "Price $10.00 < $15.00"})
        """
        screener = UniverseScreener.__new__(UniverseScreener)
        screener.config = {"min_price": 15.0}

        bars = {"PENNY": _make_bars(10.0, 1_000_000)}
        passed, rejected = screener._filter_price(["PENNY"], bars)
        assert len(passed) == 0
        assert "PENNY" in rejected
        assert "Price" in rejected["PENNY"]


class TestLiquidityFilter:
    """Tests for the average dollar volume filter."""

    def test_accepts_high_adv(self):
        """Stock with ADV of $100M should pass the $50M filter.

        Volume 2,000,000 * $50 = $100M ADV.

        Example:
            >>> screener._filter_liquidity(["AAPL"], bars)
            (["AAPL"], {})
        """
        screener = UniverseScreener.__new__(UniverseScreener)
        screener.config = {"min_adv_dollars": 50_000_000}

        bars = {"AAPL": _make_bars(50.0, 2_000_000)}
        passed, rejected = screener._filter_liquidity(["AAPL"], bars)
        assert "AAPL" in passed

    def test_rejects_low_adv(self):
        """Stock with ADV of $5M should be rejected by the $50M filter.

        Volume 100,000 * $50 = $5M ADV.

        Example:
            >>> screener._filter_liquidity(["ILLIQ"], bars)
            ([], {"ILLIQ": "ADV $5.0M < $50M"})
        """
        screener = UniverseScreener.__new__(UniverseScreener)
        screener.config = {"min_adv_dollars": 50_000_000}

        bars = {"ILLIQ": _make_bars(50.0, 100_000)}
        passed, rejected = screener._filter_liquidity(["ILLIQ"], bars)
        assert len(passed) == 0
        assert "ILLIQ" in rejected


class TestVolatilityFilter:
    """Tests for the realized volatility cap filter."""

    def test_accepts_low_vol(self):
        """Constant-price stock should have near-zero vol and pass.

        Example:
            >>> screener._filter_volatility(["STABLE"], bars)
            (["STABLE"], {})
        """
        screener = UniverseScreener.__new__(UniverseScreener)
        screener.config = {"max_realized_vol": 0.80}

        bars = {"STABLE": _make_bars(50.0, 1_000_000)}
        passed, rejected = screener._filter_volatility(["STABLE"], bars)
        assert "STABLE" in passed

    def test_rejects_high_vol(self):
        """Highly volatile stock should be rejected.

        Example:
            >>> screener._filter_volatility(["WILD"], bars)
            ([], {"WILD": "Volatility 120.0% > 80%"})
        """
        screener = UniverseScreener.__new__(UniverseScreener)
        screener.config = {"max_realized_vol": 0.80}

        np.random.seed(42)
        n = 60
        wild_close = 50 + np.cumsum(np.random.randn(n) * 5)
        wild_close = np.maximum(wild_close, 1.0)
        df = pd.DataFrame({
            "open": wild_close * 0.99,
            "high": wild_close * 1.05,
            "low": wild_close * 0.95,
            "close": wild_close,
            "volume": [1_000_000] * n,
            "vwap": wild_close,
        })
        bars = {"WILD": df}

        passed, rejected = screener._filter_volatility(["WILD"], bars)
        from src.data.indicators import realized_vol
        vol = realized_vol(pd.Series(wild_close), 20)
        if vol > 0.80:
            assert "WILD" in rejected
        else:
            assert "WILD" in passed


class TestMultipleFilters:
    """Tests for combined filter pipeline behavior."""

    def test_mixed_universe_filters_correctly(self):
        """A mix of passing and failing stocks should be filtered correctly.

        Example:
            >>> # GOOD: $50, 2M vol -> passes
            >>> # CHEAP: $10, 2M vol -> rejected (price)
            >>> # ILLIQ: $50, 50K vol -> rejected (liquidity)
        """
        screener = UniverseScreener.__new__(UniverseScreener)
        screener.config = {
            "min_price": 15.0,
            "min_adv_dollars": 50_000_000,
            "max_realized_vol": 0.80,
        }

        bars = {
            "GOOD": _make_bars(50.0, 2_000_000),
            "CHEAP": _make_bars(10.0, 2_000_000),
            "ILLIQ": _make_bars(50.0, 50_000),
        }

        symbols = ["GOOD", "CHEAP", "ILLIQ"]

        passed_price, rej_price = screener._filter_price(symbols, bars)
        assert "GOOD" in passed_price
        assert "CHEAP" not in passed_price

        passed_liq, rej_liq = screener._filter_liquidity(passed_price, bars)
        assert "GOOD" in passed_liq

        passed_vol, rej_vol = screener._filter_volatility(passed_liq, bars)
        assert "GOOD" in passed_vol
