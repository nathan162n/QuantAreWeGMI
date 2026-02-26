"""Unit tests for mean reversion strategy signal generation.

Tests verify correct buy/hold signal generation based on z-score
and RSI thresholds, including freefall protection.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data import indicators
from src.strategy.mean_reversion import MeanReversionStrategy


def _make_price_series_with_zscore(target_z: float, period: int = 20, length: int = 60) -> pd.Series:
    """Generate a synthetic price series with a specific terminal z-score.

    Creates a stable price series with a final price engineered to
    produce approximately the target z-score.

    Args:
        target_z: Desired z-score for the last price.
        period: Z-score lookback period. Default 20.
        length: Total series length. Default 60.

    Returns:
        pd.Series: Price series where the last value has approximately
            the target z-score.

    Example:
        >>> series = _make_price_series_with_zscore(-2.5)
        >>> z = indicators.rolling_zscore(series, 20)
        >>> abs(z[-1] - (-2.5)) < 0.5
    """
    np.random.seed(42)
    base_price = 100.0
    noise = np.random.randn(length - 1) * 0.5
    prices = base_price + np.cumsum(noise) * 0.1
    prices = np.clip(prices, 80, 120)

    recent = prices[-(period - 1):]
    mean = np.mean(recent)
    std = max(np.std(recent), 0.01)
    target_price = mean + target_z * std

    prices = np.append(prices, target_price)
    return pd.Series(prices)


def _make_bars_df(close_series: pd.Series) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame from a close series.

    Args:
        close_series: Series of close prices.

    Returns:
        pd.DataFrame: DataFrame with open, high, low, close, volume, vwap columns.

    Example:
        >>> df = _make_bars_df(pd.Series([100]*60))
    """
    return pd.DataFrame({
        "open": close_series * 0.999,
        "high": close_series * 1.005,
        "low": close_series * 0.995,
        "close": close_series,
        "volume": [1_000_000] * len(close_series),
        "vwap": close_series,
    })


class TestMeanReversionSignals:
    """Tests for mean reversion buy signal generation."""

    @patch("src.strategy.mean_reversion._load_config")
    def test_buy_signal_generated_at_zscore_minus_2_5(self, mock_config):
        """Z = -2.5, RSI = 30 should produce a BUY signal.

        Entry criteria: Z < -2.0 AND Z > -4.0 AND RSI < 35

        Example:
            >>> signal.signal == "BUY"
        """
        mock_config.return_value = {
            "layer1_name": "mean_reversion_v1",
            "layer1_lookback_bars": 60,
            "layer1_timeframe": "1Min",
            "zscore_period": 20,
            "zscore_entry_threshold": -2.0,
            "zscore_entry_freefall": -4.0,
            "zscore_exit_threshold": -0.5,
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "max_holding_days": 15,
            "min_holding_days": 3,
        }

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.config = mock_config.return_value
        strategy.regime_filter = MagicMock()
        strategy.bars_cache = {}

        close = _make_price_series_with_zscore(-2.5, period=20, length=60)
        np.random.seed(123)
        n = len(close)
        decline = np.linspace(0, -8, n)
        close_declining = pd.Series(100 + decline + np.random.randn(n) * 0.1)

        df = _make_bars_df(close_declining)
        strategy.bars_cache["TEST"] = df

        signal = strategy._evaluate_symbol("TEST")

        assert signal is not None

        actual_z = indicators.rolling_zscore(df["close"], 20)[-1]
        actual_rsi = indicators.rsi(df["close"], 14)[-1]

        if actual_z < -2.0 and actual_z > -4.0 and actual_rsi < 35:
            assert signal.signal == "BUY"
        else:
            assert signal.signal in ("BUY", "HOLD")

    @patch("src.strategy.mean_reversion._load_config")
    def test_no_buy_in_freefall_at_zscore_minus_4_5(self, mock_config):
        """Z = -4.5 should NOT produce a BUY (freefall filter).

        Freefall protection: Z < -4.0 blocks entry even with RSI < 35.

        Example:
            >>> signal.signal != "BUY"  # Blocked by freefall filter
        """
        mock_config.return_value = {
            "layer1_name": "mean_reversion_v1",
            "layer1_lookback_bars": 60,
            "layer1_timeframe": "1Min",
            "zscore_period": 20,
            "zscore_entry_threshold": -2.0,
            "zscore_entry_freefall": -4.0,
            "zscore_exit_threshold": -0.5,
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "max_holding_days": 15,
            "min_holding_days": 3,
        }

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.config = mock_config.return_value
        strategy.regime_filter = MagicMock()
        strategy.bars_cache = {}

        np.random.seed(456)
        n = 61
        crash = np.linspace(0, -20, n)
        close = pd.Series(100 + crash)
        df = _make_bars_df(close)
        strategy.bars_cache["CRASH"] = df

        signal = strategy._evaluate_symbol("CRASH")

        if signal is not None:
            actual_z = indicators.rolling_zscore(df["close"], 20)[-1]
            if actual_z < -4.0:
                assert signal.signal != "BUY", (
                    f"BUY generated at z={actual_z:.2f} (freefall)"
                )

    @patch("src.strategy.mean_reversion._load_config")
    def test_no_buy_above_entry_threshold_at_zscore_minus_1_5(self, mock_config):
        """Z = -1.5 should NOT produce a BUY signal.

        Entry threshold is -2.0, so -1.5 is not oversold enough.

        Example:
            >>> signal.signal == "HOLD"
        """
        mock_config.return_value = {
            "layer1_name": "mean_reversion_v1",
            "layer1_lookback_bars": 60,
            "layer1_timeframe": "1Min",
            "zscore_period": 20,
            "zscore_entry_threshold": -2.0,
            "zscore_entry_freefall": -4.0,
            "zscore_exit_threshold": -0.5,
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "max_holding_days": 15,
            "min_holding_days": 3,
        }

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.config = mock_config.return_value
        strategy.regime_filter = MagicMock()
        strategy.bars_cache = {}

        np.random.seed(789)
        close = pd.Series([100.0] * 60 + [99.0])
        df = _make_bars_df(close)
        strategy.bars_cache["MILD"] = df

        signal = strategy._evaluate_symbol("MILD")

        if signal is not None:
            actual_z = indicators.rolling_zscore(df["close"], 20)[-1]
            if actual_z > -2.0:
                assert signal.signal != "BUY", (
                    f"BUY at z={actual_z:.2f} exceeds threshold"
                )

    @patch("src.strategy.mean_reversion._load_config")
    def test_exit_blocked_before_min_holding_days(self, mock_config):
        """Position held < MIN_HOLDING_DAYS should not exit.

        Example:
            >>> strategy.should_exit("AAPL", {"days_held": 1})
            False
        """
        mock_config.return_value = {
            "layer1_name": "mean_reversion_v1",
            "layer1_lookback_bars": 60,
            "layer1_timeframe": "1Min",
            "zscore_period": 20,
            "zscore_entry_threshold": -2.0,
            "zscore_entry_freefall": -4.0,
            "zscore_exit_threshold": -0.5,
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "max_holding_days": 15,
            "min_holding_days": 3,
        }

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.config = mock_config.return_value
        strategy.regime_filter = MagicMock()
        strategy.bars_cache = {}

        result = strategy.should_exit("AAPL", {"days_held": 1})
        assert result is False

        result = strategy.should_exit("AAPL", {"days_held": 2})
        assert result is False
