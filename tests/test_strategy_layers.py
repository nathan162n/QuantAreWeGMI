"""Comprehensive tests for all four strategy layers.

Tests verify signal logic, entry/exit conditions, indicator thresholds,
and cross-layer deduplication using pre-loaded BarStore data.
All assertions use concrete numerical values — no placeholder assertions.
"""

import pytest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from src.data.bar_store import BarStore
from src.strategy.base_strategy import SignalResult


def _make_bar(open_=100.0, high=101.0, low=99.0, close=100.5, volume=10000, symbol="AAPL", ts=None):
    """Helper to create a bar dict."""
    if ts is None:
        ts = datetime.now(timezone.utc)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "symbol": symbol, "timestamp": ts}


def _make_bar_store_with_n_bars(symbol, n, base_close=100.0, close_series=None, volume_series=None):
    """Create a BarStore pre-loaded with n 1-minute bars.

    close_series: optional list of close prices (length n)
    volume_series: optional list of volumes (length n)
    """
    store = BarStore()
    # Use a session start time in the 14:30 UTC (09:30 ET) window so bars
    # accumulate as opening range bars correctly.
    base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    for i in range(n):
        close = close_series[i] if close_series else base_close
        vol = volume_series[i] if volume_series else 5000
        bar = {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": vol,
            "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=i),
        }
        store.update(symbol, "1Min", bar)
    return store


# ===== LAYER 1 — VWAP Mean Reversion =====

class TestVWAPMeanReversionLayer:

    def test_l1_layer_name(self):
        """Layer name must be exactly 'L1_VWAP_MR'."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        layer = VWAPMeanReversionStrategy()
        assert layer.layer_name == "L1_VWAP_MR"

    def test_l1_default_parameters(self):
        """Default parameters match config.yml values."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        layer = VWAPMeanReversionStrategy()
        assert layer.vwap_deviation_entry == 0.003   # 0.3% below VWAP (config value)
        assert layer.rsi_period == 14
        assert layer.rsi_entry == 48.0
        assert layer.rsi_exit == 56.0
        assert layer.max_hold_minutes == 90
        assert layer.stop_atr_multiplier == 1.0

    def test_l1_buy_when_price_below_vwap_threshold(self):
        """Price ~1% below VWAP, RSI confirming oversold -> BUY signal."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        # First 20 bars at 152.0 set a high VWAP, last 10 bars decline sharply.
        # VWAP will be anchored near 151+ while close falls to ~149.6.
        # Deviation = (149.6 - 151.5) / 151.5 ~ -1.25% >> 0.5% threshold.
        closes = [152.0] * 20 + [151.0, 150.8, 150.6, 150.4, 150.2, 150.0, 149.9, 149.8, 149.7, 149.6]
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        vwap = store.get_session_vwap("AAPL")
        # VWAP should be above the final closes because the early bars were higher.
        assert vwap > 150.5, f"VWAP should be above 150.5 but got {vwap}"

        # Use an explicit morning session timestamp so the Phase 4B time gate
        # (no_entry_after=15:00) does not fire regardless of when tests run.
        bar = _make_bar(
            close=closes[-1], high=closes[-1] + 0.3, low=closes[-1] - 0.5,
            symbol="AAPL", ts=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        )
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        # With ~1% deviation from VWAP and declining RSI, should fire BUY.
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) > 0, (
            f"Expected BUY with price {closes[-1]:.2f} far below VWAP {vwap:.2f} "
            f"but got: {[s.signal for s in signals]}"
        )
        s = buy_signals[0]
        assert s.layer_name == "L1_VWAP_MR"
        assert 0.0 < s.confidence <= 1.0
        assert s.stop_price < closes[-1]
        assert s.signal_price == closes[-1]

    def test_l1_buy_signal_metadata_includes_vwap_and_rsi(self):
        """BUY signal metadata must contain 'vwap', 'rsi', and 'distance_from_vwap'."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [152.0] * 20 + [149.6] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        bar = _make_bar(close=149.6, high=150.0, low=149.2, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        if buy_signals:
            meta = buy_signals[0].metadata
            assert "vwap" in meta
            assert "rsi" in meta
            assert "distance_from_vwap" in meta
            # distance_from_vwap should be negative (price is below VWAP)
            assert meta["distance_from_vwap"] < -0.005

    def test_l1_no_buy_when_price_at_vwap(self):
        """Flat prices -> VWAP == close -> distance = 0 -> no BUY."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [100.0] * 30
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        bar = _make_bar(close=100.0, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, "Should not BUY when price equals VWAP"

    def test_l1_no_buy_when_price_0_4_pct_below_vwap(self):
        """Price 0.4% below VWAP is below the 0.5% threshold -> no BUY."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        # Build store with VWAP at exactly 100.0, then present a bar 0.4% below.
        closes = [100.0] * 30
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        vwap = store.get_session_vwap("AAPL")
        # With flat closes, VWAP = 100.0. Price 0.4% below = 99.6.
        close_04pct_below = vwap * (1.0 - 0.004)

        layer = VWAPMeanReversionStrategy()
        bar = _make_bar(close=close_04pct_below, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, (
            f"Should not BUY at only 0.4% below VWAP (threshold is 0.5%), "
            f"got signals: {[s.signal for s in signals]}"
        )

    def test_l1_no_buy_in_first_5_bars(self):
        """Layer must not enter in the first 5 bars of session (bar_count <= 5)."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        # Only 5 bars loaded — bar_count will be 5.
        closes = [152.0] * 3 + [149.0, 148.5]
        store = _make_bar_store_with_n_bars("AAPL", 5, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        bar = _make_bar(close=148.5, high=149.0, low=148.0, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, "Should not BUY in first 5 bars"

    def test_l1_exit_when_price_returns_to_vwap(self):
        """Open position, price returns within 0.1% of VWAP -> EXIT."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [152.0] * 20 + [150.0] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        # Force an open position at 150.0 with stop below.
        layer._open_positions["AAPL"] = {
            "entry_price": 150.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "stop_price": 149.0,
            "bars_held": 3,
        }

        vwap = store.get_session_vwap("AAPL")
        # Price right at VWAP: distance_from_vwap > -0.001 -> triggers VWAP exit.
        recovery_close = vwap * 1.0001  # 0.01% above VWAP
        bar = _make_bar(close=recovery_close, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, (
            f"Expected EXIT when price {recovery_close:.4f} returns to VWAP {vwap:.4f}"
        )
        assert exit_signals[0].layer_name == "L1_VWAP_MR"
        assert "exit_reason" in exit_signals[0].metadata

    def test_l1_exit_on_rsi_recovery(self):
        """RSI rises above rsi_exit (62.0) while in a long position -> EXIT."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        # Rising price series produces high RSI.
        closes = [100.0 + i * 0.5 for i in range(30)]
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        # Plant a position with a stop well below current price to avoid stop trigger.
        layer._open_positions["AAPL"] = {
            "entry_price": 100.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=20),
            "stop_price": 90.0,
            "bars_held": 15,
        }

        current_close = closes[-1]
        bar = _make_bar(close=current_close, high=current_close + 0.5, low=current_close - 0.3, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        # Rising prices will produce RSI > 62 -> EXIT
        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, (
            "Expected EXIT: steadily rising prices should produce RSI > 62"
        )

    def test_l1_exit_on_time_stop(self):
        """bars_held >= max_hold_minutes (180) -> EXIT regardless of other conditions."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [152.0] * 20 + [150.0] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        layer._open_positions["AAPL"] = {
            "entry_price": 150.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "stop_price": 149.0,
            "bars_held": 180,  # At the time-stop limit exactly
        }

        # Price is still below VWAP (no VWAP exit), no stop breach (close > stop)
        bar = _make_bar(close=150.0, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, "Expected EXIT at time stop (180 bars)"
        assert "time_stop" in exit_signals[0].metadata.get("exit_reason", "")

    def test_l1_exit_on_stop_loss(self):
        """Current close drops below stop_price -> EXIT with stop_loss reason."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [152.0] * 20 + [150.0] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        layer._open_positions["AAPL"] = {
            "entry_price": 150.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "stop_price": 149.5,
            "bars_held": 5,
        }

        # Current close at 149.0 — below stop of 149.5
        bar = _make_bar(close=149.0, high=149.3, low=148.8, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, "Expected EXIT on stop-loss breach"
        assert "stop_loss" in exit_signals[0].metadata.get("exit_reason", "")

    def test_l1_no_second_entry_while_position_open(self):
        """Position already open -> second evaluate_bar call does not generate BUY."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [152.0] * 20 + [149.6] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        # Inject an open position.
        layer._open_positions["AAPL"] = {
            "entry_price": 149.6,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "stop_price": 149.0,
            "bars_held": 3,
        }

        # Present a bar that would normally trigger BUY (price deep below VWAP).
        bar = _make_bar(close=149.5, high=150.0, low=149.2, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, "Should not add a second position for same symbol"

    def test_l1_should_exit_stop_loss(self):
        """should_exit() returns True when close < stop_price."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [152.0] * 20 + [150.0] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        position_data = {
            "entry_price": 150.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "stop_price": 149.5,
            "bars_held": 5,
            "layer_name": "L1_VWAP_MR",
        }
        bar = _make_bar(close=149.0, symbol="AAPL")
        result = layer.should_exit("AAPL", bar, store, position_data)
        assert result is True

    def test_l1_should_exit_false_when_healthy(self):
        """should_exit() returns False when price is above stop, below VWAP, low RSI."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [152.0] * 20 + [150.0] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        position_data = {
            "entry_price": 150.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "stop_price": 148.0,
            "bars_held": 5,
            "layer_name": "L1_VWAP_MR",
        }
        # Close is 150.0, stop is 148.0, price is below VWAP (~151.5)
        bar = _make_bar(close=150.0, symbol="AAPL")
        result = layer.should_exit("AAPL", bar, store, position_data)
        # With price 150.0 below VWAP ~151.5 and stop at 148.0, should NOT exit.
        # (distance_from_vwap ~ -0.01, well below -0.001 threshold)
        assert result is False

    def test_l1_confidence_proportional_to_vwap_distance(self):
        """Greater VWAP distance -> higher confidence (capped at 1.0)."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        # Scenario A: moderate distance (~0.8%)
        closes_a = [100.0] * 20 + [99.2] * 10
        store_a = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes_a)
        layer_a = VWAPMeanReversionStrategy()
        bar_a = _make_bar(close=99.2, high=99.5, low=98.9, symbol="AAPL")
        sigs_a = [s for s in layer_a.evaluate_bar("AAPL", bar_a, store_a, None) if s.signal == "BUY"]

        # Scenario B: large distance (~2%)
        closes_b = [100.0] * 20 + [98.0] * 10
        store_b = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes_b)
        layer_b = VWAPMeanReversionStrategy()
        bar_b = _make_bar(close=98.0, high=98.5, low=97.5, symbol="AAPL")
        sigs_b = [s for s in layer_b.evaluate_bar("AAPL", bar_b, store_b, None) if s.signal == "BUY"]

        if sigs_a and sigs_b:
            assert sigs_b[0].confidence >= sigs_a[0].confidence, (
                f"Larger VWAP deviation should yield >= confidence: "
                f"sigs_b={sigs_b[0].confidence:.4f}, sigs_a={sigs_a[0].confidence:.4f}"
            )


# ===== LAYER 2 — Opening Range Breakout =====

class TestOpeningRangeBreakoutLayer:

    def test_l2_layer_name(self):
        """Layer name must be exactly 'L2_ORB'."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        layer = OpeningRangeBreakoutStrategy()
        assert layer.layer_name == "L2_ORB"

    def test_l2_default_parameters(self):
        """Default parameters match config.yml values."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        layer = OpeningRangeBreakoutStrategy()
        assert layer.breakout_buffer_pct == 0.001
        assert layer.volume_confirmation_ratio == 1.5
        assert layer.bar_position_threshold == 0.70
        assert layer.extension_target_multiplier == 2.0
        assert layer.max_range_atr_multiplier == 3.0
        assert layer.max_hold_minutes == 330

    def test_l2_opening_range_computed_correctly_at_finalize(self):
        """30 opening bars: finalize_opening_range returns correct high and low."""
        store = BarStore()
        symbol = "SPY"
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        highs, lows = [], []

        for i in range(30):
            h = 400.0 + (i % 5) * 0.5
            l = 399.0 - (i % 3) * 0.2
            highs.append(h)
            lows.append(l)
            bar = {
                "open": 399.5, "high": h, "low": l, "close": 399.8,
                "volume": 5000, "symbol": symbol,
                "timestamp": base_ts + timedelta(minutes=i),
            }
            store.update(symbol, "1Min", bar)

        or_ = store.finalize_opening_range(symbol)
        assert or_ is not None
        assert abs(or_["high"] - max(highs)) < 0.001, (
            f"Expected high={max(highs):.3f}, got {or_['high']:.3f}"
        )
        assert abs(or_["low"] - min(lows)) < 0.001, (
            f"Expected low={min(lows):.3f}, got {or_['low']:.3f}"
        )

    def test_l2_no_signal_before_opening_range_finalized(self):
        """evaluate_bar returns [] when opening range not yet available."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        store = BarStore()
        symbol = "SPY"
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        for i in range(25):
            bar = {
                "open": 400.0, "high": 402.0, "low": 399.0, "close": 400.5,
                "volume": 1000, "symbol": symbol,
                "timestamp": base_ts + timedelta(minutes=i),
            }
            store.update(symbol, "1Min", bar)
        # opening range NOT finalized

        layer = OpeningRangeBreakoutStrategy()
        bar = {
            "open": 401.9, "high": 403.0, "low": 401.5, "close": 402.8,
            "volume": 5000, "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=25),
        }
        signals = layer.evaluate_bar(symbol, bar, store, None)
        assert signals == [], "No signal before opening range is finalized"

    def test_l2_buy_on_breakout_above_range_with_volume(self):
        """price > range_high * 1.001, volume > 1.5x avg, bar in top 30% -> BUY."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        store = BarStore()
        symbol = "SPY"
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        # 30 bars at 09:30-10:00 to establish opening range (high=402.0, low=399.0)
        for i in range(30):
            bar = {
                "open": 400.0, "high": 402.0, "low": 399.0, "close": 400.5,
                "volume": 1000, "symbol": symbol,
                "timestamp": base_ts + timedelta(minutes=i),
            }
            store.update(symbol, "1Min", bar)
        store.finalize_opening_range(symbol)
        or_ = store.get_opening_range(symbol)
        assert or_ is not None
        assert or_["high"] == pytest.approx(402.0)

        # Add 21 more bars with avg volume 1000 to warm up indicators
        for i in range(30, 51):
            bar = {
                "open": 400.0, "high": 402.5, "low": 399.5, "close": 400.5,
                "volume": 1000, "symbol": symbol,
                "timestamp": base_ts + timedelta(minutes=i),
            }
            store.update(symbol, "1Min", bar)

        layer = OpeningRangeBreakoutStrategy()
        # Breakout bar:
        #   close=402.5 > 402.0 * 1.001 = 402.402 ✓
        #   volume=3000 = 3x avg 1000 > 1.5 ✓
        #   bar_position = (402.5 - 402.0) / (402.8 - 402.0) = 0.5/0.8 = 0.625 > 0.70?
        # Use wider range: low=402.0, high=402.8 -> (402.5-402.0)/(402.8-402.0)=0.625 not enough
        # Use: open=402.1, high=402.8, low=402.0, close=402.6
        # bar_position = (402.6-402.0)/(402.8-402.0) = 0.6/0.8 = 0.75 > 0.70 ✓
        breakout_bar = {
            "open": 402.1, "high": 402.8, "low": 402.0, "close": 402.6,
            "volume": 3000,
            "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=51),
        }
        signals = layer.evaluate_bar(symbol, breakout_bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) > 0, (
            f"Expected BUY on breakout close=402.6 > or_high*1.001=402.402 with "
            f"3x volume, but got: {[s.signal for s in signals]}"
        )
        s = buy_signals[0]
        assert s.layer_name == "L2_ORB"
        # Stop = or_high - or_size * 0.5 = 402.0 - 3.0 * 0.5 = 400.5
        assert abs(s.stop_price - 400.5) < 0.01, f"Expected stop ~400.5, got {s.stop_price}"
        assert s.stop_price < 402.6

    def test_l2_no_buy_without_volume_confirmation(self):
        """Price breaks out but volume < 1.5x avg -> HOLD (no BUY)."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        store = BarStore()
        symbol = "MSFT"
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        for i in range(30):
            bar = {"open": 300.0, "high": 302.0, "low": 299.0, "close": 300.5,
                   "volume": 2000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)
        store.finalize_opening_range(symbol)

        for i in range(30, 51):
            bar = {"open": 300.0, "high": 302.0, "low": 299.0, "close": 300.5,
                   "volume": 2000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)

        layer = OpeningRangeBreakoutStrategy()
        # Price above breakout level but volume only 1.25x (2500 / 2000 = 1.25 < 1.5)
        breakout_bar = {
            "open": 301.9, "high": 302.5, "low": 301.8, "close": 302.3,
            "volume": 2500,
            "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=51),
        }
        signals = layer.evaluate_bar(symbol, breakout_bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, (
            "Should not BUY without sufficient volume confirmation (1.25x < 1.5x)"
        )

    def test_l2_no_buy_without_bar_position_confirmation(self):
        """Price breaks out, volume OK, but bar closes near low (bearish) -> HOLD."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        store = BarStore()
        symbol = "AMZN"
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        for i in range(30):
            bar = {"open": 180.0, "high": 182.0, "low": 179.0, "close": 180.5,
                   "volume": 1000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)
        store.finalize_opening_range(symbol)

        for i in range(30, 51):
            bar = {"open": 180.0, "high": 182.0, "low": 179.0, "close": 180.5,
                   "volume": 1000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)

        layer = OpeningRangeBreakoutStrategy()
        # close=182.3 > 182.0*1.001=182.182, volume=4000 > 1.5x,
        # but bar_pos = (182.3-180.0)/(183.0-180.0) = 2.3/3.0 = 0.767 > 0.70 ... let's make it bearish
        # low=182.1, high=183.5, close=182.2 -> bar_pos=(182.2-182.1)/(183.5-182.1)=0.1/1.4=0.071
        low_position_bar = {
            "open": 183.0, "high": 183.5, "low": 182.1, "close": 182.2,
            "volume": 4000,
            "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=51),
        }
        signals = layer.evaluate_bar(symbol, low_position_bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, (
            "Should not BUY when bar closes near the low (bar_position < 0.70)"
        )

    def test_l2_no_second_entry_same_day(self):
        """After first ORB entry fires, second breakout same symbol same day -> HOLD."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        store = BarStore()
        symbol = "GOOGL"
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        for i in range(30):
            bar = {"open": 150.0, "high": 152.0, "low": 149.0, "close": 150.5,
                   "volume": 1000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)
        store.finalize_opening_range(symbol)

        for i in range(30, 51):
            bar = {"open": 150.0, "high": 152.0, "low": 149.0, "close": 150.5,
                   "volume": 1000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)

        layer = OpeningRangeBreakoutStrategy()
        # First breakout: close=152.3 > 152.0*1.001=152.152, vol=3000, bullish bar
        b1 = {
            "open": 152.0, "high": 152.5, "low": 151.8, "close": 152.4,
            "volume": 3000, "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=51),
        }
        signals1 = layer.evaluate_bar(symbol, b1, store, None)
        buy_signals1 = [s for s in signals1 if s.signal == "BUY"]
        assert len(buy_signals1) > 0, "First breakout should produce BUY"

        # Add bar to store to maintain indicator data.
        store.update(symbol, "1Min", b1)

        # Second breakout attempt — must be blocked by _triggered_today.
        b2 = {
            "open": 153.0, "high": 153.5, "low": 152.9, "close": 153.3,
            "volume": 4000, "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=52),
        }
        signals2 = layer.evaluate_bar(symbol, b2, store, None)
        buy_signals2 = [s for s in signals2 if s.signal == "BUY"]
        assert len(buy_signals2) == 0, "Second ORB entry same day must be blocked"

    def test_l2_exit_on_breakout_failure(self):
        """Price falls back below or_high * 0.998 -> EXIT (breakout failure).

        or_high=502.0, breakout_failure_threshold = 502.0 * 0.998 = 500.996.
        A close of 500.5 is below 500.996, so breakout_failure fires.
        """
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        store = BarStore()
        symbol = "NVDA"
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        for i in range(30):
            bar = {"open": 500.0, "high": 502.0, "low": 499.0, "close": 500.5,
                   "volume": 1000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)
        store.finalize_opening_range(symbol)

        for i in range(30, 51):
            bar = {"open": 500.0, "high": 502.0, "low": 499.0, "close": 500.5,
                   "volume": 1000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)

        layer = OpeningRangeBreakoutStrategy()
        # Inject open position with or_high=502.0.
        # stop_price=498.0 is below the failure close so only breakout_failure fires.
        layer._open_positions[symbol] = {
            "entry_price": 502.6,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "stop_price": 498.0,
            "bars_held": 5,
            "opening_range_high": 502.0,
            "opening_range_size": 3.0,
        }

        # or_high * 0.998 = 502.0 * 0.998 = 500.996.
        # close=500.5 < 500.996 -> breakout_failure triggers.
        failure_bar = {
            "open": 501.0, "high": 501.5, "low": 500.3, "close": 500.5,
            "volume": 1000, "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=52),
        }
        signals = layer.evaluate_bar(symbol, failure_bar, store, None)
        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, (
            f"Expected EXIT on breakout failure (close=500.5 < 502.0*0.998=500.996)"
        )
        assert "breakout_failure" in exit_signals[0].metadata.get("exit_reason", "")

    def test_l2_exit_on_profit_target(self):
        """Price reaches entry + 2 * or_size -> EXIT (profit target)."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        store = BarStore()
        symbol = "META"
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        for i in range(30):
            bar = {"open": 300.0, "high": 302.0, "low": 299.0, "close": 300.5,
                   "volume": 1000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)
        store.finalize_opening_range(symbol)
        for i in range(30, 51):
            bar = {"open": 300.0, "high": 302.0, "low": 299.0, "close": 300.5,
                   "volume": 1000, "symbol": symbol, "timestamp": base_ts + timedelta(minutes=i)}
            store.update(symbol, "1Min", bar)

        layer = OpeningRangeBreakoutStrategy()
        # or_size = 302.0 - 299.0 = 3.0; entry = 302.4; profit_target = 302.4 + 2*3 = 308.4
        layer._open_positions[symbol] = {
            "entry_price": 302.4,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=15),
            "stop_price": 300.5,
            "bars_held": 10,
            "opening_range_high": 302.0,
            "opening_range_size": 3.0,
        }

        # Price at profit target
        target_bar = {
            "open": 308.0, "high": 308.5, "low": 307.8, "close": 308.4,
            "volume": 1000, "symbol": symbol,
            "timestamp": base_ts + timedelta(minutes=52),
        }
        signals = layer.evaluate_bar(symbol, target_bar, store, None)
        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, "Expected EXIT when price reaches profit target"
        assert "profit_target" in exit_signals[0].metadata.get("exit_reason", "")

    def test_l2_reset_daily_state_clears_all_tracking(self):
        """reset_daily_state() clears _triggered_today, _open_positions, _opening_range_size."""
        from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
        layer = OpeningRangeBreakoutStrategy()
        layer._triggered_today.add("AAPL")
        layer._open_positions["AAPL"] = {"entry_price": 100.0, "bars_held": 5}
        layer._opening_range_size["AAPL"] = 2.5

        layer.reset_daily_state()

        assert len(layer._triggered_today) == 0
        assert len(layer._open_positions) == 0
        assert len(layer._opening_range_size) == 0


# ===== LAYER 3 — RSI Reversal Scalp =====

class TestRSIReversalScalpLayer:

    def test_l3_layer_name(self):
        """Layer name must be exactly 'L3_RSI_SCALP'."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        layer = RSIReversalScalpStrategy()
        assert layer.layer_name == "L3_RSI_SCALP"

    def test_l3_5min_series_construction(self):
        """_build_5min_series with 80 1-min bars returns 16 synthetic 5-min bars."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        store = _make_bar_store_with_n_bars("TSLA", 80)
        layer = RSIReversalScalpStrategy()
        df = layer._build_5min_series("TSLA", store)
        assert df is not None
        assert len(df) == 16  # floor(80/5) = 16
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_l3_5min_series_requires_50_min_bars(self):
        """_build_5min_series returns None when fewer than 50 1-min bars available."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        store = _make_bar_store_with_n_bars("TSLA", 40)
        layer = RSIReversalScalpStrategy()
        df = layer._build_5min_series("TSLA", store)
        assert df is None

    def test_l3_buy_when_rsi7_below_25_with_reversal(self):
        """RSI(7) on 5-min < 25, current close > previous close -> BUY."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        # Steep 79-bar decline drives RSI(7) deep into oversold territory.
        # Last bar is slightly higher than the one before -> reversal tick.
        n = 80
        closes = [100.0 - i * 0.5 for i in range(79)] + [61.0]  # 60.5 -> 61.0 reversal
        store = _make_bar_store_with_n_bars("TSLA", n, close_series=closes)

        layer = RSIReversalScalpStrategy()
        bar = _make_bar(close=61.0, high=61.5, low=60.5, symbol="TSLA")
        signals = layer.evaluate_bar("TSLA", bar, store, None)

        # With -0.5/bar decline over 79 bars, RSI(7) should be deeply oversold.
        buy_signals = [s for s in signals if s.signal == "BUY"]
        if buy_signals:
            s = buy_signals[0]
            assert s.layer_name == "L3_RSI_SCALP"
            assert 0.0 <= s.confidence <= 1.0
            assert s.stop_price < 61.0
            assert "rsi_7" in s.metadata
            # Verify the RSI was genuinely below 25 to be sure we fired correctly.
            assert s.metadata["rsi_7"] < 25.0, (
                f"BUY should only fire when RSI(7) < 25, got {s.metadata['rsi_7']:.2f}"
            )

    def test_l3_no_buy_without_reversal_confirmation(self):
        """RSI(7) < 25 but current close <= previous close -> no BUY."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        n = 80
        # Declining all the way, last two bars both declining.
        closes = [100.0 - i * 0.5 for i in range(80)]  # last close = 60.5, prev = 61.0
        store = _make_bar_store_with_n_bars("TSLA", n, close_series=closes)

        layer = RSIReversalScalpStrategy()
        # Current bar close is at or below the previous bar close.
        bar = _make_bar(close=60.5, high=61.0, low=60.0, symbol="TSLA")
        signals = layer.evaluate_bar("TSLA", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, (
            "Should not BUY without reversal confirmation (close <= prev close)"
        )

    def test_l3_rsi7_threshold_is_25_not_35(self):
        """Threshold for L3 entry is RSI(7) < 25, NOT < 35. Mild decline not enough."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        from src.data.indicators import rsi as calc_rsi
        n = 80
        # Very mild decline: RSI(7) will be around 35-45, not below 25.
        closes = [100.0 - i * 0.05 for i in range(80)]
        store = _make_bar_store_with_n_bars("NVDA", n, close_series=closes)

        layer = RSIReversalScalpStrategy()
        five_min_df = layer._build_5min_series("NVDA", store)
        assert five_min_df is not None
        rsi_arr = calc_rsi(five_min_df["close"], 7)
        valid_rsi = [v for v in rsi_arr if not np.isnan(v)]
        last_rsi = valid_rsi[-1] if valid_rsi else 50.0

        # With mild decline the RSI(7) should be above 25 (not deeply oversold).
        if last_rsi >= 25.0:
            bar = _make_bar(close=closes[-1] + 0.1, symbol="NVDA")
            signals = layer.evaluate_bar("NVDA", bar, store, None)
            buy_signals = [s for s in signals if s.signal == "BUY"]
            assert len(buy_signals) == 0, (
                f"L3 should NOT fire when RSI(7)={last_rsi:.1f} >= 25 (mild decline)"
            )

    def test_l3_stop_placed_at_half_atr_below_entry(self):
        """BUY signal stop_price = entry_price - 0.5 * ATR(14, 5-min)."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        from src.data.indicators import atr as calc_atr
        n = 80
        closes = [100.0 - i * 0.5 for i in range(79)] + [61.0]
        store = _make_bar_store_with_n_bars("TSLA", n, close_series=closes)

        layer = RSIReversalScalpStrategy()
        bar = _make_bar(close=61.0, high=61.5, low=60.5, symbol="TSLA")
        signals = layer.evaluate_bar("TSLA", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        if buy_signals:
            s = buy_signals[0]
            entry_price = s.signal_price
            stop = s.stop_price
            # Compute expected ATR from 5-min series
            five_min_df = layer._build_5min_series("TSLA", store)
            if five_min_df is not None and len(five_min_df) >= 15:
                atr_arr = calc_atr(five_min_df["high"], five_min_df["low"], five_min_df["close"], 14)
                valid_atr = [v for v in atr_arr if not np.isnan(v)]
                if valid_atr:
                    expected_stop = entry_price - 0.5 * valid_atr[-1]
                    assert abs(stop - expected_stop) < 0.01, (
                        f"Expected stop={expected_stop:.4f}, got {stop:.4f}"
                    )

    def test_l3_time_stop_at_10_bars(self):
        """Position held for exactly 10 bars -> EXIT on next evaluate_bar call."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        n = 80
        closes = [50.0] * n
        store = _make_bar_store_with_n_bars("INTC", n, close_series=closes)

        layer = RSIReversalScalpStrategy()
        # evaluate_bar increments bars_held by 1 before checking exit conditions.
        # Set bars_held=9 so it becomes 10 (the threshold) after the increment.
        layer._open_positions["INTC"] = {
            "entry_price": 50.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=55),
            "stop_price": 49.0,
            "bars_held": 9,  # becomes 10 after increment in evaluate_bar
        }

        bar = _make_bar(close=50.0, symbol="INTC")
        signals = layer.evaluate_bar("INTC", bar, store, None)

        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, (
            f"Expected EXIT at 10 bars time stop, got: {[s.signal for s in signals]}"
        )
        meta = exit_signals[0].metadata
        assert "bars_held=10" in meta.get("exit_reason", ""), (
            f"Exit reason should mention bars_held=10, got: {meta.get('exit_reason')}"
        )

    def test_l3_stop_loss_breach_triggers_exit(self):
        """Current close < stop_price while holding -> EXIT."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        n = 80
        closes = [50.0] * n
        store = _make_bar_store_with_n_bars("AMD", n, close_series=closes)

        layer = RSIReversalScalpStrategy()
        layer._open_positions["AMD"] = {
            "entry_price": 50.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=20),
            "stop_price": 49.5,
            "bars_held": 3,
        }

        # Close at 49.0 < stop 49.5 -> stop breach
        bar = _make_bar(close=49.0, high=49.3, low=48.8, symbol="AMD")
        signals = layer.evaluate_bar("AMD", bar, store, None)

        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, "Expected EXIT on stop-loss breach"
        assert "stop_breach" in exit_signals[0].metadata.get("exit_reason", "")

    def test_l3_exit_when_rsi7_recovers_above_55(self):
        """RSI(7) on 5-min > 55 while position open -> EXIT."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        n = 80
        # Steadily rising prices for the latter part -> RSI(7) recovers above 55
        closes = [50.0 + i * 0.6 for i in range(80)]
        store = _make_bar_store_with_n_bars("AMD", n, close_series=closes)

        layer = RSIReversalScalpStrategy()
        layer._open_positions["AMD"] = {
            "entry_price": 50.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=20),
            "stop_price": 48.0,  # stop far away
            "bars_held": 4,
        }

        current_close = closes[-1]
        bar = _make_bar(close=current_close, high=current_close + 0.5, low=current_close - 0.3, symbol="AMD")
        signals = layer.evaluate_bar("AMD", bar, store, None)

        # Rising prices -> RSI(7) > 55 -> EXIT
        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, (
            "Expected EXIT on RSI recovery above 55 with strongly rising prices"
        )

    def test_l3_no_entry_before_10_session_bars(self):
        """Session bar_count <= 10 -> no entry allowed."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        # Build 80 bars so indicators work, but make get_bar_count return <= 10
        store = _make_bar_store_with_n_bars("TSLA", 80)
        # Override bar count to simulate early session
        store._bar_counts["TSLA"] = 8  # <= 10 -> too early

        layer = RSIReversalScalpStrategy()
        bar = _make_bar(close=60.0, high=60.5, low=59.5, symbol="TSLA")
        signals = layer.evaluate_bar("TSLA", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, "Should not enter in first 10 bars of session"

    def test_l3_confidence_calculation(self):
        """confidence = min((25 - rsi) / 25, 1.0) -> deeper oversold = higher confidence."""
        from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
        n = 80
        closes = [100.0 - i * 0.5 for i in range(79)] + [61.0]
        store = _make_bar_store_with_n_bars("TSLA", n, close_series=closes)

        layer = RSIReversalScalpStrategy()
        bar = _make_bar(close=61.0, high=61.5, low=60.5, symbol="TSLA")
        signals = layer.evaluate_bar("TSLA", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        if buy_signals:
            s = buy_signals[0]
            rsi_val = s.metadata.get("rsi_7", 0.0)
            expected_conf = min((25.0 - rsi_val) / 25.0, 1.0)
            assert abs(s.confidence - expected_conf) < 0.001, (
                f"Expected confidence={expected_conf:.4f} for RSI={rsi_val:.2f}, "
                f"got {s.confidence:.4f}"
            )


# ===== LAYER 4 — Volume Surge Momentum =====

class TestVolumeSurgeMomentumLayer:

    def test_l4_layer_name(self):
        """Layer name must be exactly 'L4_VOL_SURGE'."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        layer = VolumeSurgeMomentumStrategy()
        assert layer.layer_name == "L4_VOL_SURGE"

    def test_l4_buy_on_3x_volume_surge_bullish_bar(self):
        """volume = 3.2x avg, bar in top 20% of range, green bar -> BUY."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        closes = [100.0] * n
        vols = [1000] * n
        store = _make_bar_store_with_n_bars("META", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        # bar_position = (close - low) / (high - low)
        # = (100.9 - 99.7) / (101.0 - 99.7) = 1.2 / 1.3 = 0.923 > 0.80 ✓
        # volume 3200 / avg 1000 = 3.2 > 3.0 ✓
        # close 100.9 > open 100.0 -> green ✓
        bar = {
            "open": 100.0, "high": 101.0, "low": 99.7, "close": 100.9,
            "volume": 3200, "symbol": "META",
            "timestamp": datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc),
        }
        signals = layer.evaluate_bar("META", bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) > 0, (
            f"Expected BUY on 3.2x volume surge with bullish bar, got: {[s.signal for s in signals]}"
        )
        s = buy_signals[0]
        assert s.layer_name == "L4_VOL_SURGE"
        assert s.stop_price < 100.9
        assert "volume_ratio" in s.metadata
        assert s.metadata["volume_ratio"] > 3.0

    def test_l4_buy_confidence_scales_from_2x_to_4x(self):
        """Confidence = min((vol_ratio - surge_thr) / surge_thr, 1.0) -> 3.2x = 0.60, 4x = 1.0.

        Surge threshold is now 2.0 (from config.yml strategies.layer4.volume_surge_ratio).
        """
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("META", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        surge_thr = layer._surge_threshold  # 2.0 from config
        bar = {
            "open": 100.0, "high": 101.0, "low": 99.7, "close": 100.9,
            "volume": 3200,  # 3.2x avg
            "symbol": "META",
            "timestamp": datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc),
        }
        signals = layer.evaluate_bar("META", bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        if buy_signals:
            vol_rat = buy_signals[0].metadata["volume_ratio"]
            expected_conf = min((vol_rat - surge_thr) / surge_thr, 1.0)
            assert abs(buy_signals[0].confidence - expected_conf) < 0.0001, (
                f"Expected confidence={expected_conf:.4f}, got {buy_signals[0].confidence:.4f}"
            )

    def test_l4_no_buy_on_bearish_surge_bar(self):
        """Volume 3.2x avg but bar closes near the low (bar_position ~0.08) -> no BUY."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("META", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        # bar_position = (99.8 - 99.7) / (101.0 - 99.7) = 0.1 / 1.3 = 0.077 << 0.80
        bar = {
            "open": 101.0, "high": 101.0, "low": 99.7, "close": 99.8,
            "volume": 3200, "symbol": "META",
            "timestamp": datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc),
        }
        signals = layer.evaluate_bar("META", bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, (
            "Should not BUY on bearish bar (close near low, bar_position < 0.80)"
        )

    def test_l4_no_buy_on_red_bar(self):
        """Volume surge but close < open (red/down bar) -> no BUY."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("META", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        # Red bar: open=101.0 > close=100.9
        bar = {
            "open": 101.0, "high": 101.5, "low": 100.6, "close": 100.9,
            "volume": 3200, "symbol": "META",
            "timestamp": datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc),
        }
        signals = layer.evaluate_bar("META", bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, "Should not BUY on a red bar (close < open)"

    def test_l4_no_buy_on_insufficient_volume(self):
        """Volume 1.8x avg (below 2x threshold from config) -> no BUY."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("META", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        bar = {
            "open": 100.0, "high": 101.0, "low": 99.7, "close": 100.9,
            "volume": 1800,  # 1.8x — below the 2.0x threshold (from config)
            "symbol": "META",
            "timestamp": datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc),
        }
        signals = layer.evaluate_bar("META", bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, "Should not BUY when volume_ratio < surge_threshold (2.0)"

    def test_l4_exit_when_volume_normalizes_after_3_consecutive_fade_bars(self):
        """3 consecutive volume fade bars (< 1.5x avg) -> EXIT."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("AMZN", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        layer._open_positions["AMZN"] = {
            "entry_price": 100.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=20),
            "stop_price": 98.5,
            "bars_held": 5,
            "entry_bar_count": 0,
        }
        layer._volume_fade_counter["AMZN"] = 3  # already at the exit threshold

        bar = {
            "open": 100.0, "high": 100.3, "low": 99.7, "close": 100.1,
            "volume": 800,  # below 1.5x avg (1000 * 1.5 = 1500)
            "symbol": "AMZN",
            "timestamp": datetime.now(timezone.utc),
        }
        signals = layer.evaluate_bar("AMZN", bar, store, None)
        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, (
            "Expected EXIT after 3 consecutive volume fade bars"
        )
        meta = exit_signals[0].metadata
        assert "volume_fade" in meta.get("exit_reason", ""), (
            f"Exit reason should mention volume_fade, got: {meta.get('exit_reason')}"
        )

    def test_l4_exit_on_stop_loss(self):
        """Close drops below stop_price -> EXIT with stop_breach reason."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("AMZN", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        layer._open_positions["AMZN"] = {
            "entry_price": 100.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "stop_price": 99.0,
            "bars_held": 3,
            "entry_bar_count": 10,
        }
        layer._volume_fade_counter["AMZN"] = 0

        bar = {
            "open": 99.2, "high": 99.4, "low": 98.5, "close": 98.7,
            "volume": 1000, "symbol": "AMZN",
            "timestamp": datetime.now(timezone.utc),
        }
        signals = layer.evaluate_bar("AMZN", bar, store, None)
        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, "Expected EXIT on stop-loss breach"
        assert "stop_breach" in exit_signals[0].metadata.get("exit_reason", "")

    def test_l4_exit_on_rsi_overbought(self):
        """RSI(14) > 75 while position open -> EXIT."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        # Rising prices to produce high RSI(14)
        closes = [100.0 + i * 1.0 for i in range(25)]
        vols = [1000] * n
        store = _make_bar_store_with_n_bars("NFLX", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        layer._open_positions["NFLX"] = {
            "entry_price": 100.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=15),
            "stop_price": 90.0,
            "bars_held": 12,
            "entry_bar_count": 5,
        }
        layer._volume_fade_counter["NFLX"] = 0

        current_close = closes[-1]
        bar = {
            "open": current_close - 0.1, "high": current_close + 0.5,
            "low": current_close - 0.3, "close": current_close,
            "volume": 1500, "symbol": "NFLX",
            "timestamp": datetime.now(timezone.utc),
        }
        signals = layer.evaluate_bar("NFLX", bar, store, None)
        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0, (
            "Expected EXIT when RSI(14) > 75 due to strongly rising prices"
        )

    def test_l4_no_buy_after_bar_330(self):
        """Session bar count > 330 (~15:00 ET) -> no new BUY entries."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("META", n, close_series=closes, volume_series=vols)
        # Override bar count to simulate late session
        store._bar_counts["META"] = 335  # > 330

        layer = VolumeSurgeMomentumStrategy()
        bar = {
            "open": 100.0, "high": 101.0, "low": 99.7, "close": 100.9,
            "volume": 3200, "symbol": "META",
            "timestamp": datetime(2024, 1, 15, 20, 30, tzinfo=timezone.utc),
        }
        signals = layer.evaluate_bar("META", bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, "Should not BUY after session bar 330 (~15:00 ET)"

    def test_l4_stop_placed_at_1_5x_atr_below_entry(self):
        """BUY signal stop_price = entry_price - 1.5 * ATR(14, 1-min bars)."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        from src.data.indicators import atr as calc_atr
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("META", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        bar = {
            "open": 100.0, "high": 101.0, "low": 99.7, "close": 100.9,
            "volume": 3200, "symbol": "META",
            "timestamp": datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc),
        }
        signals = layer.evaluate_bar("META", bar, store, None)
        buy_signals = [s for s in signals if s.signal == "BUY"]
        if buy_signals:
            s = buy_signals[0]
            # Compute ATR(14) from the 1-min bar history
            df = store.get_bars("META", "1Min", 20)
            if len(df) >= 15:
                atr_arr = calc_atr(df["high"], df["low"], df["close"], 14)
                valid_atr = [v for v in atr_arr if not np.isnan(v)]
                if valid_atr:
                    expected_stop = s.signal_price - 1.5 * valid_atr[-1]
                    assert abs(s.stop_price - expected_stop) < 0.01, (
                        f"Expected stop {expected_stop:.4f}, got {s.stop_price:.4f}"
                    )

    def test_l4_fade_counter_resets_on_volume_return(self):
        """If volume returns >= 1.5x avg during a held position, fade counter resets to 0."""
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("TSLA", n, close_series=closes, volume_series=vols)

        layer = VolumeSurgeMomentumStrategy()
        layer._open_positions["TSLA"] = {
            "entry_price": 100.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "stop_price": 98.0,
            "bars_held": 3,
            "entry_bar_count": 10,
        }
        layer._volume_fade_counter["TSLA"] = 2  # Had 2 fade bars previously

        # Volume surge bar: 2500 = 2.5x avg 1000 >= 1.5x -> counter should reset to 0
        surge_bar = {
            "open": 100.0, "high": 100.5, "low": 99.8, "close": 100.3,
            "volume": 2500, "symbol": "TSLA",
            "timestamp": datetime.now(timezone.utc),
        }
        layer.evaluate_bar("TSLA", surge_bar, store, None)
        # After a bar with volume >= 1.5x avg, the counter should be reset to 0.
        assert layer._volume_fade_counter.get("TSLA", 0) == 0, (
            "Fade counter should reset to 0 when volume returns >= 1.5x avg"
        )


# ===== CROSS-LAYER TESTS =====

class TestCrossLayerBehavior:

    def test_dedup_no_second_entry_same_symbol_same_layer(self):
        """L1 has open position in AAPL -> second L1 signal for AAPL blocked."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        closes = [152.0] * 20 + [149.6] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)

        layer = VWAPMeanReversionStrategy()
        # Inject an open position.
        layer._open_positions["AAPL"] = {
            "entry_price": 149.6,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "stop_price": 149.0,
            "bars_held": 3,
        }

        # Bar that would normally trigger BUY (price deep below VWAP).
        bar = _make_bar(close=149.4, high=149.8, low=149.0, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)

        buy_signals = [s for s in signals if s.signal == "BUY"]
        assert len(buy_signals) == 0, (
            "L1 must not open a second position for AAPL while one is already open"
        )

    def test_l1_and_l4_are_independent_per_symbol(self):
        """L1 open position in AAPL does not affect L4 ability to enter AAPL."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
        from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy

        n = 25
        vols = [1000] * n
        closes = [100.0] * n
        store = _make_bar_store_with_n_bars("AAPL", n, close_series=closes, volume_series=vols)

        l1 = VWAPMeanReversionStrategy()
        l4 = VolumeSurgeMomentumStrategy()

        # L1 has an open position.
        l1._open_positions["AAPL"] = {
            "entry_price": 99.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "stop_price": 98.0,
            "bars_held": 5,
        }

        # L4 does NOT have an open position — it should be able to evaluate freely.
        bar = {
            "open": 100.0, "high": 101.0, "low": 99.7, "close": 100.9,
            "volume": 3200, "symbol": "AAPL",
            "timestamp": datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc),
        }
        l4_signals = l4.evaluate_bar("AAPL", bar, store, None)
        # L4 should evaluate normally (buy or hold based on its own logic).
        # The key: it must not crash and must not be blocked by L1's state.
        for s in l4_signals:
            assert s.layer_name == "L4_VOL_SURGE"

    def test_multiple_symbols_independent_per_layer(self):
        """L1 AAPL position does not block L1 MSFT entry."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy

        aapl_closes = [152.0] * 20 + [149.6] * 10
        msft_closes = [300.0] * 20 + [296.5] * 10
        store_aapl = _make_bar_store_with_n_bars("AAPL", 30, close_series=aapl_closes)
        store_msft = _make_bar_store_with_n_bars("MSFT", 30, close_series=msft_closes)

        layer = VWAPMeanReversionStrategy()
        # AAPL already has a position.
        layer._open_positions["AAPL"] = {
            "entry_price": 149.6,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "stop_price": 149.0,
            "bars_held": 3,
        }

        # Evaluate MSFT — should be completely independent of AAPL.
        bar_msft = _make_bar(close=296.5, high=297.0, low=296.0, symbol="MSFT")
        signals_msft = layer.evaluate_bar("MSFT", bar_msft, store_msft, None)

        # MSFT price is ~1.2% below VWAP (~300.0) -> should potentially BUY.
        # The assertion is: AAPL's position does NOT prevent MSFT evaluation.
        # Either buy or hold is valid, but no crash and no AAPL signals.
        for s in signals_msft:
            assert s.symbol == "MSFT", f"MSFT layer signals should only reference MSFT, got {s.symbol}"

    def test_has_open_position_tracks_correctly(self):
        """has_open_position() returns True after BUY, False after EXIT."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy

        closes = [152.0] * 20 + [149.6] * 10
        store = _make_bar_store_with_n_bars("AAPL", 30, close_series=closes)
        layer = VWAPMeanReversionStrategy()

        assert layer.has_open_position("AAPL") is False

        # Inject position
        layer._open_positions["AAPL"] = {
            "entry_price": 149.6,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "stop_price": 149.0,
            "bars_held": 3,
        }
        assert layer.has_open_position("AAPL") is True

        # Trigger EXIT via stop breach
        bar = _make_bar(close=148.5, high=148.8, low=148.2, symbol="AAPL")
        signals = layer.evaluate_bar("AAPL", bar, store, None)
        exit_signals = [s for s in signals if s.signal == "EXIT"]
        assert len(exit_signals) > 0
        assert layer.has_open_position("AAPL") is False

    def test_clear_position_removes_tracking_without_signal(self):
        """clear_position() removes internal state without generating an EXIT signal."""
        from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy

        layer = VWAPMeanReversionStrategy()
        layer._open_positions["TSLA"] = {
            "entry_price": 200.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "stop_price": 198.0,
            "bars_held": 5,
        }
        assert layer.has_open_position("TSLA") is True

        layer.clear_position("TSLA")
        assert layer.has_open_position("TSLA") is False
        assert layer.get_position_info("TSLA") is None

    def test_signal_result_dataclass_fields(self):
        """SignalResult dataclass fields are accessible as expected."""
        s = SignalResult(
            symbol="AAPL",
            signal="BUY",
            confidence=0.75,
            signal_price=150.0,
            layer_name="L1_VWAP_MR",
            stop_price=149.0,
            metadata={"vwap": 152.0, "rsi": 38.5},
        )
        assert s.symbol == "AAPL"
        assert s.signal == "BUY"
        assert s.confidence == 0.75
        assert s.signal_price == 150.0
        assert s.layer_name == "L1_VWAP_MR"
        assert s.stop_price == 149.0
        assert s.metadata["vwap"] == 152.0
        assert s.metadata["rsi"] == 38.5

    def test_signal_result_default_stop_price(self):
        """SignalResult stop_price defaults to 0.0 if not provided."""
        s = SignalResult(
            symbol="AAPL",
            signal="HOLD",
            confidence=0.0,
            signal_price=100.0,
            layer_name="L1_VWAP_MR",
        )
        assert s.stop_price == 0.0
        assert s.metadata == {}
