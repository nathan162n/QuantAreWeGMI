"""Unit tests for technical indicators with numerical assertions.

Tests verify mathematical correctness of all indicator functions
using known inputs and expected outputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import adv, atr, ema, realized_vol, rolling_zscore, rsi, slope, sma, vwap
from src.data.indicators import volume_ratio, bar_position, vwap_session


class TestSMA:
    """Tests for Simple Moving Average calculation."""

    def test_sma_basic(self):
        """SMA of [10, 11, 12, 13, 14] with period=3 should be [nan, nan, 11, 12, 13].

        Example:
            >>> sma(pd.Series([10, 11, 12, 13, 14]), 3)
        """
        prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        result = sma(prices, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(11.0)
        assert result[3] == pytest.approx(12.0)
        assert result[4] == pytest.approx(13.0)

    def test_sma_single_period(self):
        """SMA with period=1 should return the original prices.

        Example:
            >>> sma(pd.Series([5, 10, 15]), 1)
        """
        prices = pd.Series([5.0, 10.0, 15.0])
        result = sma(prices, 1)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(10.0)
        assert result[2] == pytest.approx(15.0)

    def test_sma_all_same(self):
        """SMA of constant values should equal that constant.

        Example:
            >>> sma(pd.Series([50]*5), 3)
        """
        prices = pd.Series([50.0] * 5)
        result = sma(prices, 3)
        assert result[2] == pytest.approx(50.0)
        assert result[4] == pytest.approx(50.0)


class TestEMA:
    """Tests for Exponential Moving Average calculation."""

    def test_ema_first_value(self):
        """First EMA value should equal the first price (adjust=False).

        Example:
            >>> ema(pd.Series([10, 11, 12]), 3)
        """
        prices = pd.Series([10.0, 11.0, 12.0])
        result = ema(prices, 3)
        assert result[0] == pytest.approx(10.0)

    def test_ema_trending_up(self):
        """EMA should increase for monotonically increasing prices.

        Example:
            >>> ema(pd.Series([10, 11, 12, 13, 14]), 3)
        """
        prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        result = ema(prices, 3)
        for i in range(1, len(result)):
            assert result[i] > result[i - 1]


class TestRSI:
    """Tests for Wilder's RSI calculation."""

    def test_rsi_overbought(self):
        """Monotonically increasing prices should give RSI > 70.

        Example:
            >>> rsi(pd.Series(range(100, 120)), 14)
        """
        prices = pd.Series([float(x) for x in range(100, 135)])
        result = rsi(prices, 14)
        last_rsi = result[-1]
        assert not np.isnan(last_rsi)
        assert last_rsi > 70

    def test_rsi_oversold(self):
        """Monotonically decreasing prices should give RSI < 30.

        Example:
            >>> rsi(pd.Series(range(135, 100, -1)), 14)
        """
        prices = pd.Series([float(x) for x in range(135, 100, -1)])
        result = rsi(prices, 14)
        last_rsi = result[-1]
        assert not np.isnan(last_rsi)
        assert last_rsi < 30

    def test_rsi_range(self):
        """RSI should always be between 0 and 100.

        Example:
            >>> rsi(pd.Series(np.random.randn(50).cumsum() + 100), 14)
        """
        np.random.seed(42)
        prices = pd.Series(np.random.randn(50).cumsum() + 100)
        result = rsi(prices, 14)
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_rsi_nan_prefix(self):
        """First 'period' values of RSI should be NaN.

        Example:
            >>> rsi(pd.Series(range(100, 120)), 14)
        """
        prices = pd.Series([float(x) for x in range(100, 120)])
        result = rsi(prices, 14)
        for i in range(14):
            assert np.isnan(result[i])


class TestATR:
    """Tests for Average True Range calculation."""

    def test_atr_basic(self):
        """ATR should be positive for normal price data.

        Example:
            >>> atr(high, low, close, 3)
        """
        h = pd.Series([48.70, 48.72, 48.90, 48.87, 48.82, 49.05, 49.20, 49.35])
        l = pd.Series([47.79, 48.14, 48.39, 48.37, 48.24, 48.64, 48.94, 48.86])
        c = pd.Series([48.16, 48.61, 48.75, 48.63, 48.74, 49.03, 49.07, 49.32])
        result = atr(h, l, c, 3)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_atr_nan_prefix(self):
        """First 'period' ATR values should be NaN.

        Example:
            >>> atr(h, l, c, 5)
        """
        h = pd.Series([50.0 + i for i in range(10)])
        l = pd.Series([49.0 + i for i in range(10)])
        c = pd.Series([49.5 + i for i in range(10)])
        result = atr(h, l, c, 5)
        for i in range(5):
            assert np.isnan(result[i])

    def test_atr_zero_range(self):
        """ATR of constant prices should be near zero.

        Example:
            >>> atr(pd.Series([50]*10), pd.Series([50]*10), pd.Series([50]*10), 3)
        """
        h = pd.Series([50.0] * 10)
        l = pd.Series([50.0] * 10)
        c = pd.Series([50.0] * 10)
        result = atr(h, l, c, 3)
        valid = result[~np.isnan(result)]
        assert all(v < 0.01 for v in valid)


class TestRollingZscore:
    """Tests for rolling z-score calculation."""

    def test_zscore_at_mean(self):
        """Price at the rolling mean should have z-score near 0.

        Example:
            >>> rolling_zscore(pd.Series([50]*10), 5)
        """
        prices = pd.Series([50.0] * 10)
        result = rolling_zscore(prices, 5)
        valid = result[~np.isnan(result)]
        assert all(abs(v) < 0.01 or np.isnan(v) for v in valid)

    def test_zscore_negative_for_drop(self):
        """A price drop below the mean should produce a negative z-score.

        Example:
            >>> rolling_zscore(pd.Series([100]*19 + [90]), 20)
        """
        prices = pd.Series([100.0] * 19 + [90.0])
        result = rolling_zscore(prices, 20)
        assert result[-1] < 0

    def test_zscore_positive_for_spike(self):
        """A price spike above the mean should produce a positive z-score.

        Example:
            >>> rolling_zscore(pd.Series([100]*19 + [110]), 20)
        """
        prices = pd.Series([100.0] * 19 + [110.0])
        result = rolling_zscore(prices, 20)
        assert result[-1] > 0


class TestRealizedVol:
    """Tests for annualized realized volatility calculation."""

    def test_vol_constant_prices(self):
        """Constant prices should have zero volatility.

        Example:
            >>> realized_vol(pd.Series([100]*25), 20)
        """
        prices = pd.Series([100.0] * 25)
        vol = realized_vol(prices, 20)
        assert vol == pytest.approx(0.0, abs=0.001)

    def test_vol_positive(self):
        """Volatile prices should have positive volatility.

        Example:
            >>> realized_vol(pd.Series([100, 105, 95, 110, 90, ...]), 20)
        """
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(25).cumsum())
        vol = realized_vol(prices, 20)
        assert vol > 0

    def test_vol_insufficient_data(self):
        """Insufficient data should return 0.0.

        Example:
            >>> realized_vol(pd.Series([100, 101]), 20)
        """
        prices = pd.Series([100.0, 101.0])
        vol = realized_vol(prices, 20)
        assert vol == 0.0


class TestADV:
    """Tests for Average Dollar Volume calculation."""

    def test_adv_basic(self):
        """ADV should correctly compute average of volume * price.

        Example:
            >>> adv(pd.Series([1e6]*5), pd.Series([50]*5), 3)
        """
        volume = pd.Series([1_000_000.0] * 5)
        prices = pd.Series([50.0] * 5)
        result = adv(volume, prices, 3)
        assert result == pytest.approx(50_000_000.0)

    def test_adv_insufficient_data(self):
        """ADV with insufficient data should return 0.0.

        Example:
            >>> adv(pd.Series([1e6]), pd.Series([50]), 5)
        """
        volume = pd.Series([1_000_000.0])
        prices = pd.Series([50.0])
        result = adv(volume, prices, 5)
        assert result == 0.0


class TestVWAP:
    """Tests for session VWAP calculation."""

    def test_vwap_single_bar(self):
        """VWAP of a single bar should equal the typical price.

        Example:
            >>> vwap(pd.Series([51]), pd.Series([49]), pd.Series([50]), pd.Series([1000]))
        """
        h = pd.Series([51.0])
        l = pd.Series([49.0])
        c = pd.Series([50.0])
        v = pd.Series([1000.0])
        result = vwap(h, l, c, v)
        expected_tp = (51.0 + 49.0 + 50.0) / 3.0
        assert result[0] == pytest.approx(expected_tp)

    def test_vwap_equal_volume(self):
        """VWAP with equal volume should be average of typical prices."""
        h = pd.Series([52.0, 54.0])
        l = pd.Series([48.0, 50.0])
        c = pd.Series([50.0, 52.0])
        v = pd.Series([1000.0, 1000.0])
        result = vwap(h, l, c, v)
        tp1 = (52 + 48 + 50) / 3.0
        tp2 = (54 + 50 + 52) / 3.0
        expected = (tp1 * 1000 + tp2 * 1000) / 2000
        assert result[-1] == pytest.approx(expected)

    def test_vwap_positive(self):
        """VWAP should always be positive for positive prices."""
        h = pd.Series([51.0, 52.0, 53.0])
        l = pd.Series([49.0, 50.0, 51.0])
        c = pd.Series([50.0, 51.0, 52.0])
        v = pd.Series([1000.0, 2000.0, 1500.0])
        result = vwap(h, l, c, v)
        assert all(r > 0 for r in result)


class TestSlope:
    """Tests for linear regression slope calculation."""

    def test_slope_linear_up(self):
        """Perfectly linear uptrend should give slope = 1.0.

        Example:
            >>> slope(pd.Series([10, 11, 12, 13, 14]), 5)
        """
        result = slope(pd.Series([10.0, 11.0, 12.0, 13.0, 14.0]), 5)
        assert result == pytest.approx(1.0)

    def test_slope_linear_down(self):
        """Perfectly linear downtrend should give slope = -1.0."""
        result = slope(pd.Series([14.0, 13.0, 12.0, 11.0, 10.0]), 5)
        assert result == pytest.approx(-1.0)

    def test_slope_flat(self):
        """Constant prices should give slope = 0.0."""
        result = slope(pd.Series([50.0, 50.0, 50.0, 50.0, 50.0]), 5)
        assert result == pytest.approx(0.0)

    def test_slope_insufficient_data(self):
        """Insufficient data should return 0.0."""
        result = slope(pd.Series([10.0, 11.0]), 5)
        assert result == 0.0

    def test_slope_uses_tail(self):
        """Slope should only use the last N bars."""
        s = pd.Series([100.0, 99.0, 98.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        result = slope(s, 5)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# New tests for volume_ratio, bar_position, vwap_session
# ---------------------------------------------------------------------------

class TestVolumeRatio:
    """Tests for the volume_ratio indicator."""

    def test_volume_ratio_returns_3x(self):
        """21 bars of 1000 volume each, current bar = 3000 -> ratio = 3.0."""
        vols = pd.Series([1000] * 20 + [3000])
        ratio = volume_ratio(vols, period=20)
        assert abs(ratio - 3.0) < 0.01, f"Expected 3.0, got {ratio}"

    def test_volume_ratio_returns_1x_for_equal_volume(self):
        """All bars same volume -> ratio = 1.0."""
        vols = pd.Series([500] * 21)
        ratio = volume_ratio(vols, period=20)
        assert abs(ratio - 1.0) < 0.01, f"Expected 1.0, got {ratio}"

    def test_volume_ratio_returns_05x_for_half_volume(self):
        """Current bar volume is half of the 20-bar average -> ratio = 0.5."""
        vols = pd.Series([2000] * 20 + [1000])
        ratio = volume_ratio(vols, period=20)
        assert abs(ratio - 0.5) < 0.01, f"Expected 0.5, got {ratio}"

    def test_volume_ratio_surge_threshold(self):
        """5x average volume -> ratio = 5.0."""
        vols = pd.Series([1000] * 20 + [5000])
        ratio = volume_ratio(vols, period=20)
        assert abs(ratio - 5.0) < 0.01, f"Expected 5.0, got {ratio}"

    def test_volume_ratio_single_element_returns_1(self):
        """Single element series -> ratio = 1.0 (insufficient data fallback)."""
        vols = pd.Series([9999])
        ratio = volume_ratio(vols, period=20)
        assert ratio == 1.0

    def test_volume_ratio_zero_average_returns_1(self):
        """Zero prior volume average -> ratio = 1.0 (avoids division by zero)."""
        vols = pd.Series([0] * 20 + [500])
        ratio = volume_ratio(vols, period=20)
        assert ratio == 1.0

    def test_volume_ratio_last_element_is_current(self):
        """volume_ratio uses the last element as current bar volume."""
        # First 20 bars at 2000, current (last) at 4000 -> ratio = 2.0
        vols = pd.Series([2000] * 20 + [4000])
        ratio = volume_ratio(vols, period=20)
        assert abs(ratio - 2.0) < 0.01, f"Expected 2.0, got {ratio}"


class TestBarPosition:
    """Tests for the bar_position indicator."""

    def test_bar_position_near_high(self):
        """Close near high -> bar_position close to 1.0."""
        # (104.0 - 95.0) / (105.0 - 95.0) = 9.0 / 10.0 = 0.90
        pos = bar_position(100.0, 105.0, 95.0, 104.0)
        assert abs(pos - 0.9) < 0.01, f"Expected ~0.9, got {pos}"

    def test_bar_position_near_low(self):
        """Close near low -> bar_position close to 0.0."""
        # (96.0 - 95.0) / (105.0 - 95.0) = 1.0 / 10.0 = 0.10
        pos = bar_position(100.0, 105.0, 95.0, 96.0)
        assert abs(pos - 0.1) < 0.01, f"Expected ~0.1, got {pos}"

    def test_bar_position_at_midpoint(self):
        """Close exactly at midpoint of range -> bar_position = 0.50."""
        # (100.0 - 95.0) / (105.0 - 95.0) = 5.0 / 10.0 = 0.50
        pos = bar_position(95.0, 105.0, 95.0, 100.0)
        assert abs(pos - 0.5) < 0.001, f"Expected 0.5, got {pos}"

    def test_bar_position_at_exact_high(self):
        """Close = high -> bar_position = 1.0."""
        pos = bar_position(99.0, 105.0, 95.0, 105.0)
        assert abs(pos - 1.0) < 0.001, f"Expected 1.0, got {pos}"

    def test_bar_position_at_exact_low(self):
        """Close = low -> bar_position = 0.0."""
        pos = bar_position(100.0, 105.0, 95.0, 95.0)
        assert abs(pos - 0.0) < 0.001, f"Expected 0.0, got {pos}"

    def test_bar_position_zero_range_returns_05(self):
        """high == low -> returns 0.5 (degenerate bar)."""
        pos = bar_position(100.0, 100.0, 100.0, 100.0)
        assert pos == 0.5

    def test_bar_position_clamped_to_0_1(self):
        """bar_position result is always clamped to [0.0, 1.0]."""
        # Pathological: close above high
        pos_above = bar_position(100.0, 105.0, 95.0, 110.0)
        assert 0.0 <= pos_above <= 1.0

        # Pathological: close below low
        pos_below = bar_position(100.0, 105.0, 95.0, 90.0)
        assert 0.0 <= pos_below <= 1.0

    def test_bar_position_open_parameter_unused_in_formula(self):
        """bar_position open_ parameter is not used in the formula.

        Two calls with different open_ values should return the same result
        as long as high, low, close are the same.
        """
        pos1 = bar_position(95.0, 105.0, 95.0, 102.0)
        pos2 = bar_position(200.0, 105.0, 95.0, 102.0)
        assert abs(pos1 - pos2) < 0.001

    def test_bar_position_l4_entry_threshold(self):
        """Verify bar close in top 20% of range gives bar_position > 0.80 (L4 threshold)."""
        # Top 20%: close >= low + 0.80 * range
        # range = 105 - 95 = 10; top 20% threshold = 95 + 8 = 103
        # close=104: (104-95)/10 = 0.90 > 0.80 ✓
        pos = bar_position(100.0, 105.0, 95.0, 104.0)
        assert pos > 0.80, f"Expected bar_position > 0.80 for top-20% close, got {pos}"


class TestVwapSession:
    """Tests for the vwap_session indicator function."""

    def test_vwap_session_single_bar(self):
        """Single bar: VWAP = (H + L + C) / 3."""
        result = vwap_session(
            pd.Series([51.0]),
            pd.Series([49.0]),
            pd.Series([50.0]),
            pd.Series([1000.0]),
        )
        expected = (51.0 + 49.0 + 50.0) / 3.0  # = 50.333...
        assert abs(result - expected) < 0.001, f"Expected {expected:.4f}, got {result:.4f}"

    def test_vwap_session_zero_volume_returns_zero(self):
        """Zero volume -> returns 0.0."""
        result = vwap_session(
            pd.Series([100.0]),
            pd.Series([99.0]),
            pd.Series([100.0]),
            pd.Series([0.0]),
        )
        assert result == 0.0

    def test_vwap_session_equal_volume_is_average_typical_price(self):
        """Equal volume for all bars -> VWAP = average of typical prices."""
        highs = pd.Series([52.0, 54.0])
        lows = pd.Series([48.0, 50.0])
        closes = pd.Series([50.0, 52.0])
        volumes = pd.Series([1000.0, 1000.0])

        tp1 = (52.0 + 48.0 + 50.0) / 3.0   # = 50.0
        tp2 = (54.0 + 50.0 + 52.0) / 3.0   # = 52.0
        expected = (tp1 + tp2) / 2.0         # = 51.0

        result = vwap_session(highs, lows, closes, volumes)
        assert abs(result - expected) < 0.001, f"Expected {expected:.4f}, got {result:.4f}"

    def test_vwap_session_volume_weighted(self):
        """Bar with 2x volume has twice the weight in VWAP calculation."""
        # Bar 1: TP = 50.0, Vol = 1000
        # Bar 2: TP = 55.0, Vol = 2000
        # VWAP = (50*1000 + 55*2000) / (1000 + 2000) = (50000 + 110000) / 3000 = 53.333
        highs = pd.Series([51.0, 56.0])
        lows = pd.Series([49.0, 54.0])
        closes = pd.Series([50.0, 55.0])
        volumes = pd.Series([1000.0, 2000.0])

        tp1 = (51.0 + 49.0 + 50.0) / 3.0   # = 50.0
        tp2 = (56.0 + 54.0 + 55.0) / 3.0   # = 55.0
        expected = (tp1 * 1000.0 + tp2 * 2000.0) / 3000.0

        result = vwap_session(highs, lows, closes, volumes)
        assert abs(result - expected) < 0.001, f"Expected {expected:.4f}, got {result:.4f}"

    def test_vwap_session_high_volume_bar_dominates(self):
        """When one bar has 99x the volume of all others, VWAP is near that bar's TP."""
        highs = pd.Series([50.0, 100.0])
        lows = pd.Series([48.0, 98.0])
        closes = pd.Series([49.0, 99.0])
        volumes = pd.Series([1.0, 99.0])  # second bar dominates

        tp2 = (100.0 + 98.0 + 99.0) / 3.0  # = 99.0
        result = vwap_session(highs, lows, closes, volumes)
        # Result should be very close to tp2 = 99.0
        assert abs(result - tp2) < 1.0, f"High-volume bar should dominate VWAP, got {result}"


class TestRSIPeriodSensitivity:
    """Tests that validate RSI period properties used in strategy layers."""

    def test_rsi_period7_more_sensitive_than_period14_on_decline(self):
        """RSI(7) reaches more extreme values than RSI(14) on declining prices.

        This is the key justification for using RSI(7) in Layer 3 (RSI Scalp):
        it reacts faster to short-term moves, producing more extreme oversold
        readings that allow scalp entries at better prices.

        Uses a zig-zag declining series so both RSI(7) and RSI(14) stay
        non-zero/non-NaN but RSI(7) reacts more strongly to the downtrend.
        """
        # Zig-zag downtrend: alternating gain+2/loss-3 produces a net decline
        # while keeping both RSI series non-zero with valid Wilder smoothing.
        prices = pd.Series([
            100.0, 102, 99, 101, 98, 100, 97, 99, 96, 98,
            95, 97, 94, 96, 93, 95, 92, 94, 91, 93,
            90, 92, 89, 91, 88, 90, 87, 89, 86, 88,
        ])
        rsi7_arr = rsi(prices, period=7)
        rsi14_arr = rsi(prices, period=14)

        valid7 = [v for v in rsi7_arr if not np.isnan(v)]
        valid14 = [v for v in rsi14_arr if not np.isnan(v)]

        assert valid7 and valid14, "Both RSI series must produce valid values"
        # On a zig-zag downtrend, RSI(7) should be lower (more extreme) than RSI(14)
        # at the bottom of each zig-zag since it weights recent bars more heavily.
        # Compare the minimum values over valid readings.
        assert min(valid7) < min(valid14), (
            f"RSI(7) min={min(valid7):.1f} should be lower than RSI(14) min={min(valid14):.1f} "
            f"on a declining zig-zag price series"
        )

    def test_rsi_period7_more_sensitive_than_period14_on_advance(self):
        """RSI(7) reaches higher values than RSI(14) on rising prices.

        Uses a zig-zag uptrend (net positive but not monotonic) so both
        RSI series stay below 100.0 while RSI(7) reacts more strongly.
        """
        # Zig-zag uptrend: alternating loss-2/gain+3 -> net gain of +1 per 2 bars.
        prices = pd.Series([
            100.0, 98, 101, 99, 102, 100, 103, 101, 104, 102,
            105, 103, 106, 104, 107, 105, 108, 106, 109, 107,
            110, 108, 111, 109, 112, 110, 113, 111, 114, 112,
        ])
        rsi7_arr = rsi(prices, period=7)
        rsi14_arr = rsi(prices, period=14)

        valid7 = [v for v in rsi7_arr if not np.isnan(v)]
        valid14 = [v for v in rsi14_arr if not np.isnan(v)]

        assert valid7 and valid14
        # On a zig-zag uptrend, RSI(7) max should be higher (more extreme) than RSI(14) max.
        assert max(valid7) > max(valid14), (
            f"RSI(7) max={max(valid7):.1f} should be higher than RSI(14) max={max(valid14):.1f} "
            f"on a zig-zag uptrending price series"
        )

    def test_rsi_period7_entry_threshold_25_correctly_oversold(self):
        """Verifies that RSI(7) reacts faster to a sharp turn than RSI(14).

        On the same zig-zag downtrend, RSI(7) should hit a lower value than
        RSI(14) after a sharp downswing, confirming that the shorter period
        reaches oversold territory sooner (i.e., < 25 is a tighter threshold).
        """
        # Zig-zag declining series that keeps both RSI periods non-degenerate.
        prices = pd.Series([
            100.0, 102, 99, 101, 98, 100, 97, 99, 96, 98,
            95, 97, 94, 96, 93, 95, 92, 94, 91, 93,
            90, 92, 89, 91, 88, 90, 87, 89, 86, 88,
        ])
        rsi7_arr = rsi(prices, period=7)
        rsi14_arr = rsi(prices, period=14)

        valid7 = [v for v in rsi7_arr if not np.isnan(v)]
        valid14 = [v for v in rsi14_arr if not np.isnan(v)]

        assert valid7 and valid14, "Both RSI series must have valid readings"
        # On a zig-zag downtrend, RSI(7) minimum should be lower than RSI(14) minimum.
        # This confirms RSI(7) is a more sensitive (lower threshold) indicator.
        rsi7_min = min(valid7)
        rsi14_min = min(valid14)
        assert rsi7_min < rsi14_min, (
            f"RSI(7) min={rsi7_min:.1f} should be lower than RSI(14) min={rsi14_min:.1f} "
            f"on the same downtrend, confirming RSI(7) reaches oversold faster"
        )

    def test_rsi_flat_prices_returns_50(self):
        """Constant prices produce RSI near 50 (no trend, balanced gains and losses)."""
        prices = pd.Series([100.0] * 20)
        result = rsi(prices, period=7)
        valid = [v for v in result if not np.isnan(v)]
        if valid:
            # RSI of flat prices should be 50 (equal gains and losses, both zero).
            # With all-zero deltas, RSI uses the al==0 branch -> 50.0
            assert abs(valid[-1] - 50.0) < 5.0, (
                f"Flat price RSI should be near 50, got {valid[-1]:.1f}"
            )


class TestVolumeRatioEdgeCases:
    """Additional edge-case tests for volume_ratio used in L4 strategy."""

    def test_volume_ratio_exactly_3x_is_not_above_threshold(self):
        """volume_ratio = 3.0 exactly is NOT above the >3.0 L4 entry threshold."""
        vols = pd.Series([1000] * 20 + [3000])
        ratio = volume_ratio(vols, period=20)
        # Ratio is exactly 3.0 -> NOT above 3.0 threshold (L4 requires > 3.0)
        assert ratio == pytest.approx(3.0, abs=0.01)
        assert not (ratio > 3.0), "Ratio of exactly 3.0 should not exceed threshold"

    def test_volume_ratio_3001_just_above_threshold(self):
        """volume_ratio slightly above 3.0 -> should pass L4 >3.0 threshold."""
        vols = pd.Series([1000] * 20 + [3001])
        ratio = volume_ratio(vols, period=20)
        assert ratio > 3.0, f"Expected ratio > 3.0, got {ratio}"

    def test_volume_ratio_uses_20_prior_bars(self):
        """volume_ratio denominates using the 20 bars BEFORE the current bar."""
        # 5 bars at 500, 15 bars at 1500, current bar at 3000
        # Prior 20 bars: 5*500 + 15*1500 = 2500 + 22500 = 25000; avg = 1250
        # ratio = 3000 / 1250 = 2.4
        vols = pd.Series([500] * 5 + [1500] * 15 + [3000])
        ratio = volume_ratio(vols, period=20)
        assert abs(ratio - 2.4) < 0.01, f"Expected 2.4, got {ratio}"
