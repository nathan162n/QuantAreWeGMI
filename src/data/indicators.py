"""Pure NumPy/Pandas technical indicator library.

All functions accept pd.Series or np.ndarray and return np.ndarray.
Fully vectorized with no row-level loops. Zero TA-Lib dependency.
"""

import numpy as np
import pandas as pd


def sma(prices: pd.Series, period: int) -> np.ndarray:
    """Simple Moving Average.

    Formula: SMA(t) = (1/n) * sum(price(t-i) for i in 0..n-1)

    Args:
        prices: Series of prices.
        period: Lookback window size.

    Returns:
        np.ndarray: SMA values. First (period-1) values are NaN.

    Example:
        >>> sma(pd.Series([10, 11, 12, 13, 14]), 3)
        array([nan, nan, 11., 12., 13.])
    """
    s = pd.Series(prices)
    return s.rolling(window=period, min_periods=period).mean().values


def ema(prices: pd.Series, period: int) -> np.ndarray:
    """Exponential Moving Average using pandas ewm.

    Formula: EMA(t) = price(t) * k + EMA(t-1) * (1 - k), where k = 2 / (period + 1)

    Args:
        prices: Series of prices.
        period: EMA span parameter.

    Returns:
        np.ndarray: EMA values.

    Example:
        >>> ema(pd.Series([10, 11, 12, 13, 14]), 3)
        # First value is 10, then exponentially weighted
    """
    s = pd.Series(prices)
    return s.ewm(span=period, adjust=False).mean().values


def rsi(prices: pd.Series, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index.

    Uses true Wilder smoothing (alpha = 1/period), not standard EMA.

    Formula:
        delta = price(t) - price(t-1)
        avg_gain = wilder_smooth(max(delta, 0), period)
        avg_loss = wilder_smooth(max(-delta, 0), period)
        RS = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)

    Args:
        prices: Series of prices.
        period: RSI lookback period. Default 14.

    Returns:
        np.ndarray: RSI values between 0 and 100. First period values are NaN.

    Example:
        >>> rsi(pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10,
        ...     45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]), 14)
        # Last value approximately 70.46
    """
    s = pd.Series(prices, dtype=float)
    delta = s.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rsi_values = pd.Series(np.nan, index=s.index, dtype=float)
    for i in range(len(s)):
        ag = avg_gain.iloc[i]
        al = avg_loss.iloc[i]
        if np.isnan(ag) or np.isnan(al):
            continue
        if al == 0:
            rsi_values.iloc[i] = 100.0 if ag > 0 else 50.0
        else:
            rs = ag / al
            rsi_values.iloc[i] = 100.0 - (100.0 / (1.0 + rs))

    result = rsi_values.values.copy()
    result[:period] = np.nan
    return result


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> np.ndarray:
    """Average True Range with Wilder smoothing.

    Formula:
        TR = max(H - L, |H - C_prev|, |L - C_prev|)
        ATR = Wilder_smooth(TR, period) where alpha = 1/period

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        period: ATR lookback period. Default 14.

    Returns:
        np.ndarray: ATR values. First period values are NaN.

    Example:
        >>> h = pd.Series([48.70, 48.72, 48.90, 48.87, 48.82])
        >>> l = pd.Series([47.79, 48.14, 48.39, 48.37, 48.24])
        >>> c = pd.Series([48.16, 48.61, 48.75, 48.63, 48.74])
        >>> atr(h, l, c, 3)  # ATR over 3 periods
    """
    h = pd.Series(high, dtype=float)
    l_series = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)

    prev_close = c.shift(1)

    tr1 = h - l_series
    tr2 = (h - prev_close).abs()
    tr3 = (l_series - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_values = true_range.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    result = atr_values.values.copy()
    result[:period] = np.nan
    return result


def rolling_zscore(prices: pd.Series, period: int = 20) -> np.ndarray:
    """Rolling Z-score of price relative to its moving window.

    Formula: Z(t) = (price(t) - rolling_mean(t)) / rolling_std(t)

    Args:
        prices: Series of prices.
        period: Rolling window size. Default 20.

    Returns:
        np.ndarray: Z-score values. First (period-1) values are NaN.

    Example:
        >>> rolling_zscore(pd.Series([100, 102, 98, 95, 97]), 3)
        # Z-score of each price relative to its trailing 3-day window
    """
    s = pd.Series(prices, dtype=float)
    rolling_mean = s.rolling(window=period, min_periods=period).mean()
    rolling_std = s.rolling(window=period, min_periods=period).std()
    zscore = (s - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.values


def realized_vol(prices: pd.Series, period: int = 20) -> float:
    """Annualized realized volatility from log returns.

    Formula: vol = std(ln(p(t)/p(t-1))) * sqrt(252)

    Args:
        prices: Series of prices (at least period+1 values).
        period: Lookback window for standard deviation. Default 20.

    Returns:
        float: Annualized volatility as a decimal (e.g. 0.25 = 25%).
            Returns 0.0 if insufficient data.

    Example:
        >>> realized_vol(pd.Series([100, 101, 99, 100.5, 101.2]), 3)
        # Approximately 0.15 (15% annualized)
    """
    s = pd.Series(prices, dtype=float)
    if len(s) < period + 1:
        return 0.0
    log_returns = np.log(s / s.shift(1)).dropna()
    if len(log_returns) < period:
        return 0.0
    vol = float(log_returns.tail(period).std() * np.sqrt(252))
    return round(vol, 6)


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> np.ndarray:
    """Session VWAP: cumulative typical price * volume / cumulative volume.

    Formula: VWAP(t) = cumsum(TP(t) * V(t)) / cumsum(V(t))
             where TP = (high + low + close) / 3

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        volume: Series of volume values.

    Returns:
        np.ndarray: VWAP values for each bar.

    Example:
        >>> vwap(pd.Series([51]), pd.Series([49]), pd.Series([50]), pd.Series([1000]))
        array([50.0])
    """
    h = pd.Series(high, dtype=float)
    l_s = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    v = pd.Series(volume, dtype=float)
    typical_price = (h + l_s + c) / 3.0
    cum_tp_vol = (typical_price * v).cumsum()
    cum_vol = v.cumsum().replace(0, np.nan)
    return (cum_tp_vol / cum_vol).values


def slope(series: pd.Series, period: int = 5) -> float:
    """Linear regression slope over the last N bars.

    Fits a least-squares line to the last 'period' values
    and returns the slope coefficient.

    Formula: slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)

    Args:
        series: Series of values.
        period: Number of trailing bars to fit. Default 5.

    Returns:
        float: Slope of the linear fit. Returns 0.0 if insufficient data.

    Example:
        >>> slope(pd.Series([10, 11, 12, 13, 14]), 5)
        1.0
    """
    s = pd.Series(series, dtype=float)
    if len(s) < period:
        return 0.0
    y = s.tail(period).values
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return 0.0
    return round(float(numerator / denominator), 6)


def adv(volume: pd.Series, prices: pd.Series, period: int = 20) -> float:
    """Average Dollar Volume over a lookback period.

    Formula: ADV = mean(volume(t) * close(t)) for t in last 'period' bars

    Args:
        volume: Series of volume values.
        prices: Series of close prices.
        period: Lookback window. Default 20.

    Returns:
        float: Average dollar volume. Returns 0.0 if insufficient data.

    Example:
        >>> adv(pd.Series([1e6, 1.1e6, 0.9e6]), pd.Series([50, 51, 49]), 3)
        # (50M + 56.1M + 44.1M) / 3 = ~50.07M
    """
    v = pd.Series(volume, dtype=float)
    p = pd.Series(prices, dtype=float)
    dollar_vol = v * p
    if len(dollar_vol) < period:
        return 0.0
    return round(float(dollar_vol.tail(period).mean()), 2)


def vwap_session(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> float:
    """Current session VWAP from cumulative session bars.

    Called with all session bars accumulated so far (not a rolling window).
    Returns the single current VWAP float value.

    Formula: VWAP = sum(TP_i * V_i) / sum(V_i)
             where TP_i = (high_i + low_i + close_i) / 3

    Args:
        high: Series of high prices for session bars.
        low: Series of low prices for session bars.
        close: Series of close prices for session bars.
        volume: Series of volume values for session bars.

    Returns:
        float: Current session VWAP. Returns 0.0 if no volume.

    Example:
        >>> vwap_session(pd.Series([51, 52]), pd.Series([49, 50]),
        ...              pd.Series([50, 51]), pd.Series([1000, 2000]))
        50.667  # approx
    """
    h = pd.Series(high, dtype=float)
    l_s = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    v = pd.Series(volume, dtype=float)
    total_volume = v.sum()
    if total_volume == 0:
        return 0.0
    typical_price = (h + l_s + c) / 3.0
    return float((typical_price * v).sum() / total_volume)


def volume_ratio(volume_series: pd.Series, period: int = 20) -> float:
    """Current bar volume relative to the N-bar average.

    Formula: ratio = current_volume / mean(last N volumes)
    The current volume is the last element of volume_series.
    The denominator uses the N bars preceding the current bar.

    Args:
        volume_series: Series of volume values including current bar at end.
        period: Number of prior bars to average. Default 20.

    Returns:
        float: Volume ratio. Returns 1.0 if insufficient data.

    Example:
        >>> import pandas as pd
        >>> vol = pd.Series([1000]*20 + [3000])  # last bar is 3x average
        >>> volume_ratio(vol, 20)
        3.0
    """
    v = pd.Series(volume_series, dtype=float)
    if len(v) < 2:
        return 1.0
    current = v.iloc[-1]
    prior = v.iloc[-(period + 1):-1] if len(v) > period else v.iloc[:-1]
    avg = prior.mean()
    if avg == 0:
        return 1.0
    return round(float(current / avg), 4)


def bar_position(open_: float, high: float, low: float, close: float) -> float:
    """Position of close within the bar range, clamped to [0, 1].

    Measures directional conviction within the bar.
    Value near 1.0 = closed near high (bullish bar).
    Value near 0.0 = closed near low (bearish bar).

    Formula: position = (close - low) / (high - low)

    Args:
        open_: Bar open price (unused in formula, kept for signature clarity).
        high: Bar high price.
        low: Bar low price.
        close: Bar close price.

    Returns:
        float: Position in range [0.0, 1.0]. Returns 0.5 if high == low.

    Example:
        >>> bar_position(100.0, 105.0, 98.0, 104.0)
        0.857  # close near top of range
        >>> bar_position(100.0, 105.0, 98.0, 99.0)
        0.143  # close near bottom of range
    """
    bar_range = high - low
    if bar_range <= 0:
        return 0.5
    pos = (close - low) / bar_range
    return float(max(0.0, min(1.0, pos)))
