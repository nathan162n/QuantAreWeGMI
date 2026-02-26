"""In-memory rolling bar buffer for all symbols and timeframes.

Central data store used by all strategy layers. Bars arrive from the
WebSocket stream and are stored here. Strategy layers read from this
store — they never call the Alpaca API directly during signal evaluation.
"""

from collections import deque
from datetime import datetime, timezone, time as dt_time
from typing import Dict, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data.bar_store")

_BAR_STORE_INSTANCE: Optional["BarStore"] = None

# ET = UTC-5 standard / UTC-4 daylight. We use a fixed offset approach here;
# for precise DST handling the caller should pass ET-aware datetimes.
_SESSION_OPEN_HOUR_ET = 9
_SESSION_OPEN_MIN_ET = 30
_SESSION_CLOSE_HOUR_ET = 16

# Constant for market-open time of day (ET) used to detect session resets
_MARKET_OPEN_TIME = dt_time(9, 30)


class BarStore:
    """Central in-memory store for OHLCV bars, organized by symbol and timeframe.

    Storage structure:
        _store: Dict[str, Dict[str, deque]]
          key1: symbol (e.g., "AAPL")
          key2: timeframe (e.g., "1Min", "5Min")
          value: deque(maxlen=200) of dict bars (most recent at right)

    Additionally maintains:
        _session_vwap: Dict[str, float] — running session VWAP per symbol
        _session_cumvol: Dict[str, float] — cumulative session volume
        _session_cum_tp_vol: Dict[str, float] — sum(TP * V) for VWAP
        _opening_range: Dict[str, dict] — {high, low, computed_at} per symbol
        _bar_counts: Dict[str, int] — total bars received per symbol (for cadence)

    VWAP is session-cumulative (resets at 09:30 ET each day).
    Opening range is computed once at 10:00 ET and stored until reset.

    Args:
        max_bars: Maximum bars per symbol per timeframe. Default 200.

    Example:
        >>> store = BarStore()
        >>> store.update("AAPL", "1Min", {"open": 150, "high": 151,
        ...     "low": 149, "close": 150.5, "volume": 10000,
        ...     "timestamp": datetime.now(timezone.utc)})
        >>> df = store.get_bars("AAPL", "1Min", 10)
    """

    def __init__(self, max_bars: int = 200):
        """Initialize empty bar store.

        Args:
            max_bars: Maximum bars per symbol/timeframe deque. Default 200.

        Example:
            >>> store = BarStore(max_bars=200)
        """
        self._max_bars = max_bars
        # _store[symbol][timeframe] = deque of bar dicts
        self._store: Dict[str, Dict[str, deque]] = {}
        # Session VWAP accumulators
        self._session_cum_tp_vol: Dict[str, float] = {}
        self._session_cumvol: Dict[str, float] = {}
        # Opening range per symbol: {high, low, computed_at}
        self._opening_range: Dict[str, Optional[dict]] = {}
        # Opening range accumulators (bars between 09:30 and 10:00)
        self._opening_range_bars: Dict[str, list] = {}
        # Bar counter per symbol (used for Layer 3 cadence)
        self._bar_counts: Dict[str, int] = {}

    def _ensure_symbol(self, symbol: str, timeframe: str) -> None:
        """Ensure storage structures exist for symbol/timeframe.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe string.

        Example:
            >>> store._ensure_symbol("AAPL", "1Min")
        """
        if symbol not in self._store:
            self._store[symbol] = {}
            self._session_cum_tp_vol[symbol] = 0.0
            self._session_cumvol[symbol] = 0.0
            self._opening_range[symbol] = None
            self._opening_range_bars[symbol] = []
            self._bar_counts[symbol] = 0
        if timeframe not in self._store[symbol]:
            self._store[symbol][timeframe] = deque(maxlen=self._max_bars)

    def update(self, symbol: str, timeframe: str, bar: dict) -> None:
        """Add a bar to the store and update session accumulators.

        Updates VWAP accumulators and opening range if applicable.
        The bar dict must have keys: open, high, low, close, volume, timestamp.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe (e.g., "1Min", "5Min").
            bar: Dict with keys open, high, low, close, volume, timestamp.

        Example:
            >>> store.update("AAPL", "1Min", {
            ...     "open": 150.0, "high": 151.0, "low": 149.5,
            ...     "close": 150.8, "volume": 5000,
            ...     "timestamp": datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)})
        """
        self._ensure_symbol(symbol, timeframe)
        self._store[symbol][timeframe].append(bar)
        self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1

        # Update VWAP accumulators (only for 1-minute bars to avoid double counting)
        if timeframe == "1Min":
            h = float(bar.get("high", 0))
            l_val = float(bar.get("low", 0))
            c = float(bar.get("close", 0))
            v = float(bar.get("volume", 0))
            typical_price = (h + l_val + c) / 3.0
            self._session_cum_tp_vol[symbol] = (
                self._session_cum_tp_vol.get(symbol, 0.0) + typical_price * v
            )
            self._session_cumvol[symbol] = (
                self._session_cumvol.get(symbol, 0.0) + v
            )

            # Accumulate opening range bars (09:30 – 10:00 ET)
            ts = bar.get("timestamp")
            if ts is not None:
                self._maybe_accumulate_opening_range(symbol, bar, ts)

    def _maybe_accumulate_opening_range(
        self, symbol: str, bar: dict, ts: datetime
    ) -> None:
        """Accumulate a bar into the opening range if within 09:30–10:00 window.

        Opening range is not finalized here; it is finalized by calling
        finalize_opening_range() at exactly 10:00 ET.

        Args:
            symbol: Ticker symbol.
            bar: Bar dict.
            ts: Bar timestamp (timezone-aware preferred).

        Example:
            >>> # Called internally from update()
        """
        try:
            # Convert to ET by subtracting UTC offset (approximate: UTC-5 or UTC-4)
            # We check both 9:30 ET (14:30 UTC in EST, 13:30 UTC in EDT)
            # Use a simple heuristic: if ts has tzinfo use it, else assume UTC
            if ts.tzinfo is not None:
                # Convert to ET by checking hour in UTC range 14-15 (EST) or 13-14 (EDT)
                h_utc = ts.hour
                m_utc = ts.minute
                # ET is UTC-5 (EST) or UTC-4 (EDT). Check 09:30–10:00 ET
                # 09:30 ET = 14:30 UTC (EST) or 13:30 UTC (EDT)
                # 10:00 ET = 15:00 UTC (EST) or 14:00 UTC (EDT)
                # To handle both, check if bar is in 09:30-10:00 window using
                # a minute-of-day approach. We store all bars in the 09:30-10:00
                # range and let finalize_opening_range figure it out.
                # Simple approach: track by bar hour/minute in ET.
                # Since we can't reliably detect DST here, we check UTC offsets
                # and accept bars in the union of both windows.
                # 09:30-10:00 ET in UTC = 13:30-15:00 (covers both DST/non-DST)
                minutes_utc = h_utc * 60 + m_utc
                # EST range: 14:30–15:00 UTC = 870–900 minutes
                # EDT range: 13:30–14:00 UTC = 810–840 minutes
                in_window = (870 <= minutes_utc <= 900) or (810 <= minutes_utc <= 840)
                if in_window:
                    self._opening_range_bars[symbol].append(bar)
        except Exception:
            pass

    def finalize_opening_range(self, symbol: str) -> Optional[dict]:
        """Compute and store the opening range from accumulated 09:30-10:00 bars.

        Should be called at exactly 10:00 ET. After this call,
        get_opening_range() will return the computed range.

        Args:
            symbol: Ticker symbol.

        Returns:
            Optional[dict]: {high, low, computed_at} or None if no bars.

        Example:
            >>> store.finalize_opening_range("AAPL")
            {'high': 152.5, 'low': 150.0, 'computed_at': datetime(...)}
        """
        bars = self._opening_range_bars.get(symbol, [])
        if not bars:
            logger.warning(
                "No opening range bars for %s — cannot compute opening range", symbol
            )
            return None
        high = max(float(b["high"]) for b in bars)
        low = min(float(b["low"]) for b in bars)
        result = {
            "high": high,
            "low": low,
            "computed_at": datetime.now(timezone.utc),
        }
        self._opening_range[symbol] = result
        logger.info(
            "Opening range finalized for %s: high=%.4f, low=%.4f (%d bars)",
            symbol,
            high,
            low,
            len(bars),
        )
        return result

    def get_bars(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        """Return last n bars as DataFrame.

        Columns: open, high, low, close, volume, timestamp.
        Most recent bar is at index -1 (last row).

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe (e.g., "1Min", "5Min").
            n: Number of most recent bars to return.

        Returns:
            pd.DataFrame: DataFrame with OHLCV columns. Empty DataFrame if
                insufficient data.

        Example:
            >>> df = store.get_bars("AAPL", "1Min", 20)
            >>> df.iloc[-1]["close"]  # most recent close
            150.8
        """
        self._ensure_symbol(symbol, timeframe)
        dq = self._store[symbol][timeframe]
        bars_list = list(dq)
        if not bars_list:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "timestamp"]
            )
        if n < len(bars_list):
            bars_list = bars_list[-n:]
        df = pd.DataFrame(bars_list)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_session_vwap(self, symbol: str) -> float:
        """Return the current session VWAP for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            float: Session VWAP. Returns 0.0 if no session data.

        Example:
            >>> store.get_session_vwap("AAPL")
            150.32
        """
        vol = self._session_cumvol.get(symbol, 0.0)
        if vol == 0:
            return 0.0
        return round(self._session_cum_tp_vol.get(symbol, 0.0) / vol, 6)

    def get_opening_range(self, symbol: str) -> Optional[dict]:
        """Return the stored opening range for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Optional[dict]: {high, low, computed_at} if computed, else None.

        Example:
            >>> or_ = store.get_opening_range("AAPL")
            >>> if or_: print(or_["high"])
            152.5
        """
        return self._opening_range.get(symbol)

    def reset_session(self, symbol: str) -> None:
        """Reset all session accumulators for a symbol at 09:30 ET.

        Clears VWAP accumulators and opening range data. Called at
        market open each day before the stream starts.

        Args:
            symbol: Ticker symbol.

        Example:
            >>> store.reset_session("AAPL")
        """
        self._session_cum_tp_vol[symbol] = 0.0
        self._session_cumvol[symbol] = 0.0
        self._opening_range[symbol] = None
        self._opening_range_bars[symbol] = []
        logger.debug("Session reset for %s", symbol)

    def reset_all_sessions(self) -> None:
        """Reset session data for all symbols.

        Called at 09:30 ET each day before market open.

        Example:
            >>> store.reset_all_sessions()
        """
        for symbol in list(self._store.keys()):
            self.reset_session(symbol)
        logger.info("All session data reset for %d symbols", len(self._store))

    def get_latest_close(self, symbol: str) -> float:
        """Return the most recent close price for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            float: Most recent close price. Returns 0.0 if no bars.

        Example:
            >>> store.get_latest_close("AAPL")
            150.8
        """
        if symbol not in self._store:
            return 0.0
        for tf in ["1Min", "5Min", "1Day"]:
            if tf in self._store[symbol] and self._store[symbol][tf]:
                dq = self._store[symbol][tf]
                last_bar = dq[-1]
                return float(last_bar.get("close", 0.0))
        return 0.0

    def get_bar_count(self, symbol: str) -> int:
        """Return total 1-minute bars received for a symbol in this session.

        Used by BarDispatcher for Layer 3 cadence (every 3rd bar).

        Args:
            symbol: Ticker symbol.

        Returns:
            int: Count of bars received.

        Example:
            >>> store.get_bar_count("AAPL")
            47
        """
        return self._bar_counts.get(symbol, 0)

    def get_synthetic_5min_bar(self, symbol: str) -> Optional[dict]:
        """Construct a synthetic 5-minute bar from the last 5 1-minute bars.

        Used by Layer 3 (RSI Scalp) which evaluates on 5-minute resolution.
        Synthetic bar: open=first 1-min open, high=max high, low=min low,
        close=last close, volume=sum of volumes.

        Args:
            symbol: Ticker symbol.

        Returns:
            Optional[dict]: Synthetic bar dict or None if < 5 bars available.

        Example:
            >>> bar = store.get_synthetic_5min_bar("AAPL")
            >>> bar["close"]  # last 1-min close
            150.8
        """
        if symbol not in self._store or "1Min" not in self._store[symbol]:
            return None
        dq = self._store[symbol]["1Min"]
        if len(dq) < 5:
            return None
        last5 = list(dq)[-5:]
        return {
            "open": float(last5[0].get("open", 0)),
            "high": max(float(b.get("high", 0)) for b in last5),
            "low": min(float(b.get("low", 0)) for b in last5),
            "close": float(last5[-1].get("close", 0)),
            "volume": sum(float(b.get("volume", 0)) for b in last5),
            "timestamp": last5[-1].get("timestamp"),
        }

    def preload_bars(
        self, symbol: str, timeframe: str, bars: list
    ) -> None:
        """Bulk-load historical bars into the store (used at startup/backtest).

        Each element of bars must be a dict with open, high, low, close,
        volume, timestamp keys.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            bars: List of bar dicts in chronological order (oldest first).

        Example:
            >>> store.preload_bars("AAPL", "1Min", historical_bar_list)
        """
        self._ensure_symbol(symbol, timeframe)
        for bar in bars:
            self._store[symbol][timeframe].append(bar)
        logger.debug(
            "Preloaded %d %s bars for %s", len(bars), timeframe, symbol
        )


def get_bar_store() -> BarStore:
    """Get or create the singleton BarStore instance.

    Returns:
        BarStore: The global bar store.

    Example:
        >>> store = get_bar_store()
    """
    global _BAR_STORE_INSTANCE
    if _BAR_STORE_INSTANCE is None:
        _BAR_STORE_INSTANCE = BarStore()
    return _BAR_STORE_INSTANCE


def reset_bar_store() -> None:
    """Reset the singleton (used in tests).

    Example:
        >>> reset_bar_store()
    """
    global _BAR_STORE_INSTANCE
    _BAR_STORE_INSTANCE = None
