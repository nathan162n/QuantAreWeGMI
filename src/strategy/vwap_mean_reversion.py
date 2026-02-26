"""Layer 1 — VWAP Mean Reversion strategy.

Generates LONG-only BUY signals when price is at least 0.5% below the
session VWAP while RSI(14) confirms oversold conditions. Exits when price
recovers toward VWAP, RSI reaches the exit threshold, the time stop fires,
or a stop-loss is triggered.

Layer name: L1_VWAP_MR

Typical usage in BarDispatcher:
    layer = VWAPMeanReversionStrategy()
    signals = layer.evaluate_bar(symbol, bar, bar_store, open_position)
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.indicators import atr, rsi, volume_ratio
from src.strategy.base_strategy import BaseStrategy, SignalResult
from src.utils.logger import get_logger

logger = get_logger("strategy.vwap_mean_reversion")

# Default parameters used when config.yml is absent (e.g. in tests).
_LAYER1_DEFAULTS = {
    "vwap_deviation_entry": 0.005,
    "rsi_entry_threshold": 42.0,
    "rsi_exit_threshold": 62.0,
    "max_hold_minutes": 180,
    "stop_atr_multiplier": 1.0,
    "vwap_proximity_exit_pct": 0.001,
    "no_entry_after": "15:30",
}


def _load_layer_config(layer_key: str, defaults: dict) -> dict:
    """Load strategy layer config from config.yml, falling back to defaults.

    Reads config.yml → strategies → layer_key and merges with defaults.
    On any error (missing file, missing key) returns the defaults unchanged.

    Args:
        layer_key: Key under strategies section (e.g. 'layer1').
        defaults: Fallback values for each parameter.

    Returns:
        dict: Merged configuration values.
    """
    try:
        import os
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
        )
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        layer_cfg = cfg.get("strategies", {}).get(layer_key, {})
        result = dict(defaults)
        for k, v in layer_cfg.items():
            if k in result and v is not None:
                target_type = type(result[k])
                try:
                    result[k] = target_type(v)
                except (ValueError, TypeError):
                    pass
        return result
    except Exception:
        return dict(defaults)


def _is_past_et_time(time_str: str, bar_ts=None) -> bool:
    """Return True if the reference time is >= time_str (HH:MM) ET.

    Only fires during the regular US equity session (09:15-16:00 ET) to avoid
    false positives when tests or the system run outside of trading hours.

    Args:
        time_str: Threshold as "HH:MM" in Eastern Time.
        bar_ts: Optional UTC-aware datetime from the bar. If provided, the
            bar's local ET time is used instead of the wall clock.
    """
    try:
        import pytz
        et = pytz.timezone("America/New_York")
        h, m = map(int, time_str.split(":"))
        if bar_ts is not None:
            ref_et = bar_ts.astimezone(et)
        else:
            ref_et = datetime.now(et)
        ref_hm = (ref_et.hour, ref_et.minute)
        # Gate is only meaningful during the regular trading session.
        if not ((9, 15) <= ref_hm < (16, 0)):
            return False
        return ref_hm >= (h, m)
    except Exception:
        return False


class VWAPMeanReversionStrategy(BaseStrategy):
    """Layer 1: VWAP Mean Reversion — long-only intraday mean reversion.

    Buys when the current close is materially below the session VWAP and RSI
    confirms oversold conditions. Holds until price reverts back toward VWAP,
    RSI recovers, the time stop expires, or a hard stop-loss is hit.

    Strategy is LONG only. No short signals are generated. Each symbol may
    have at most one open Layer-1 position at a time.

    Attributes:
        vwap_deviation_entry: Minimum negative deviation from VWAP required
            for entry (e.g. 0.005 = price must be 0.5% below VWAP).
        rsi_period: RSI lookback period. Default 14.
        rsi_entry: RSI must be strictly below this level to enter.
        rsi_exit: RSI above this level triggers an exit.
        max_hold_minutes: Maximum minutes to hold before time-stop exit.
        stop_atr_multiplier: ATR multiplier for stop-loss calculation.
        min_bars_required: Minimum 1-min bars needed to compute indicators.

    Example:
        >>> layer = VWAPMeanReversionStrategy()
        >>> signals = layer.evaluate_bar("AAPL", bar, bar_store, None)
        >>> if signals and signals[0].signal == "BUY":
        ...     print(f"Entry at {signals[0].signal_price}, stop={signals[0].stop_price}")
    """

    # Minimum bars required before any indicator can be computed reliably.
    # RSI(14) needs at least 14+1 deltas; ATR(14) needs 14 true ranges.
    # We require 20 bars so both are warm.
    _MIN_BARS: int = 20

    def __init__(self) -> None:
        """Initialize configuration from config.yml with hardcoded fallbacks.

        Reads strategy parameters from the nested strategies.layer1 section
        in config.yml. Falls back to hardcoded defaults when config is absent
        (e.g. in test environments).

        Example:
            >>> layer = VWAPMeanReversionStrategy()
            >>> layer.rsi_period
            14
        """
        cfg = _load_layer_config("layer1", _LAYER1_DEFAULTS)

        self.vwap_deviation_entry: float = float(cfg["vwap_deviation_entry"])
        self.rsi_period: int = 14  # fixed; not per-layer configurable
        self.rsi_entry: float = float(cfg["rsi_entry_threshold"])
        self.rsi_exit: float = float(cfg["rsi_exit_threshold"])
        self.max_hold_minutes: int = int(cfg["max_hold_minutes"])
        self.stop_atr_multiplier: float = float(cfg["stop_atr_multiplier"])
        self.vwap_proximity_exit_pct: float = float(cfg["vwap_proximity_exit_pct"])
        self._no_entry_after: str = str(cfg.get("no_entry_after", "15:30"))

        # Internal per-symbol position state.
        # Each entry: {entry_price, entry_time, stop_price, bars_held}
        self._open_positions: Dict[str, dict] = {}

        logger.info(
            "%s initialized — vwap_dev=%.4f rsi_period=%d "
            "rsi_entry=%.1f rsi_exit=%.1f max_hold=%dmin stop_atr=%.2f no_entry_after=%s",
            self.layer_name,
            self.vwap_deviation_entry,
            self.rsi_period,
            self.rsi_entry,
            self.rsi_exit,
            self.max_hold_minutes,
            self.stop_atr_multiplier,
            self._no_entry_after,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def layer_name(self) -> str:
        """Unique identifier for this strategy layer.

        Returns:
            str: Always "L1_VWAP_MR".

        Example:
            >>> VWAPMeanReversionStrategy().layer_name
            'L1_VWAP_MR'
        """
        return "L1_VWAP_MR"

    def evaluate_bar(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        open_position: Optional[dict] = None,
    ) -> List[SignalResult]:
        """Evaluate one 1-minute bar and return any BUY or EXIT signals.

        Execution order per bar:
          1. Increment bars_held for any open internal position.
          2. Fetch indicator data from bar_store.
          3. Check all exit conditions — return EXIT if any are met.
          4. Check all entry conditions — return BUY if all are met.
          5. Return empty list (HOLD) otherwise.

        The ``open_position`` argument from the dispatcher is used only to
        detect whether a position already exists externally. Internal state
        is maintained in ``_open_positions``.

        Args:
            symbol: Ticker symbol being evaluated.
            bar: Current bar dict with open, high, low, close, volume, timestamp.
            bar_store: BarStore instance providing historical bars and session VWAP.
            open_position: Optional position dict passed by the dispatcher.
                If not None and ``layer_name`` matches, this layer treats
                the symbol as already having an open position.

        Returns:
            List[SignalResult]: Empty list, a single EXIT signal, or a single
                BUY signal. Never returns more than one element.

        Example:
            >>> layer = VWAPMeanReversionStrategy()
            >>> signals = layer.evaluate_bar("AAPL", bar, bar_store, None)
            >>> # Returns [] (HOLD), or a list with one BUY or EXIT signal.
        """
        current_close = float(bar.get("close", 0.0))
        if current_close <= 0.0:
            logger.debug("%s: skipping bar with invalid close=%.4f", symbol, current_close)
            return []

        # Sync external position state into internal tracking if needed.
        if (
            open_position is not None
            and open_position.get("layer_name") == self.layer_name
            and symbol not in self._open_positions
        ):
            self._open_positions[symbol] = {
                "entry_price": float(open_position.get("entry_price", current_close)),
                "entry_time": open_position.get("entry_time", datetime.now(timezone.utc)),
                "stop_price": float(open_position.get("stop_price", 0.0)),
                "bars_held": int(open_position.get("bars_held", 0)),
            }

        # Step 1 — increment bars_held for any live internal position.
        if symbol in self._open_positions:
            self._open_positions[symbol]["bars_held"] += 1

        # Step 2 — fetch bar history and compute indicators.
        df = bar_store.get_bars(symbol, "1Min", self._MIN_BARS + self.rsi_period)
        if len(df) < self._MIN_BARS:
            logger.debug(
                "%s: insufficient bars (%d < %d)", symbol, len(df), self._MIN_BARS
            )
            return []

        session_vwap = bar_store.get_session_vwap(symbol)
        bar_count = bar_store.get_bar_count(symbol)

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]

        rsi_arr = rsi(closes, self.rsi_period)
        atr_arr = atr(highs, lows, closes, self.rsi_period)

        current_rsi = rsi_arr[-1]
        current_atr = atr_arr[-1]

        if np.isnan(current_rsi) or np.isnan(current_atr):
            logger.debug("%s: indicator NaN — rsi=%.2f atr=%.4f", symbol, current_rsi, current_atr)
            return []

        distance_from_vwap = (
            (current_close - session_vwap) / session_vwap
            if session_vwap > 0.0
            else 0.0
        )

        # Step 3 — check exit conditions for open position.
        if symbol in self._open_positions:
            exit_signal = self._check_exit(
                symbol=symbol,
                current_close=current_close,
                distance_from_vwap=distance_from_vwap,
                current_rsi=current_rsi,
                session_vwap=session_vwap,
            )
            if exit_signal is not None:
                return [exit_signal]

        # Step 4 — check entry conditions (no open position allowed).
        if symbol not in self._open_positions:
            # Phase 4B time gate: no new entries after configured cut-off.
            # Uses bar timestamp so test bars with historical timestamps are
            # correctly evaluated rather than comparing against wall-clock time.
            bar_ts = bar.get("timestamp")
            if self._no_entry_after and _is_past_et_time(self._no_entry_after, bar_ts):
                logger.debug(
                    "%s [%s] entry blocked — past no_entry_after=%s",
                    symbol, self.layer_name, self._no_entry_after,
                )
                return []

            entry_signal = self._check_entry(
                symbol=symbol,
                current_close=current_close,
                distance_from_vwap=distance_from_vwap,
                current_rsi=current_rsi,
                current_atr=current_atr,
                session_vwap=session_vwap,
                bar_count=bar_count,
            )
            if entry_signal is not None:
                return [entry_signal]

        return []

    def should_exit(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        position_data: dict,
    ) -> bool:
        """Determine whether an open position should be closed immediately.

        Called by StreamEngine on stop-loss tick checks and by EODManager at
        15:30 ET. Evaluates the same exit conditions as evaluate_bar() but
        without modifying any internal state (no bars_held increment).

        Args:
            symbol: Ticker symbol of the open position.
            bar: Current bar dict with open, high, low, close, volume, timestamp.
            bar_store: BarStore instance.
            position_data: Dict with at minimum entry_price, entry_time,
                stop_price, bars_held, layer_name.

        Returns:
            bool: True if the position should be closed, False otherwise.

        Example:
            >>> layer = VWAPMeanReversionStrategy()
            >>> should_close = layer.should_exit(
            ...     "AAPL", current_bar, bar_store,
            ...     {"entry_price": 150.0, "stop_price": 149.5,
            ...      "bars_held": 20, "entry_time": datetime(...)})
            True
        """
        current_close = float(bar.get("close", 0.0))
        if current_close <= 0.0:
            return False

        session_vwap = bar_store.get_session_vwap(symbol)
        bars_held = int(position_data.get("bars_held", 0))
        stop_price = float(position_data.get("stop_price", 0.0))
        entry_time = position_data.get("entry_time")

        distance_from_vwap = (
            (current_close - session_vwap) / session_vwap
            if session_vwap > 0.0
            else 0.0
        )

        # Check time stop using wall-clock minutes if entry_time is available.
        minutes_held = bars_held
        if entry_time is not None:
            try:
                now = datetime.now(timezone.utc)
                if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is not None:
                    elapsed = (now - entry_time).total_seconds() / 60.0
                else:
                    elapsed = bars_held
                minutes_held = max(int(elapsed), bars_held)
            except Exception:
                minutes_held = bars_held

        # Stop loss
        if stop_price > 0.0 and current_close < stop_price:
            logger.info(
                "%s [%s] EXIT — stop-loss hit close=%.4f stop=%.4f",
                symbol, self.layer_name, current_close, stop_price,
            )
            return True

        # Time stop
        if minutes_held >= self.max_hold_minutes:
            logger.info(
                "%s [%s] EXIT — time stop minutes_held=%d >= %d",
                symbol, self.layer_name, minutes_held, self.max_hold_minutes,
            )
            return True

        # VWAP proximity exit
        if distance_from_vwap > -self.vwap_proximity_exit_pct:
            logger.info(
                "%s [%s] EXIT — near VWAP dist=%.4f (threshold=%.4f)",
                symbol, self.layer_name, distance_from_vwap, self.vwap_proximity_exit_pct,
            )
            return True

        # RSI exit — need indicator data for this check
        df = bar_store.get_bars(symbol, "1Min", self._MIN_BARS + self.rsi_period)
        if len(df) >= self._MIN_BARS:
            rsi_arr = rsi(df["close"], self.rsi_period)
            current_rsi = rsi_arr[-1]
            if not np.isnan(current_rsi) and current_rsi > self.rsi_exit:
                logger.info(
                    "%s [%s] EXIT — RSI=%.2f > exit=%.1f",
                    symbol, self.layer_name, current_rsi, self.rsi_exit,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_exit(
        self,
        symbol: str,
        current_close: float,
        distance_from_vwap: float,
        current_rsi: float,
        session_vwap: float,
    ) -> Optional[SignalResult]:
        """Evaluate all exit conditions for an open position.

        Exit conditions (any one triggers exit):
          1. Price within 0.1% of VWAP (distance_from_vwap > -0.001).
          2. RSI above rsi_exit threshold.
          3. Time stop — bars_held >= max_hold_minutes.
          4. Stop-loss — current_close < stop_price.

        Args:
            symbol: Ticker symbol.
            current_close: Current bar close price.
            distance_from_vwap: Signed (close - vwap) / vwap.
            current_rsi: Current RSI value.
            session_vwap: Current session VWAP.

        Returns:
            Optional[SignalResult]: EXIT signal if any condition is met, else None.

        Example:
            >>> signal = layer._check_exit("AAPL", 151.0, -0.0003, 65.0, 151.5)
            >>> signal.signal
            'EXIT'
        """
        pos = self._open_positions[symbol]
        bars_held = pos["bars_held"]
        stop_price = pos["stop_price"]
        entry_price = pos["entry_price"]

        exit_reason: Optional[str] = None

        if stop_price > 0.0 and current_close < stop_price:
            exit_reason = f"stop_loss close={current_close:.4f} < stop={stop_price:.4f}"
        elif bars_held >= self.max_hold_minutes:
            exit_reason = f"time_stop bars_held={bars_held} >= {self.max_hold_minutes}"
        elif distance_from_vwap > -self.vwap_proximity_exit_pct:
            exit_reason = f"vwap_proximity dist={distance_from_vwap:.4f} threshold={self.vwap_proximity_exit_pct:.4f}"
        elif current_rsi > self.rsi_exit:
            exit_reason = f"rsi_exit rsi={current_rsi:.2f} > {self.rsi_exit}"

        if exit_reason is None:
            return None

        pnl_pct = (current_close - entry_price) / entry_price if entry_price > 0.0 else 0.0
        logger.info(
            "%s [%s] EXIT — reason=%s pnl=%.2f%% entry=%.4f close=%.4f bars=%d",
            symbol,
            self.layer_name,
            exit_reason,
            pnl_pct * 100.0,
            entry_price,
            current_close,
            bars_held,
        )

        del self._open_positions[symbol]

        return SignalResult(
            symbol=symbol,
            signal="EXIT",
            confidence=1.0,
            signal_price=current_close,
            layer_name=self.layer_name,
            stop_price=stop_price,
            metadata={
                "exit_reason": exit_reason,
                "entry_price": entry_price,
                "bars_held": bars_held,
                "pnl_pct": round(pnl_pct, 6),
                "vwap": session_vwap,
                "rsi": round(current_rsi, 2),
            },
        )

    def _check_entry(
        self,
        symbol: str,
        current_close: float,
        distance_from_vwap: float,
        current_rsi: float,
        current_atr: float,
        session_vwap: float,
        bar_count: int,
    ) -> Optional[SignalResult]:
        """Evaluate all entry conditions for a new long position.

        Entry requires ALL of:
          1. Session VWAP is valid (> 0).
          2. At least 5 bars into the session (bar_count > 5).
          3. Price is at least vwap_deviation_entry below VWAP.
          4. RSI(14) is below rsi_entry threshold.

        Args:
            symbol: Ticker symbol.
            current_close: Current bar close price.
            distance_from_vwap: Signed (close - vwap) / vwap.
            current_rsi: Current RSI value.
            current_atr: Current ATR(14) value.
            session_vwap: Current session VWAP.
            bar_count: Total 1-min bars received for this symbol today.

        Returns:
            Optional[SignalResult]: BUY signal if all conditions pass, else None.

        Example:
            >>> signal = layer._check_entry("AAPL", 149.25, -0.0083, 38.5,
            ...                              0.45, 150.5, 30)
            >>> signal.signal if signal else None
            'BUY'
        """
        # Condition 4: session must have started
        if session_vwap <= 0.0:
            return None

        # Condition 3: not in first 5 bars
        if bar_count <= 5:
            return None

        # Condition 1: price at least vwap_deviation_entry below VWAP
        if distance_from_vwap >= -self.vwap_deviation_entry:
            return None

        # Condition 2: RSI confirms oversold
        if current_rsi >= self.rsi_entry:
            return None

        # All conditions met — compute stop price and confidence.
        stop_price = current_close - self.stop_atr_multiplier * current_atr
        confidence = min(abs(distance_from_vwap) / 0.02, 1.0)

        logger.info(
            "%s [%s] BUY — close=%.4f vwap=%.4f dist=%.4f rsi=%.2f "
            "atr=%.4f stop=%.4f conf=%.3f bars=%d",
            symbol,
            self.layer_name,
            current_close,
            session_vwap,
            distance_from_vwap,
            current_rsi,
            current_atr,
            stop_price,
            confidence,
            bar_count,
        )

        self._open_positions[symbol] = {
            "entry_price": current_close,
            "entry_time": datetime.now(timezone.utc),
            "stop_price": stop_price,
            "bars_held": 0,
        }

        return SignalResult(
            symbol=symbol,
            signal="BUY",
            confidence=confidence,
            signal_price=current_close,
            layer_name=self.layer_name,
            stop_price=stop_price,
            metadata={
                "vwap": session_vwap,
                "distance_from_vwap": round(distance_from_vwap, 6),
                "rsi": round(current_rsi, 2),
                "atr": round(current_atr, 4),
                "bar_count": bar_count,
            },
        )

    def clear_position(self, symbol: str) -> None:
        """Remove internal position state for a symbol without generating a signal.

        Called by the order manager or EOD handler after a fill is confirmed,
        ensuring internal state stays in sync with actual broker positions.

        Args:
            symbol: Ticker symbol whose position should be cleared.

        Example:
            >>> layer.clear_position("AAPL")
        """
        if symbol in self._open_positions:
            pos = self._open_positions.pop(symbol)
            logger.debug(
                "%s [%s] internal position cleared (entry=%.4f bars_held=%d)",
                symbol,
                self.layer_name,
                pos.get("entry_price", 0.0),
                pos.get("bars_held", 0),
            )

    def has_open_position(self, symbol: str) -> bool:
        """Check whether this layer has an internally tracked open position.

        Args:
            symbol: Ticker symbol to check.

        Returns:
            bool: True if there is an open position for this symbol.

        Example:
            >>> layer.has_open_position("AAPL")
            False
        """
        return symbol in self._open_positions

    def get_position_info(self, symbol: str) -> Optional[dict]:
        """Return the internal position state for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Optional[dict]: Position dict with entry_price, entry_time,
                stop_price, bars_held — or None if no open position exists.

        Example:
            >>> info = layer.get_position_info("AAPL")
            >>> info["bars_held"] if info else None
            15
        """
        return self._open_positions.get(symbol)
