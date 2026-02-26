"""Layer 2 — Opening Range Breakout strategy.

Generates LONG-only BUY signals when price breaks convincingly above the
09:30–10:00 ET opening range with above-average volume confirmation and a
bullish bar structure. Enters only once per symbol per session.

Opening range is provided by bar_store.get_opening_range(symbol) and is
finalized at exactly 10:00 ET by the BarStore. This layer will not fire
until that range is available.

Layer name: L2_ORB

Typical usage in BarDispatcher:
    layer = OpeningRangeBreakoutStrategy()
    signals = layer.evaluate_bar(symbol, bar, bar_store, open_position)

Daily state must be reset at the start of each session:
    layer.reset_daily_state()
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.data.indicators import atr, bar_position, volume_ratio
from src.strategy.base_strategy import BaseStrategy, SignalResult
from src.utils.logger import get_logger

logger = get_logger("strategy.opening_range_breakout")

# Default parameters used when config.yml is absent (e.g. in tests).
_LAYER2_DEFAULTS = {
    "breakout_buffer_pct": 0.001,
    "volume_confirmation_ratio": 1.5,
    "bar_position_threshold": 0.70,
    "extension_target_multiplier": 2.0,
    "max_range_atr_multiplier": 3.0,
    "max_hold_minutes": 330,
    "breakout_failure_pct": 0.002,
    "stop_pct_of_range": 0.5,
}


def _load_layer_config(layer_key: str, defaults: dict) -> dict:
    """Load strategy layer config from config.yml, falling back to defaults."""
    try:
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


def _env_float(key: str, default: float) -> float:
    """Read a float from environment variables with a fallback default.

    Args:
        key: Environment variable name.
        default: Value to use when the variable is absent or unparseable.

    Returns:
        float: Parsed value or the default.

    Example:
        >>> _env_float("L2_BREAKOUT_BUFFER_PCT", 0.001)
        0.001
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid env value for %s='%s', using default %.4f", key, raw, default)
        return default


def _env_int(key: str, default: int) -> int:
    """Read an integer from environment variables with a fallback default.

    Args:
        key: Environment variable name.
        default: Value to use when the variable is absent or unparseable.

    Returns:
        int: Parsed value or the default.

    Example:
        >>> _env_int("L2_MAX_HOLD_MINUTES", 330)
        330
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid env value for %s='%s', using default %d", key, raw, default)
        return default


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Layer 2: Opening Range Breakout — long-only momentum after opening range.

    Waits for the 09:30–10:00 ET opening range to be finalized by the
    BarStore, then monitors for a sustained break above the range high with
    confirming volume and bullish bar structure. Each symbol can trigger at
    most one ORB entry per trading session.

    Entry gating:
      - Opening range must be finalized (bar_store.get_opening_range not None).
      - Range size must not exceed 3 * daily ATR proxy (avoids gapped/wide days).
      - Symbol must not already have an ORB entry for today.
      - Price must close > range_high * (1 + breakout_buffer_pct).
      - Volume ratio of current bar vs. prior 20 bars must exceed 1.5.
      - Bar position (close in top 30% of bar range) > 0.70.

    Exits:
      - Price falls back below range_high * (1 - 0.002): breakout failure.
      - Price reaches entry_price + 2 * opening_range_size: profit target.
      - Price falls below stop_price.
      - bars_held >= max_hold_minutes: hard time-stop (15:30 ET equivalent).

    Stop price = opening_range_high - opening_range_size * 0.5

    Attributes:
        breakout_buffer_pct: Fractional buffer above range high required for
            entry. Default 0.001 (0.1%).
        volume_confirmation_ratio: Minimum volume ratio required to confirm
            breakout. Default 1.5.
        bar_position_threshold: Minimum (close-low)/(high-low) ratio. Default 0.70.
        extension_target_multiplier: Profit target expressed as a multiple of
            the opening range size above entry. Default 2.0.
        max_range_atr_multiplier: Skip entry if range > this multiple of the
            daily ATR proxy. Default 3.0.
        max_hold_minutes: Maximum bars/minutes to hold (maps to 15:30 ET hard
            close if session starts at 09:30 ET). Default 330.

    Example:
        >>> layer = OpeningRangeBreakoutStrategy()
        >>> layer.reset_daily_state()          # call at 09:30 ET each day
        >>> signals = layer.evaluate_bar("MSFT", bar, bar_store, None)
        >>> if signals and signals[0].signal == "BUY":
        ...     print(f"ORB entry at {signals[0].signal_price}")
    """

    # Number of 1-min bars needed to compute ATR(14) reliably.
    _MIN_BARS: int = 21  # 20 for volume ratio denominator + 1 current bar

    def __init__(self) -> None:
        """Initialize configuration from config.yml with environment variable overrides.

        Reads strategy parameters from strategies.layer2 in config.yml. Environment
        variables (e.g. L2_BREAKOUT_BUFFER_PCT) override config values when set,
        preserving backward compatibility.

        Example:
            >>> layer = OpeningRangeBreakoutStrategy()
            >>> layer.breakout_buffer_pct
            0.0005
        """
        cfg = _load_layer_config("layer2", _LAYER2_DEFAULTS)

        # Config.yml is the primary source; env vars override when explicitly set.
        self.breakout_buffer_pct: float = _env_float(
            "L2_BREAKOUT_BUFFER_PCT", float(cfg["breakout_buffer_pct"])
        )
        self.volume_confirmation_ratio: float = _env_float(
            "L2_VOLUME_CONFIRMATION_RATIO", float(cfg["volume_confirmation_ratio"])
        )
        self.bar_position_threshold: float = _env_float(
            "L2_BAR_POSITION_THRESHOLD", float(cfg["bar_position_threshold"])
        )
        self.extension_target_multiplier: float = _env_float(
            "L2_EXTENSION_TARGET_MULTIPLIER", float(cfg["extension_target_multiplier"])
        )
        self.max_range_atr_multiplier: float = _env_float(
            "L2_MAX_RANGE_ATR_MULTIPLIER", float(cfg["max_range_atr_multiplier"])
        )
        self.max_hold_minutes: int = _env_int(
            "L2_MAX_HOLD_MINUTES", int(cfg["max_hold_minutes"])
        )
        self.breakout_failure_pct: float = _env_float(
            "L2_BREAKOUT_FAILURE_PCT", float(cfg["breakout_failure_pct"])
        )
        self.stop_pct_of_range: float = _env_float(
            "L2_STOP_PCT_OF_RANGE", float(cfg["stop_pct_of_range"])
        )

        # Symbols that already had an ORB entry today — reset each morning.
        self._triggered_today: Set[str] = set()

        # Stored opening range size at the time of entry, keyed by symbol.
        self._opening_range_size: Dict[str, float] = {}

        # Per-symbol open position state.
        # Each entry: {entry_price, entry_time, stop_price, bars_held,
        #              opening_range_high, opening_range_size}
        self._open_positions: Dict[str, dict] = {}

        logger.info(
            "%s initialized — buffer=%.4f vol_ratio=%.2f bar_pos=%.2f "
            "ext_mult=%.2f max_range_atr=%.2f max_hold=%dmin "
            "breakout_fail=%.4f stop_pct_range=%.2f",
            self.layer_name,
            self.breakout_buffer_pct,
            self.volume_confirmation_ratio,
            self.bar_position_threshold,
            self.extension_target_multiplier,
            self.max_range_atr_multiplier,
            self.max_hold_minutes,
            self.breakout_failure_pct,
            self.stop_pct_of_range,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def layer_name(self) -> str:
        """Unique identifier for this strategy layer.

        Returns:
            str: Always "L2_ORB".

        Example:
            >>> OpeningRangeBreakoutStrategy().layer_name
            'L2_ORB'
        """
        return "L2_ORB"

    def evaluate_bar(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        open_position: Optional[dict] = None,
    ) -> List[SignalResult]:
        """Evaluate one 1-minute bar and return any BUY or EXIT signals.

        Execution order per bar:
          1. Validate bar data.
          2. Sync external position state into internal tracking if needed.
          3. Increment bars_held for any open internal position.
          4. Verify opening range is available — skip if not.
          5. Compute indicators (ATR, volume ratio, bar position).
          6. Check all exit conditions if a position is open.
          7. Check all entry conditions if no position is open.
          8. Return empty list (HOLD) otherwise.

        Args:
            symbol: Ticker symbol being evaluated.
            bar: Current bar dict with open, high, low, close, volume, timestamp.
            bar_store: BarStore instance providing bar history and opening range.
            open_position: Optional position dict passed by the dispatcher.
                If not None and layer_name matches, this layer treats the symbol
                as having an existing open position.

        Returns:
            List[SignalResult]: Empty list, a single EXIT signal, or a single
                BUY signal. Never returns more than one element.

        Example:
            >>> layer = OpeningRangeBreakoutStrategy()
            >>> signals = layer.evaluate_bar("AAPL", bar, bar_store, None)
        """
        current_close = float(bar.get("close", 0.0))
        current_high = float(bar.get("high", 0.0))
        current_low = float(bar.get("low", 0.0))
        current_open = float(bar.get("open", 0.0))
        current_volume = float(bar.get("volume", 0.0))

        if current_close <= 0.0 or current_high <= 0.0 or current_low <= 0.0:
            logger.debug("%s: invalid bar data close=%.4f", symbol, current_close)
            return []

        # Sync external position state if dispatcher provides it and we lost tracking.
        if (
            open_position is not None
            and open_position.get("layer_name") == self.layer_name
            and symbol not in self._open_positions
        ):
            or_data = bar_store.get_opening_range(symbol)
            or_high = float(or_data["high"]) if or_data else 0.0
            or_size = self._opening_range_size.get(symbol, 0.0)
            self._open_positions[symbol] = {
                "entry_price": float(open_position.get("entry_price", current_close)),
                "entry_time": open_position.get("entry_time", datetime.now(timezone.utc)),
                "stop_price": float(open_position.get("stop_price", 0.0)),
                "bars_held": int(open_position.get("bars_held", 0)),
                "opening_range_high": or_high,
                "opening_range_size": or_size,
            }
            # Ensure symbol is in triggered set since it already has a position.
            self._triggered_today.add(symbol)

        # Increment bars_held for any open position.
        if symbol in self._open_positions:
            self._open_positions[symbol]["bars_held"] += 1

        # Opening range must be finalized before this layer operates at all.
        opening_range = bar_store.get_opening_range(symbol)
        if opening_range is None:
            return []

        or_high = float(opening_range["high"])
        or_low = float(opening_range["low"])
        or_size = or_high - or_low

        if or_size <= 0.0:
            logger.debug("%s: degenerate opening range (size=%.4f)", symbol, or_size)
            return []

        # Fetch bar history for indicator computation.
        df = bar_store.get_bars(symbol, "1Min", self._MIN_BARS + 14)
        if len(df) < self._MIN_BARS:
            logger.debug(
                "%s: insufficient bars (%d < %d)", symbol, len(df), self._MIN_BARS
            )
            return []

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]
        volumes = df["volume"]

        atr_arr = atr(highs, lows, closes, 14)
        current_atr = atr_arr[-1]

        if np.isnan(current_atr) or current_atr <= 0.0:
            logger.debug("%s: ATR NaN or zero — skipping", symbol)
            return []

        # volume_ratio needs a series that includes the current bar.
        # Append current bar volume to the historical volume series.
        vol_series = pd.concat(
            [volumes, pd.Series([current_volume])], ignore_index=True
        )
        vol_ratio = volume_ratio(vol_series, period=20)

        # Bar position for the current bar.
        bp = bar_position(current_open, current_high, current_low, current_close)

        # Check exit conditions if a position is open.
        if symbol in self._open_positions:
            exit_signal = self._check_exit(
                symbol=symbol,
                current_close=current_close,
                or_high=or_high,
                or_size=or_size,
            )
            if exit_signal is not None:
                return [exit_signal]

        # Check entry conditions — only if no open position and not triggered today.
        if symbol not in self._open_positions and symbol not in self._triggered_today:
            entry_signal = self._check_entry(
                symbol=symbol,
                current_close=current_close,
                current_high=current_high,
                current_low=current_low,
                current_atr=current_atr,
                vol_ratio=vol_ratio,
                bp=bp,
                or_high=or_high,
                or_low=or_low,
                or_size=or_size,
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
        """Determine whether an open ORB position should be closed immediately.

        Called by StreamEngine on stop-loss tick checks and by EODManager at
        15:30 ET. Does not modify internal bars_held counter.

        Args:
            symbol: Ticker symbol of the open position.
            bar: Current bar dict with open, high, low, close, volume, timestamp.
            bar_store: BarStore instance.
            position_data: Dict with at minimum entry_price, entry_time,
                stop_price, bars_held, layer_name.

        Returns:
            bool: True if the position should be closed.

        Example:
            >>> layer = OpeningRangeBreakoutStrategy()
            >>> should_close = layer.should_exit(
            ...     "AAPL", current_bar, bar_store,
            ...     {"entry_price": 155.0, "stop_price": 153.5,
            ...      "bars_held": 45, "entry_time": datetime(...)})
            True
        """
        current_close = float(bar.get("close", 0.0))
        if current_close <= 0.0:
            return False

        stop_price = float(position_data.get("stop_price", 0.0))
        bars_held = int(position_data.get("bars_held", 0))
        entry_price = float(position_data.get("entry_price", 0.0))
        entry_time = position_data.get("entry_time")

        # Retrieve opening range from bar_store.
        opening_range = bar_store.get_opening_range(symbol)
        or_high = 0.0
        or_size = 0.0
        if opening_range is not None:
            or_high = float(opening_range["high"])
            or_low = float(opening_range["low"])
            or_size = or_high - or_low

        # Use stored opening range size if available from internal state.
        if symbol in self._opening_range_size:
            or_size = self._opening_range_size[symbol]

        # Compute minutes held from wall clock when possible.
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

        # Hard time-stop
        if minutes_held >= self.max_hold_minutes:
            logger.info(
                "%s [%s] EXIT — time stop minutes_held=%d >= %d",
                symbol, self.layer_name, minutes_held, self.max_hold_minutes,
            )
            return True

        # Stop-loss
        if stop_price > 0.0 and current_close < stop_price:
            logger.info(
                "%s [%s] EXIT — stop_loss close=%.4f < stop=%.4f",
                symbol, self.layer_name, current_close, stop_price,
            )
            return True

        # Breakout failure: price falls back below OR high
        if or_high > 0.0 and current_close < or_high * (1.0 - self.breakout_failure_pct):
            logger.info(
                "%s [%s] EXIT — breakout failure close=%.4f < or_high*(1-%.4f)=%.4f",
                symbol, self.layer_name, current_close,
                self.breakout_failure_pct, or_high * (1.0 - self.breakout_failure_pct),
            )
            return True

        # Profit target: 2x extension
        if entry_price > 0.0 and or_size > 0.0:
            profit_target = entry_price + self.extension_target_multiplier * or_size
            if current_close >= profit_target:
                logger.info(
                    "%s [%s] EXIT — profit target close=%.4f >= target=%.4f",
                    symbol, self.layer_name, current_close, profit_target,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Daily state management
    # ------------------------------------------------------------------

    def reset_daily_state(self) -> None:
        """Reset all per-session state at market open (09:30 ET).

        Clears the triggered-today set, open positions, and stored opening
        range sizes. Call this once each morning before the bar stream starts.

        Example:
            >>> layer.reset_daily_state()
        """
        prev_triggered = len(self._triggered_today)
        prev_positions = len(self._open_positions)
        self._triggered_today.clear()
        self._open_positions.clear()
        self._opening_range_size.clear()
        logger.info(
            "%s daily state reset — cleared %d triggered symbols, %d open positions",
            self.layer_name,
            prev_triggered,
            prev_positions,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_exit(
        self,
        symbol: str,
        current_close: float,
        or_high: float,
        or_size: float,
    ) -> Optional[SignalResult]:
        """Evaluate all exit conditions for an open ORB position.

        Exit conditions (any one triggers exit):
          1. Price falls below or_high * 0.998 (breakout failure).
          2. Price reaches entry_price + 2 * or_size (profit target).
          3. Price falls below stop_price (hard stop-loss).
          4. bars_held >= max_hold_minutes (time stop).

        Args:
            symbol: Ticker symbol.
            current_close: Current bar close price.
            or_high: Opening range high price.
            or_size: Opening range size (or_high - or_low).

        Returns:
            Optional[SignalResult]: EXIT signal if any condition is met, else None.

        Example:
            >>> signal = layer._check_exit("AAPL", 154.0, 155.0, 2.0)
            >>> signal.signal if signal else None
            'EXIT'
        """
        pos = self._open_positions[symbol]
        bars_held = pos["bars_held"]
        stop_price = pos["stop_price"]
        entry_price = pos["entry_price"]
        stored_or_size = pos.get("opening_range_size", or_size)

        exit_reason: Optional[str] = None

        # Hard time-stop
        if bars_held >= self.max_hold_minutes:
            exit_reason = f"time_stop bars_held={bars_held} >= {self.max_hold_minutes}"

        # Stop-loss
        elif stop_price > 0.0 and current_close < stop_price:
            exit_reason = f"stop_loss close={current_close:.4f} < stop={stop_price:.4f}"

        # Breakout failure
        elif or_high > 0.0 and current_close < or_high * (1.0 - self.breakout_failure_pct):
            exit_reason = (
                f"breakout_failure close={current_close:.4f} < "
                f"or_high*(1-{self.breakout_failure_pct:.4f})="
                f"{or_high * (1.0 - self.breakout_failure_pct):.4f}"
            )

        # Profit target
        elif stored_or_size > 0.0:
            profit_target = entry_price + self.extension_target_multiplier * stored_or_size
            if current_close >= profit_target:
                exit_reason = (
                    f"profit_target close={current_close:.4f} >= target={profit_target:.4f}"
                )

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
        # Note: symbol stays in _triggered_today to prevent re-entry same session.

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
                "opening_range_high": or_high,
                "opening_range_size": stored_or_size,
            },
        )

    def _check_entry(
        self,
        symbol: str,
        current_close: float,
        current_high: float,
        current_low: float,
        current_atr: float,
        vol_ratio: float,
        bp: float,
        or_high: float,
        or_low: float,
        or_size: float,
    ) -> Optional[SignalResult]:
        """Evaluate all ORB entry conditions for a new long position.

        Entry requires ALL of:
          1. current_close > or_high * (1 + breakout_buffer_pct).
          2. vol_ratio > volume_confirmation_ratio (default 1.5).
          3. bar_position > bar_position_threshold (close in top 30%).
          4. or_size <= max_range_atr_multiplier * daily_atr_proxy, where
             daily_atr_proxy = current_atr * 30.

        Stop price = or_high - or_size * 0.5
        Confidence = min(vol_ratio / 3.0, 1.0)

        Args:
            symbol: Ticker symbol.
            current_close: Current bar close price.
            current_high: Current bar high price.
            current_low: Current bar low price.
            current_atr: Current ATR(14) on 1-min bars.
            vol_ratio: Current bar volume ratio vs. prior 20-bar average.
            bp: Bar position — (close - low) / (high - low).
            or_high: Opening range high price.
            or_low: Opening range low price.
            or_size: Opening range size (or_high - or_low).

        Returns:
            Optional[SignalResult]: BUY signal if all conditions pass, else None.

        Example:
            >>> signal = layer._check_entry(
            ...     "AAPL", 155.20, 155.30, 154.90, 0.25, 2.3, 0.82,
            ...     154.80, 152.50, 2.30)
            >>> signal.signal if signal else None
            'BUY'
        """
        # Condition 1: price convincingly above opening range high
        breakout_level = or_high * (1.0 + self.breakout_buffer_pct)
        if current_close <= breakout_level:
            return None

        # Condition 2: volume confirmation
        if vol_ratio <= self.volume_confirmation_ratio:
            return None

        # Condition 3: bullish bar structure
        if bp <= self.bar_position_threshold:
            return None

        # Condition 5: skip if opening range is too large relative to daily ATR proxy
        # daily_atr_proxy = current_atr * 30 (30 minutes ~ typical daily range factor)
        daily_atr_proxy = current_atr * 30.0
        if daily_atr_proxy > 0.0 and or_size > self.max_range_atr_multiplier * daily_atr_proxy:
            logger.debug(
                "%s [%s] skipping — or_size=%.4f > %.1f * daily_atr_proxy=%.4f",
                symbol,
                self.layer_name,
                or_size,
                self.max_range_atr_multiplier,
                daily_atr_proxy,
            )
            return None

        # All conditions met — compute stop price and confidence.
        stop_price = or_high - or_size * self.stop_pct_of_range
        confidence = min(vol_ratio / 3.0, 1.0)

        logger.info(
            "%s [%s] BUY — close=%.4f or_high=%.4f or_size=%.4f "
            "vol_ratio=%.2f bar_pos=%.3f stop=%.4f conf=%.3f atr=%.4f",
            symbol,
            self.layer_name,
            current_close,
            or_high,
            or_size,
            vol_ratio,
            bp,
            stop_price,
            confidence,
            current_atr,
        )

        self._triggered_today.add(symbol)
        self._opening_range_size[symbol] = or_size
        self._open_positions[symbol] = {
            "entry_price": current_close,
            "entry_time": datetime.now(timezone.utc),
            "stop_price": stop_price,
            "bars_held": 0,
            "opening_range_high": or_high,
            "opening_range_size": or_size,
        }

        return SignalResult(
            symbol=symbol,
            signal="BUY",
            confidence=confidence,
            signal_price=current_close,
            layer_name=self.layer_name,
            stop_price=stop_price,
            metadata={
                "opening_range_high": or_high,
                "opening_range_low": or_low,
                "opening_range_size": round(or_size, 4),
                "breakout_level": round(breakout_level, 4),
                "volume_ratio": round(vol_ratio, 4),
                "bar_position": round(bp, 4),
                "atr_1min": round(current_atr, 4),
                "daily_atr_proxy": round(daily_atr_proxy, 4),
            },
        )

    def clear_position(self, symbol: str) -> None:
        """Remove internal position state for a symbol without generating a signal.

        Called by the order manager or EOD handler after a fill is confirmed.
        The symbol remains in _triggered_today so no second entry can fire.

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

    def was_triggered_today(self, symbol: str) -> bool:
        """Check whether an ORB entry has already been taken for this symbol today.

        Args:
            symbol: Ticker symbol to check.

        Returns:
            bool: True if an ORB signal was generated for this symbol today.

        Example:
            >>> layer.was_triggered_today("AAPL")
            False
        """
        return symbol in self._triggered_today

    def get_position_info(self, symbol: str) -> Optional[dict]:
        """Return the internal position state for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Optional[dict]: Position dict with entry_price, entry_time,
                stop_price, bars_held, opening_range_high, opening_range_size
                — or None if no open position exists.

        Example:
            >>> info = layer.get_position_info("AAPL")
            >>> info["bars_held"] if info else None
            12
        """
        return self._open_positions.get(symbol)
