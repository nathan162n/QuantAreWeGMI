"""Layer 4 — Volume Surge Momentum strategy for high-frequency intraday trading.

Fires on every 1-minute bar. Identifies explosive volume events where the
current bar's volume exceeds 3x the rolling 20-bar average, the bar closes
in the top 20% of its own range (bullish bar), and the bar is a green candle
(close > open). Designed to capture the initial momentum leg of volume-driven
price surges before the first 15:00 ET cut-off.

Exits when volume momentum fades (3 consecutive sub-1.5x bars), RSI(14) hits
overbought territory above 75, a hard 120-minute ceiling is reached, or the
stop-loss is breached. Stop is set at 1.5 * ATR(14, 1-min bars) below entry.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.indicators import atr, bar_position, rsi, volume_ratio
from src.strategy.base_strategy import BaseStrategy, SignalResult
from src.utils.logger import get_logger

logger = get_logger("strategy.L4_VOL_SURGE")

# Module-level named constants kept as fallback defaults for test compatibility.
_VOLUME_SURGE_THRESHOLD = 3.0      # current bar must be > 3x the 20-bar avg
_BAR_POSITION_THRESHOLD = 0.80     # close must be in top 20% of bar range
_MAX_SESSION_BARS = 330            # ~15:00 ET (09:30 + 330 min = 15:00)

# Exit thresholds
_VOLUME_FADE_RATIO = 1.5           # below this ratio = fading bar
_CONSECUTIVE_FADE_BARS = 3         # exit when 3 consecutive fade bars observed
_RSI_OVERBOUGHT = 75.0             # RSI(14) exit level
_MAX_MINUTES_HELD = 120            # hard 2-hour safety ceiling

# Stop multiplier
_STOP_ATR_MULTIPLIER = 1.5         # 1.5 * ATR(14, 1-min) below entry

# Indicator periods
_RSI_PERIOD = 14
_ATR_PERIOD = 14
_VOLUME_AVG_PERIOD = 20

# Number of 1-min bars to fetch for indicator computation
_FETCH_1MIN_BARS = 25              # 21 for volume_ratio (20 prior + 1 current) + buffer
_FETCH_RSI_BARS = 20               # RSI(14) needs 15+ bars; 20 gives a safe margin
_FETCH_ATR_BARS = 20               # ATR(14) lookback

# Default parameters for config loading
_LAYER4_DEFAULTS = {
    "volume_surge_ratio": float(_VOLUME_SURGE_THRESHOLD),
    "bar_position_threshold": float(_BAR_POSITION_THRESHOLD),
    "rsi_overbought_exit": float(_RSI_OVERBOUGHT),
    "max_hold_minutes": float(_MAX_MINUTES_HELD),
    "consecutive_fade_bars": float(_CONSECUTIVE_FADE_BARS),
    "volume_fade_ratio": float(_VOLUME_FADE_RATIO),
    "stop_atr_multiplier": float(_STOP_ATR_MULTIPLIER),
    "stop_min_pct": 0.003,
    "no_entry_after": "15:00",
}


def _load_layer_config(layer_key: str, defaults: dict) -> dict:
    """Load strategy layer config from config.yml, falling back to defaults."""
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


class VolumeSurgeMomentumStrategy(BaseStrategy):
    """Layer 4 momentum strategy that enters on explosive volume surges.

    Entry fires when the current 1-minute bar exhibits all of:
        - Volume > 3x the rolling 20-bar average
        - Bar closes in the top 20% of its high-low range (bullish conviction)
        - Green bar: close > open
        - Session bar count <= 330 (before approximately 15:00 ET)
        - No existing open position from this layer for the symbol

    Exit fires when any of:
        - Three consecutive 1-min bars each have volume below 1.5x avg (fade)
        - RSI(14) on recent 1-min closes exceeds 75 (overbought)
        - Position has been held for >= 120 minutes (hard ceiling)
        - Current close drops below the computed stop price

    Stop is placed at entry_price - 1.5 * ATR(14, 1-min bars).

    Attributes:
        _open_positions: Per-symbol state for open positions managed by this layer.
            Keys are symbol strings. Each value is a dict with:
                entry_price (float): Fill price at entry.
                entry_time (datetime): UTC timestamp at entry.
                stop_price (float): Computed stop-loss price.
                bars_held (int): 1-min bars elapsed since entry.
                entry_bar_count (int): Session bar count at entry time.
        _volume_fade_counter: Per-symbol count of consecutive sub-1.5x volume bars
            observed since the position was opened. Resets to 0 on any bar
            where volume is >= 1.5x the 20-bar average.
    """

    def __init__(self) -> None:
        """Initialize Layer 4, loading parameters from config.yml.

        Config values (strategies.layer4) override module-level defaults at
        runtime. Module-level constants (_VOLUME_SURGE_THRESHOLD etc.) are
        preserved as named fallbacks for test compatibility.

        Example:
            >>> layer = VolumeSurgeMomentumStrategy()
            >>> layer.layer_name
            'L4_VOL_SURGE'
        """
        cfg = _load_layer_config("layer4", _LAYER4_DEFAULTS)

        self._surge_threshold: float = float(cfg["volume_surge_ratio"])
        self._bar_pos_threshold: float = float(cfg["bar_position_threshold"])
        self._rsi_overbought: float = float(cfg["rsi_overbought_exit"])
        self._max_minutes: float = float(cfg["max_hold_minutes"])
        self._fade_bars: int = int(cfg["consecutive_fade_bars"])
        self._fade_ratio: float = float(cfg["volume_fade_ratio"])
        self._stop_atr_multiplier: float = float(cfg["stop_atr_multiplier"])
        self._stop_min_pct: float = float(cfg["stop_min_pct"])
        self._no_entry_after: str = str(cfg.get("no_entry_after", "15:00"))

        self._open_positions: Dict[str, dict] = {}
        self._volume_fade_counter: Dict[str, int] = {}

        logger.info(
            "L4_VOL_SURGE initialized — surge_thr=%.1f bar_pos=%.2f rsi_ob=%.0f "
            "max_min=%.0f fade_bars=%d fade_ratio=%.1f stop_atr=%.2f stop_min_pct=%.4f "
            "no_entry_after=%s",
            self._surge_threshold,
            self._bar_pos_threshold,
            self._rsi_overbought,
            self._max_minutes,
            self._fade_bars,
            self._fade_ratio,
            self._stop_atr_multiplier,
            self._stop_min_pct,
            self._no_entry_after,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def layer_name(self) -> str:
        """Unique identifier for this strategy layer.

        Returns:
            str: Always "L4_VOL_SURGE".

        Example:
            >>> VolumeSurgeMomentumStrategy().layer_name
            'L4_VOL_SURGE'
        """
        return "L4_VOL_SURGE"

    def evaluate_bar(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        open_position: Optional[dict] = None,
    ) -> List[SignalResult]:
        """Evaluate the current 1-minute bar and decide whether to BUY or EXIT.

        Called by BarDispatcher on every 1-minute bar. Checks exit conditions
        first for any open position, then evaluates entry if no position exists.

        Args:
            symbol: Ticker symbol being evaluated.
            bar: Current 1-minute bar dict with open, high, low, close,
                volume, timestamp keys.
            bar_store: BarStore instance providing access to historical bars.
            open_position: Optional position dict passed by the caller.
                If not None and layer_name matches, exit logic is applied.
                Format: entry_price, entry_time, qty, stop_price, bars_held,
                layer_name.

        Returns:
            List[SignalResult]: Zero or one element.
                - [] when there is nothing to do (HOLD).
                - [SignalResult(signal='EXIT')] when exit criteria are met.
                - [SignalResult(signal='BUY')] when entry criteria are met.

        Example:
            >>> signals = layer.evaluate_bar("NVDA", bar, store, None)
            >>> if signals and signals[0].signal == "BUY":
            ...     print(f"Surge detected: confidence={signals[0].confidence:.2f}")
        """
        current_close = float(bar.get("close", 0.0))
        if current_close <= 0.0:
            logger.debug("%s: bar close is zero/negative, skipping", symbol)
            return []

        # Increment bars_held for any tracked open position.
        if symbol in self._open_positions:
            self._open_positions[symbol]["bars_held"] += 1

        # Reconcile externally supplied open position not yet in our state.
        if (
            symbol not in self._open_positions
            and open_position is not None
            and open_position.get("layer_name") == self.layer_name
        ):
            self._adopt_external_position(symbol, open_position)
            self._open_positions[symbol]["bars_held"] += 1

        # --- EXIT EVALUATION ---
        if symbol in self._open_positions:
            # Update the volume fade counter for this bar before checking exit.
            self._update_fade_counter(symbol, bar, bar_store)
            exit_signal = self._evaluate_exit(symbol, bar, bar_store)
            if exit_signal is not None:
                del self._open_positions[symbol]
                self._volume_fade_counter.pop(symbol, None)
                return [exit_signal]
            return []

        # --- ENTRY EVALUATION ---
        return self._evaluate_entry(symbol, bar, bar_store)

    def should_exit(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        position_data: dict,
    ) -> bool:
        """Determine if an open Layer 4 position should be exited.

        Called by StreamEngine for stop-loss checks and by EODManager.
        Applies the same exit criteria as evaluate_bar() without modifying
        internal counter state.

        Args:
            symbol: Ticker symbol of the open position.
            bar: Current 1-minute bar dict.
            bar_store: BarStore instance.
            position_data: Dict with entry_price, entry_time, qty,
                stop_price, bars_held, layer_name.

        Returns:
            bool: True if the position should be closed immediately.

        Example:
            >>> layer.should_exit("NVDA", bar, store,
            ...     {"entry_price": 500.0, "stop_price": 496.0,
            ...      "bars_held": 45, "entry_time": datetime(...),
            ...      "layer_name": "L4_VOL_SURGE"})
            False
        """
        current_close = float(bar.get("close", 0.0))
        stop_price = float(position_data.get("stop_price", 0.0))
        bars_held = int(position_data.get("bars_held", 0))
        entry_time = position_data.get("entry_time")

        # Stop-loss breach
        if stop_price > 0.0 and current_close < stop_price:
            logger.info(
                "%s: should_exit=True — stop breach close=%.4f < stop=%.4f",
                symbol,
                current_close,
                stop_price,
            )
            return True

        # Hard minutes ceiling
        if entry_time is not None:
            now_utc = datetime.now(timezone.utc)
            if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            minutes_elapsed = (now_utc - entry_time).total_seconds() / 60.0
            if minutes_elapsed >= self._max_minutes:
                logger.info(
                    "%s: should_exit=True — minutes_held=%.1f >= %.0f",
                    symbol,
                    minutes_elapsed,
                    self._max_minutes,
                )
                return True

        # Volume fade: check via internal counter if available, otherwise
        # compute fresh from bar_store.
        fade_count = self._volume_fade_counter.get(symbol, 0)
        if fade_count >= self._fade_bars:
            logger.info(
                "%s: should_exit=True — consecutive fade bars=%d >= %d",
                symbol,
                fade_count,
                self._fade_bars,
            )
            return True

        # RSI overbought check
        rsi_bars = bar_store.get_bars(symbol, "1Min", _FETCH_RSI_BARS)
        if rsi_bars is not None and len(rsi_bars) >= _RSI_PERIOD + 1:
            rsi_arr = rsi(rsi_bars["close"], _RSI_PERIOD)
            last_rsi = rsi_arr[-1]
            if not np.isnan(last_rsi) and last_rsi > self._rsi_overbought:
                logger.info(
                    "%s: should_exit=True — RSI(14)=%.2f > %.1f",
                    symbol,
                    last_rsi,
                    self._rsi_overbought,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_entry(
        self,
        symbol: str,
        bar: dict,
        bar_store,
    ) -> List[SignalResult]:
        """Evaluate all entry conditions and return a BUY signal if met.

        Entry requires ALL of:
            1. volume_ratio(last 21 bars, period=20) > 3.0
            2. bar_position(open, high, low, close) > 0.80
            3. close > open (green bar)
            4. bar_store.get_bar_count(symbol) <= 330 (before ~15:00 ET)
            5. No existing position from this layer

        Stop is set at entry_price - 1.5 * ATR(14, 1-min bars).
        Confidence = min((volume_ratio - 3.0) / 3.0, 1.0).

        Args:
            symbol: Ticker symbol.
            bar: Current 1-minute bar dict.
            bar_store: BarStore instance.

        Returns:
            List[SignalResult]: Single-element list with BUY signal if all
                entry conditions are met, otherwise empty list.

        Example:
            >>> signals = layer._evaluate_entry("NVDA", bar, store)
            >>> if signals:
            ...     print(f"vol_ratio={signals[0].metadata['volume_ratio']:.2f}")
        """
        bar_open = float(bar.get("open", 0.0))
        bar_high = float(bar.get("high", 0.0))
        bar_low = float(bar.get("low", 0.0))
        bar_close = float(bar.get("close", 0.0))
        bar_vol = float(bar.get("volume", 0.0))

        if bar_open <= 0.0 or bar_high <= 0.0 or bar_close <= 0.0:
            return []

        # Phase 4B time gate: no new entries after configured cut-off.
        # Uses bar timestamp so test bars with historical timestamps are
        # correctly evaluated rather than comparing against wall-clock time.
        bar_ts = bar.get("timestamp")
        if self._no_entry_after and _is_past_et_time(self._no_entry_after, bar_ts):
            logger.debug(
                "%s: entry blocked — past no_entry_after=%s",
                symbol, self._no_entry_after,
            )
            return []

        # Condition 4: session time guard — must be before bar 330
        bar_count = bar_store.get_bar_count(symbol)
        if bar_count > _MAX_SESSION_BARS:
            logger.debug(
                "%s: bar_count=%d > %d, too late in session for entry",
                symbol,
                bar_count,
                _MAX_SESSION_BARS,
            )
            return []

        # Condition 3: green bar (close > open)
        if bar_close <= bar_open:
            logger.debug(
                "%s: bar is not green — close=%.4f <= open=%.4f",
                symbol,
                bar_close,
                bar_open,
            )
            return []

        # Condition 2: bullish bar — close in top threshold% of range
        bp = bar_position(bar_open, bar_high, bar_low, bar_close)
        if bp <= self._bar_pos_threshold:
            logger.debug(
                "%s: bar_position=%.4f <= %.2f, not bullish enough",
                symbol,
                bp,
                self._bar_pos_threshold,
            )
            return []

        # Condition 1: volume surge — fetch 21 bars (20 prior + current)
        # volume_ratio() uses the last element as current and the preceding
        # 'period' elements as the baseline average.
        vol_bars = bar_store.get_bars(symbol, "1Min", _VOLUME_AVG_PERIOD + 1)
        if vol_bars is None or len(vol_bars) < 2:
            logger.debug(
                "%s: insufficient volume bars for ratio computation", symbol
            )
            return []

        # Append current bar volume to the tail of the series so that
        # volume_ratio() picks it up as the current observation.
        vol_series = pd.concat(
            [vol_bars["volume"], pd.Series([bar_vol])], ignore_index=True
        )
        vol_rat = volume_ratio(vol_series, period=_VOLUME_AVG_PERIOD)

        if vol_rat <= self._surge_threshold:
            logger.debug(
                "%s: volume_ratio=%.4f <= %.1f, no surge",
                symbol,
                vol_rat,
                self._surge_threshold,
            )
            return []

        # Compute ATR(14) for stop placement
        atr_value = 0.0
        atr_bars = bar_store.get_bars(symbol, "1Min", _FETCH_ATR_BARS)
        if atr_bars is not None and len(atr_bars) >= _ATR_PERIOD + 1:
            atr_arr = atr(atr_bars["high"], atr_bars["low"], atr_bars["close"], _ATR_PERIOD)
            last_atr = atr_arr[-1]
            if not np.isnan(last_atr):
                atr_value = float(last_atr)

        stop_price = (
            bar_close - self._stop_atr_multiplier * atr_value
            if atr_value > 0.0
            else bar_close * (1.0 - self._stop_min_pct)  # fallback: stop_min_pct below close
        )

        # Confidence: linear scale from surge_threshold (0%) to 2x threshold (100%)
        confidence = min((vol_rat - self._surge_threshold) / self._surge_threshold, 1.0)
        confidence = round(max(confidence, 0.0), 4)

        entry_time = bar.get("timestamp")
        if entry_time is None:
            entry_time = datetime.now(timezone.utc)
        elif hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        self._open_positions[symbol] = {
            "entry_price": bar_close,
            "entry_time": entry_time,
            "stop_price": stop_price,
            "bars_held": 0,
            "entry_bar_count": bar_count,
        }
        # Reset the fade counter for this new position.
        self._volume_fade_counter[symbol] = 0

        logger.info(
            "%s: BUY signal — vol_ratio=%.2f, bar_pos=%.4f, close=%.4f > open=%.4f, "
            "stop=%.4f, atr=%.4f, confidence=%.4f",
            symbol,
            vol_rat,
            bp,
            bar_close,
            bar_open,
            stop_price,
            atr_value,
            confidence,
        )

        return [
            SignalResult(
                symbol=symbol,
                signal="BUY",
                confidence=confidence,
                signal_price=bar_close,
                layer_name=self.layer_name,
                stop_price=round(stop_price, 6),
                metadata={
                    "volume_ratio": round(float(vol_rat), 4),
                    "bar_position": round(float(bp), 4),
                    "bar_open": round(bar_open, 4),
                    "bar_close": round(bar_close, 4),
                    "bar_volume": bar_vol,
                    "atr_14_1min": round(atr_value, 6),
                    "session_bar_count": bar_count,
                },
            )
        ]

    def _evaluate_exit(
        self,
        symbol: str,
        bar: dict,
        bar_store,
    ) -> Optional[SignalResult]:
        """Evaluate exit conditions for a currently held position.

        Exit fires when ANY of:
            1. Consecutive sub-1.5x volume fade bars >= 3
            2. RSI(14) on recent 1-min closes > 75 (overbought)
            3. Minutes held >= 120 (hard 2-hour safety ceiling)
            4. Current 1-min close < stop_price (stop-loss breach)

        The volume fade counter is updated externally by _update_fade_counter()
        before this method is called.

        Args:
            symbol: Ticker symbol with an open position.
            bar: Current 1-minute bar dict.
            bar_store: BarStore instance.

        Returns:
            Optional[SignalResult]: EXIT signal if any exit criterion fires,
                otherwise None (continue holding).

        Example:
            >>> signal = layer._evaluate_exit("NVDA", bar, store)
            >>> signal.metadata["exit_reason"] if signal else "holding"
            'holding'
        """
        pos = self._open_positions[symbol]
        current_close = float(bar.get("close", 0.0))
        stop_price = pos["stop_price"]
        bars_held = pos["bars_held"]
        entry_time = pos["entry_time"]
        entry_price = pos["entry_price"]
        fade_count = self._volume_fade_counter.get(symbol, 0)

        exit_reason: Optional[str] = None

        # Condition 4: stop-loss breach
        if stop_price > 0.0 and current_close < stop_price:
            exit_reason = f"stop_breach close={current_close:.4f} < stop={stop_price:.4f}"

        # Condition 3: hard minutes ceiling
        elif entry_time is not None:
            now_utc = datetime.now(timezone.utc)
            if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            minutes_elapsed = (now_utc - entry_time).total_seconds() / 60.0
            if minutes_elapsed >= self._max_minutes:
                exit_reason = (
                    f"max_minutes_held={minutes_elapsed:.1f} >= {self._max_minutes:.0f}"
                )

        # Condition 1: consecutive volume fade
        if exit_reason is None and fade_count >= self._fade_bars:
            exit_reason = (
                f"volume_fade consecutive_fade_bars={fade_count} >= {self._fade_bars}"
            )

        # Condition 2: RSI overbought
        if exit_reason is None:
            rsi_bars = bar_store.get_bars(symbol, "1Min", _FETCH_RSI_BARS)
            if rsi_bars is not None and len(rsi_bars) >= _RSI_PERIOD + 1:
                rsi_arr = rsi(rsi_bars["close"], _RSI_PERIOD)
                last_rsi = rsi_arr[-1]
                if not np.isnan(last_rsi) and last_rsi > self._rsi_overbought:
                    exit_reason = (
                        f"rsi_overbought RSI(14)={last_rsi:.2f} > {self._rsi_overbought:.0f}"
                    )

        if exit_reason is None:
            return None

        pnl_pct = (
            (current_close - entry_price) / entry_price * 100.0
            if entry_price > 0
            else 0.0
        )

        logger.info(
            "%s: EXIT signal — reason=%s, bars_held=%d, pnl=%.2f%%",
            symbol,
            exit_reason,
            bars_held,
            pnl_pct,
        )

        return SignalResult(
            symbol=symbol,
            signal="EXIT",
            confidence=1.0,
            signal_price=current_close,
            layer_name=self.layer_name,
            stop_price=stop_price,
            metadata={
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "fade_counter": fade_count,
                "entry_price": round(entry_price, 4),
                "exit_price": round(current_close, 4),
                "pnl_pct": round(pnl_pct, 4),
            },
        )

    def _update_fade_counter(
        self,
        symbol: str,
        bar: dict,
        bar_store,
    ) -> None:
        """Update the consecutive volume fade counter for an open position.

        Fetches the 20-bar rolling average volume and compares the current
        bar's volume against the _VOLUME_FADE_RATIO threshold (1.5x). The
        counter is incremented when volume falls below the threshold and
        reset to 0 when volume meets or exceeds it.

        Args:
            symbol: Ticker symbol with an open position.
            bar: Current 1-minute bar dict containing the current volume.
            bar_store: BarStore instance used to retrieve recent bars for
                computing the 20-bar rolling average volume.

        Example:
            >>> layer._update_fade_counter("NVDA", bar, store)
            >>> layer._volume_fade_counter.get("NVDA")
            1
        """
        current_vol = float(bar.get("volume", 0.0))

        # Fetch recent bars to compute the 20-bar baseline average.
        vol_bars = bar_store.get_bars(symbol, "1Min", _VOLUME_AVG_PERIOD + 1)
        if vol_bars is None or len(vol_bars) < 2:
            # Cannot determine average — do not update counter.
            return

        # Append current bar volume so volume_ratio() can compute correctly.
        vol_series = pd.concat(
            [vol_bars["volume"], pd.Series([current_vol])], ignore_index=True
        )
        vol_rat = volume_ratio(vol_series, period=_VOLUME_AVG_PERIOD)

        if vol_rat < self._fade_ratio:
            self._volume_fade_counter[symbol] = (
                self._volume_fade_counter.get(symbol, 0) + 1
            )
            logger.debug(
                "%s: fade bar — vol_ratio=%.4f < %.1f, counter=%d",
                symbol,
                vol_rat,
                self._fade_ratio,
                self._volume_fade_counter[symbol],
            )
        else:
            self._volume_fade_counter[symbol] = 0
            logger.debug(
                "%s: volume sustained — vol_ratio=%.4f >= %.1f, counter reset",
                symbol,
                vol_rat,
                self._fade_ratio,
            )

    def _adopt_external_position(self, symbol: str, position_data: dict) -> None:
        """Reconcile an externally supplied position into internal state.

        Called when evaluate_bar() receives an open_position from the caller
        that has not yet been registered in _open_positions. This can happen
        after a restart or when the position manager rehydrates state.

        Args:
            symbol: Ticker symbol of the position to adopt.
            position_data: Dict with entry_price, entry_time, stop_price,
                bars_held, entry_bar_count, layer_name keys.

        Example:
            >>> layer._adopt_external_position("NVDA", {
            ...     "entry_price": 500.0, "entry_time": datetime(...),
            ...     "stop_price": 496.0, "bars_held": 12,
            ...     "entry_bar_count": 120, "layer_name": "L4_VOL_SURGE"})
        """
        entry_time = position_data.get("entry_time")
        if entry_time is None:
            entry_time = datetime.now(timezone.utc)
        elif hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        self._open_positions[symbol] = {
            "entry_price": float(position_data.get("entry_price", 0.0)),
            "entry_time": entry_time,
            "stop_price": float(position_data.get("stop_price", 0.0)),
            "bars_held": int(position_data.get("bars_held", 0)),
            "entry_bar_count": int(position_data.get("entry_bar_count", 0)),
        }
        # Initialise fade counter to zero if not already tracking.
        if symbol not in self._volume_fade_counter:
            self._volume_fade_counter[symbol] = 0

        logger.debug(
            "%s: adopted external position entry=%.4f stop=%.4f bars_held=%d",
            symbol,
            self._open_positions[symbol]["entry_price"],
            self._open_positions[symbol]["stop_price"],
            self._open_positions[symbol]["bars_held"],
        )
