"""Layer 3 — RSI(7) Reversal Scalp strategy for high-frequency intraday trading.

Operates on synthetic 5-minute bars constructed from 1-minute bar data stored
in the BarStore. Called every 3rd 1-minute bar by the BarDispatcher (cadence
is managed externally). Identifies extreme oversold conditions using RSI(7)
below 25 and requires a confirming 1-minute reversal bar before entering.

Exit logic is time-bounded at 10 bars (50 minutes) of synthetic 5-minute bars
held, or a hard 120-minute safety ceiling, whichever comes first. Stop-loss is
placed at 0.5 * ATR(14) below entry using the same synthetic 5-minute series.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.indicators import atr, rsi
from src.strategy.base_strategy import BaseStrategy, SignalResult
from src.utils.logger import get_logger

logger = get_logger("strategy.L3_RSI_SCALP")

# Minimum 1-minute bars needed to build enough synthetic 5-min bars for RSI(7).
# RSI(7) needs at least 8 5-min bars to produce a valid reading.
# We target 10 5-min bars minimum -> 50 1-min bars minimum.
# The helper fetches 80 to give ATR(14) room as well.
_MIN_1MIN_BARS_FOR_HELPER = 50
_FETCH_1MIN_BARS = 80

# Default parameters used when config.yml is absent (e.g. in tests).
_LAYER3_DEFAULTS = {
    "rsi_entry_threshold": 25.0,
    "rsi_exit_threshold": 55.0,
    "rsi_period": 7,
    "atr_period": 14,
    "max_hold_bars": 10,
    "max_hold_minutes": 120,
    "stop_atr_multiplier": 0.5,
    "stop_min_pct": 0.003,
    "min_session_bars": 10,
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


class RSIReversalScalpStrategy(BaseStrategy):
    """Layer 3 intraday scalp strategy based on RSI(7) reversals on synthetic 5-min bars.

    Entry fires when RSI(7) computed on synthetic 5-minute closes drops below 25
    (extreme oversold) while the current 1-minute bar closes higher than the
    previous 1-minute bar (confirming a reversal tick). The session must have
    at least 10 bars of data before entries are considered.

    Exit fires when any of the following are true:
        - RSI(7) on current synthetic 5-min closes exceeds 55 (recovered)
        - Current 1-min close drops below the computed stop price
        - Position has been held for 10 or more synthetic 5-min bars (~50 min)
        - Position has been held for 120 or more minutes (hard safety ceiling)

    Stop is placed at entry_price - 0.5 * ATR(14, synthetic 5-min bars).

    Attributes:
        _open_positions: Per-symbol state for open positions managed by this layer.
            Keys are symbol strings. Each value is a dict with:
                entry_price (float): Fill price at entry.
                entry_time (datetime): UTC timestamp at entry.
                stop_price (float): Computed stop-loss price.
                bars_held (int): Count of synthetic 5-min bars elapsed since entry.
    """

    def __init__(self) -> None:
        """Initialize Layer 3, loading parameters from config.yml.

        Config values (strategies.layer3) override module-level defaults at
        runtime. Falls back to hardcoded defaults when config is absent.

        Example:
            >>> layer = RSIReversalScalpStrategy()
            >>> layer.layer_name
            'L3_RSI_SCALP'
        """
        cfg = _load_layer_config("layer3", _LAYER3_DEFAULTS)

        self._rsi_entry: float = float(cfg["rsi_entry_threshold"])
        self._rsi_exit: float = float(cfg["rsi_exit_threshold"])
        self._rsi_period: int = int(cfg["rsi_period"])
        self._atr_period: int = int(cfg["atr_period"])
        self._max_bars_held: int = int(cfg["max_hold_bars"])
        self._max_minutes: float = float(cfg["max_hold_minutes"])
        self._stop_atr_multiplier: float = float(cfg["stop_atr_multiplier"])
        self._stop_min_pct: float = float(cfg["stop_min_pct"])
        self._min_session_bars: int = int(cfg["min_session_bars"])

        self._open_positions: Dict[str, dict] = {}

        logger.info(
            "L3_RSI_SCALP initialized — rsi_entry=%.1f rsi_exit=%.1f "
            "rsi_period=%d max_bars=%d max_min=%.0f stop_atr=%.2f stop_min_pct=%.4f",
            self._rsi_entry,
            self._rsi_exit,
            self._rsi_period,
            self._max_bars_held,
            self._max_minutes,
            self._stop_atr_multiplier,
            self._stop_min_pct,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def layer_name(self) -> str:
        """Unique identifier for this strategy layer.

        Returns:
            str: Always "L3_RSI_SCALP".

        Example:
            >>> RSIReversalScalpStrategy().layer_name
            'L3_RSI_SCALP'
        """
        return "L3_RSI_SCALP"

    def evaluate_bar(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        open_position: Optional[dict] = None,
    ) -> List[SignalResult]:
        """Evaluate the current 1-min bar and decide whether to BUY or EXIT.

        Called by BarDispatcher on every 3rd 1-minute bar for this symbol.
        First checks for exit conditions on any existing open position from
        this layer, then evaluates entry conditions if no position is held.

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
            >>> signals = layer.evaluate_bar("AAPL", current_bar, store, None)
            >>> if signals and signals[0].signal == "BUY":
            ...     print(signals[0].confidence)
        """
        current_close = float(bar.get("close", 0.0))
        if current_close <= 0.0:
            logger.debug("%s: bar close is zero/negative, skipping", symbol)
            return []

        # Increment bars_held for any tracked position before evaluating exit.
        # bars_held counts how many 5-min bar intervals have passed since entry.
        # Because this method is called every 3rd 1-min bar (~5 min cadence),
        # each call represents approximately one 5-min bar elapsed.
        if symbol in self._open_positions:
            self._open_positions[symbol]["bars_held"] += 1

        # --- EXIT EVALUATION ---
        if symbol in self._open_positions:
            exit_signal = self._evaluate_exit(symbol, bar, bar_store)
            if exit_signal is not None:
                del self._open_positions[symbol]
                return [exit_signal]
            # Position is open and holding — do not evaluate entry.
            return []

        # Also check caller-supplied open_position for this layer.
        if (
            open_position is not None
            and open_position.get("layer_name") == self.layer_name
        ):
            # External position not yet in our internal state; re-adopt it.
            self._adopt_external_position(symbol, open_position)
            self._open_positions[symbol]["bars_held"] += 1
            exit_signal = self._evaluate_exit(symbol, bar, bar_store)
            if exit_signal is not None:
                del self._open_positions[symbol]
                return [exit_signal]
            return []

        # --- ENTRY EVALUATION ---
        bar_count = bar_store.get_bar_count(symbol)
        if bar_count <= self._min_session_bars:
            logger.debug(
                "%s: bar_count=%d <= %d, too early in session for entry",
                symbol,
                bar_count,
                self._min_session_bars,
            )
            return []

        return self._evaluate_entry(symbol, bar, bar_store)

    def should_exit(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        position_data: dict,
    ) -> bool:
        """Determine if an open Layer 3 position should be exited.

        Called by StreamEngine for stop-loss checks and by EODManager.
        Applies the same exit criteria as evaluate_bar() without consuming
        or modifying internal bars_held state.

        Args:
            symbol: Ticker symbol of the open position.
            bar: Current 1-minute bar dict.
            bar_store: BarStore instance.
            position_data: Dict with entry_price, entry_time, qty,
                stop_price, bars_held, layer_name.

        Returns:
            bool: True if the position should be closed immediately.

        Example:
            >>> layer.should_exit("AAPL", bar, store,
            ...     {"entry_price": 150.0, "stop_price": 149.5,
            ...      "bars_held": 8, "entry_time": datetime(...),
            ...      "layer_name": "L3_RSI_SCALP"})
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

        # Maximum bars held (time stop)
        if bars_held >= self._max_bars_held:
            logger.info(
                "%s: should_exit=True — bars_held=%d >= %d",
                symbol,
                bars_held,
                self._max_bars_held,
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

        # RSI recovery exit
        five_min_df = self._build_5min_series(symbol, bar_store)
        if five_min_df is not None and len(five_min_df) >= self._rsi_period + 1:
            rsi_arr = rsi(five_min_df["close"], self._rsi_period)
            last_rsi = rsi_arr[-1]
            if not np.isnan(last_rsi) and last_rsi > self._rsi_exit:
                logger.info(
                    "%s: should_exit=True — RSI(%d)=%.2f > %.1f",
                    symbol,
                    self._rsi_period,
                    last_rsi,
                    self._rsi_exit,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_5min_series(self, symbol: str, bar_store) -> Optional[pd.DataFrame]:
        """Construct a synthetic 5-minute OHLCV DataFrame from 1-minute bars.

        Fetches the last _FETCH_1MIN_BARS (80) 1-minute bars from bar_store
        and groups them in non-overlapping windows of 5 consecutive bars.
        Each synthetic 5-min bar has:
            open  = first bar's open
            high  = max of 5 highs
            low   = min of 5 lows
            close = last bar's close
            volume = sum of 5 volumes

        Returns None if fewer than _MIN_1MIN_BARS_FOR_HELPER (50) 1-minute
        bars are available, which prevents RSI computation on insufficient data.

        Args:
            symbol: Ticker symbol to build the series for.
            bar_store: BarStore instance providing access to raw 1-min bars.

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns [open, high, low,
                close, volume], indexed 0..N-1, oldest bar first. Returns None
                if insufficient 1-min bars are available.

        Example:
            >>> df = layer._build_5min_series("AAPL", store)
            >>> df.columns.tolist()
            ['open', 'high', 'low', 'close', 'volume']
            >>> len(df)  # floor(80 / 5) = 16 when 80 bars available
            16
        """
        one_min_df = bar_store.get_bars(symbol, "1Min", _FETCH_1MIN_BARS)
        if one_min_df is None or len(one_min_df) < _MIN_1MIN_BARS_FOR_HELPER:
            logger.debug(
                "%s: only %d 1-min bars available, need %d for 5-min series",
                symbol,
                len(one_min_df) if one_min_df is not None else 0,
                _MIN_1MIN_BARS_FOR_HELPER,
            )
            return None

        # Reset index so we can slice cleanly by position
        df = one_min_df.reset_index(drop=True)
        n_bars = len(df)
        n_groups = n_bars // 5  # floor division: discard remainder

        if n_groups < 2:
            # Not enough complete 5-min groups to do anything useful
            return None

        synthetic_rows = []
        for g in range(n_groups):
            start = g * 5
            end = start + 5
            group = df.iloc[start:end]
            synthetic_rows.append(
                {
                    "open": float(group["open"].iloc[0]),
                    "high": float(group["high"].max()),
                    "low": float(group["low"].min()),
                    "close": float(group["close"].iloc[-1]),
                    "volume": float(group["volume"].sum()),
                }
            )

        result = pd.DataFrame(synthetic_rows)
        logger.debug(
            "%s: built %d synthetic 5-min bars from %d 1-min bars",
            symbol,
            len(result),
            n_bars,
        )
        return result

    def _evaluate_entry(
        self,
        symbol: str,
        bar: dict,
        bar_store,
    ) -> List[SignalResult]:
        """Evaluate all entry conditions and return a BUY signal if met.

        Entry requires ALL of:
            1. RSI(7) on synthetic 5-min closes < 25 (extreme oversold)
            2. Current 1-min close > previous 1-min close (reversal confirmation)
            3. Session bar count > 10 (not in the first 10 bars of session)
            4. No open position for this symbol from this layer

        Stop is set at entry_price - 0.5 * ATR(14, synthetic 5-min bars).
        Confidence = min((25 - rsi_value) / 25.0, 1.0), ensuring that a
        deeper oversold reading yields a higher confidence score.

        Args:
            symbol: Ticker symbol.
            bar: Current 1-minute bar dict.
            bar_store: BarStore instance.

        Returns:
            List[SignalResult]: Single-element list with BUY signal if all
                entry conditions are met, otherwise empty list.

        Example:
            >>> signals = layer._evaluate_entry("AAPL", bar, store)
            >>> if signals:
            ...     print(signals[0].signal, signals[0].confidence)
            BUY 0.4
        """
        five_min_df = self._build_5min_series(symbol, bar_store)
        if five_min_df is None or len(five_min_df) < self._rsi_period + 1:
            logger.debug(
                "%s: insufficient 5-min bars for RSI(%d) entry check",
                symbol,
                self._rsi_period,
            )
            return []

        rsi_arr = rsi(five_min_df["close"], self._rsi_period)
        current_rsi = rsi_arr[-1]

        if np.isnan(current_rsi):
            logger.debug("%s: RSI is NaN, skipping entry", symbol)
            return []

        # Condition 1: extreme oversold
        if current_rsi >= self._rsi_entry:
            logger.debug(
                "%s: RSI(%d)=%.2f not below threshold %.1f, no entry",
                symbol,
                self._rsi_period,
                current_rsi,
                self._rsi_entry,
            )
            return []

        # Condition 2: 1-min reversal confirmation — current close > previous close
        one_min_df = bar_store.get_bars(symbol, "1Min", 3)
        if one_min_df is None or len(one_min_df) < 2:
            logger.debug("%s: insufficient 1-min bars for reversal confirmation", symbol)
            return []

        current_close = float(bar.get("close", 0.0))
        prev_close = float(one_min_df["close"].iloc[-2])

        if current_close <= prev_close:
            logger.debug(
                "%s: no reversal — close=%.4f <= prev_close=%.4f",
                symbol,
                current_close,
                prev_close,
            )
            return []

        # Compute ATR on synthetic 5-min series for stop placement
        atr_value = 0.0
        if len(five_min_df) >= self._atr_period + 1:
            atr_arr = atr(
                five_min_df["high"],
                five_min_df["low"],
                five_min_df["close"],
                self._atr_period,
            )
            last_atr = atr_arr[-1]
            if not np.isnan(last_atr):
                atr_value = float(last_atr)

        stop_price = (
            current_close - self._stop_atr_multiplier * atr_value
            if atr_value > 0.0
            else current_close * (1.0 - self._stop_min_pct)  # fallback: stop_min_pct below close
        )

        # Confidence: deeper oversold = higher confidence
        confidence = min((self._rsi_entry - current_rsi) / self._rsi_entry, 1.0)
        confidence = round(max(confidence, 0.0), 4)

        entry_time = bar.get("timestamp")
        if entry_time is None:
            entry_time = datetime.now(timezone.utc)
        elif hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        self._open_positions[symbol] = {
            "entry_price": current_close,
            "entry_time": entry_time,
            "stop_price": stop_price,
            "bars_held": 0,
        }

        logger.info(
            "%s: BUY signal — RSI(%d)=%.2f < %.1f, close=%.4f > prev=%.4f, "
            "stop=%.4f, atr=%.4f, confidence=%.4f",
            symbol,
            self._rsi_period,
            current_rsi,
            self._rsi_entry,
            current_close,
            prev_close,
            stop_price,
            atr_value,
            confidence,
        )

        return [
            SignalResult(
                symbol=symbol,
                signal="BUY",
                confidence=confidence,
                signal_price=current_close,
                layer_name=self.layer_name,
                stop_price=round(stop_price, 6),
                metadata={
                    f"rsi_{self._rsi_period}": round(float(current_rsi), 4),
                    f"atr_{self._atr_period}_5min": round(atr_value, 6),
                    "prev_1min_close": round(prev_close, 4),
                    "entry_bar_close": round(current_close, 4),
                    "synthetic_5min_bars_used": len(five_min_df),
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

        Exit fires when ANY of the following are true:
            1. RSI(7) on current synthetic 5-min closes > 55 (recovered to neutral)
            2. Current 1-min close < stop_price (stop-loss breach)
            3. bars_held >= 10 (10 synthetic 5-min bars ~ 50 minutes)
            4. Minutes held >= 120 (hard 2-hour safety ceiling)

        Args:
            symbol: Ticker symbol with an open position.
            bar: Current 1-minute bar dict.
            bar_store: BarStore instance.

        Returns:
            Optional[SignalResult]: EXIT signal if any exit criterion is met,
                otherwise None (continue holding).

        Example:
            >>> signal = layer._evaluate_exit("AAPL", bar, store)
            >>> signal.signal if signal else "HOLD"
            'HOLD'
        """
        pos = self._open_positions[symbol]
        current_close = float(bar.get("close", 0.0))
        stop_price = pos["stop_price"]
        bars_held = pos["bars_held"]
        entry_time = pos["entry_time"]
        entry_price = pos["entry_price"]

        exit_reason: Optional[str] = None

        # Condition 2: stop-loss breach
        if stop_price > 0.0 and current_close < stop_price:
            exit_reason = f"stop_breach close={current_close:.4f} < stop={stop_price:.4f}"

        # Condition 3: bars-held time stop
        elif bars_held >= self._max_bars_held:
            exit_reason = f"bars_held={bars_held} >= {self._max_bars_held}"

        else:
            # Condition 4: hard minutes ceiling
            if entry_time is not None:
                now_utc = datetime.now(timezone.utc)
                if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                minutes_elapsed = (now_utc - entry_time).total_seconds() / 60.0
                if minutes_elapsed >= self._max_minutes:
                    exit_reason = (
                        f"minutes_held={minutes_elapsed:.1f} >= {self._max_minutes:.0f}"
                    )

            # Condition 1: RSI recovery exit (only if not already triggering)
            if exit_reason is None:
                five_min_df = self._build_5min_series(symbol, bar_store)
                if five_min_df is not None and len(five_min_df) >= self._rsi_period + 1:
                    rsi_arr = rsi(five_min_df["close"], self._rsi_period)
                    last_rsi = rsi_arr[-1]
                    if not np.isnan(last_rsi) and last_rsi > self._rsi_exit:
                        exit_reason = (
                            f"rsi_recovered RSI({self._rsi_period})={last_rsi:.2f} > {self._rsi_exit}"
                        )

        if exit_reason is None:
            return None

        pnl_pct = ((current_close - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0

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
                "entry_price": round(entry_price, 4),
                "exit_price": round(current_close, 4),
                "pnl_pct": round(pnl_pct, 4),
            },
        )

    def _adopt_external_position(self, symbol: str, position_data: dict) -> None:
        """Reconcile an externally supplied position into internal state.

        Called when evaluate_bar() receives an open_position from the caller
        that has not yet been registered in _open_positions. This can happen
        after a restart or when the position manager rehydrates state.

        Args:
            symbol: Ticker symbol of the position to adopt.
            position_data: Dict with entry_price, entry_time, stop_price,
                bars_held, layer_name keys.

        Example:
            >>> layer._adopt_external_position("AAPL", {
            ...     "entry_price": 150.0, "entry_time": datetime(...),
            ...     "stop_price": 149.5, "bars_held": 3,
            ...     "layer_name": "L3_RSI_SCALP"})
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
        }
        logger.debug(
            "%s: adopted external position entry=%.4f stop=%.4f bars_held=%d",
            symbol,
            self._open_positions[symbol]["entry_price"],
            self._open_positions[symbol]["stop_price"],
            self._open_positions[symbol]["bars_held"],
        )
