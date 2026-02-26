"""VWAP momentum breakout strategy (Layer 2) on 5-minute bars.

Secondary alpha layer that identifies momentum breakouts using
VWAP deviation, ATR breakout levels, and volume confirmation.
Complements the mean reversion layer by capturing trending moves.
"""

import asyncio
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.data import indicators
from src.data.market_data import fetch_bars_bulk
from src.strategy.base_strategy import BaseStrategy, SignalResult
from src.strategy.regime_filter import RegimeFilter
from src.utils.logger import get_logger

logger = get_logger("strategy.momentum_breakout")


def _load_config() -> dict:
    """Load strategy configuration from config.yml.

    Returns:
        dict: Full strategy configuration parameters.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("strategy", {})


class MomentumBreakoutStrategy(BaseStrategy):
    """VWAP momentum breakout on 5-minute bars with regime overlay.

    Entry logic:
        Price > VWAP + (ATR * breakout_multiplier)
        AND volume > (volume_breakout_ratio * 20-bar average volume)
        AND regime is not BEAR
        -> BUY

    Exit logic:
        Price < VWAP (dropped below session VWAP)
        OR held > MAX_HOLDING_DAYS
        -> EXIT

    Structural constraints:
        - No shorts (only BUY signals)
        - PDT check: EXIT requires age >= MIN_HOLDING_DAYS
        - Signals pass through regime filter

    Attributes:
        config: Strategy parameters from config.yml.
        regime_filter: RegimeFilter instance.
        bars_cache: Cached 5-min bar data from latest fetch.
    """

    def __init__(self, regime_filter: Optional[RegimeFilter] = None):
        """Initialize the momentum breakout strategy.

        Args:
            regime_filter: Optional pre-configured RegimeFilter.
        """
        self.config = _load_config()
        self.regime_filter = regime_filter or RegimeFilter()
        self.bars_cache: Dict[str, pd.DataFrame] = {}

    def generate_signals(self, universe: List[str]) -> List[SignalResult]:
        """Generate momentum breakout signals for all symbols.

        Fetches 5-minute bars, computes VWAP, ATR, and volume ratio,
        then applies breakout criteria with regime filtering.

        Args:
            universe: List of ticker symbols to evaluate.

        Returns:
            List[SignalResult]: List of actionable signals.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self._generate_signals_async(universe)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._generate_signals_async(universe)
                )
        except RuntimeError:
            return asyncio.run(self._generate_signals_async(universe))

    async def _generate_signals_async(
        self, universe: List[str]
    ) -> List[SignalResult]:
        """Async implementation of signal generation.

        Args:
            universe: List of ticker symbols.

        Returns:
            List[SignalResult]: Generated signals filtered by regime.
        """
        lookback = self.config.get("layer2_lookback_bars", 78)
        timeframe = self.config.get("layer2_timeframe", "5Min")
        self.bars_cache = await fetch_bars_bulk(universe, timeframe, lookback)

        raw_signals = []
        for symbol in universe:
            signal = self._evaluate_symbol(symbol)
            if signal is not None:
                raw_signals.append(signal)

        self.regime_filter.detect_regime()
        filtered_signals = self.regime_filter.filter_signals(raw_signals)

        buy_count = sum(1 for s in filtered_signals if s.signal == "BUY")
        logger.info(
            "Momentum signals: %d total, %d BUY (regime=%s)",
            len(filtered_signals),
            buy_count,
            self.regime_filter.current_regime,
        )
        return filtered_signals

    def _evaluate_symbol(self, symbol: str) -> Optional[SignalResult]:
        """Evaluate a single symbol for momentum breakout entry.

        Args:
            symbol: Ticker symbol to evaluate.

        Returns:
            Optional[SignalResult]: BUY signal if breakout criteria met, else None.
        """
        df = self.bars_cache.get(symbol)
        if df is None or len(df) < 30:
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        current_price = float(close.iloc[-1])

        # Compute VWAP
        vwap_values = indicators.vwap(high, low, close, volume)
        current_vwap = float(vwap_values[-1])

        if np.isnan(current_vwap) or current_vwap == 0:
            return None

        # Compute ATR
        atr_period = self.config.get("atr_period", 14)
        atr_values = indicators.atr(high, low, close, min(atr_period, len(df) - 1))
        current_atr = float(atr_values[-1]) if not np.isnan(atr_values[-1]) else 0

        if current_atr == 0:
            return None

        # Breakout level = VWAP + (ATR * multiplier)
        breakout_mult = self.config.get("vwap_atr_breakout_multiplier", 1.0)
        breakout_level = current_vwap + (current_atr * breakout_mult)

        # Volume confirmation
        vol_ratio_threshold = self.config.get("volume_breakout_ratio", 1.5)
        vol_series = volume.astype(float)
        avg_volume = float(vol_series.tail(20).mean()) if len(vol_series) >= 20 else float(vol_series.mean())
        current_volume = float(vol_series.iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Check breakout conditions
        price_breakout = current_price > breakout_level
        volume_confirms = volume_ratio >= vol_ratio_threshold

        # RSI check - avoid overbought breakouts
        rsi_values = indicators.rsi(close, 14)
        current_rsi = float(rsi_values[-1]) if not np.isnan(rsi_values[-1]) else 50
        not_overbought = current_rsi < 80

        # Price slope for trend confirmation
        price_slope = indicators.slope(close, 5)

        metadata = {
            "vwap": round(current_vwap, 4),
            "atr": round(current_atr, 4),
            "breakout_level": round(breakout_level, 4),
            "volume_ratio": round(volume_ratio, 4),
            "rsi": round(current_rsi, 2),
            "price_slope": round(price_slope, 6),
            "strategy": self.config.get("layer2_name", "momentum_breakout_v1"),
            "layer": "layer2",
        }

        if price_breakout and volume_confirms and not_overbought:
            # Confidence based on how far above breakout and volume strength
            price_excess = (current_price - breakout_level) / current_atr
            vol_excess = volume_ratio / vol_ratio_threshold
            confidence = min((price_excess * 0.4 + vol_excess * 0.4 + 0.2), 1.0)
            confidence = max(confidence, 0.3)

            logger.info(
                "BUY breakout: %s price=$%.2f > breakout=$%.2f, "
                "vol_ratio=%.2f, RSI=%.1f, conf=%.2f",
                symbol,
                current_price,
                breakout_level,
                volume_ratio,
                current_rsi,
                confidence,
            )
            return SignalResult(
                symbol=symbol,
                signal="BUY",
                confidence=round(confidence, 4),
                signal_price=current_price,
                metadata=metadata,
            )

        return None

    def should_exit(self, symbol: str, position_data: dict) -> bool:
        """Determine if a momentum position should be exited.

        Exit criteria:
        1. Price < VWAP (momentum lost)
        2. Held > MAX_HOLDING_DAYS

        PDT protection: Never exits before MIN_HOLDING_DAYS.

        Args:
            symbol: Ticker symbol.
            position_data: Dict with days_held, entry_price, current_price.

        Returns:
            bool: True if position should be exited.
        """
        min_holding = self.config.get("min_holding_days", 3)
        max_holding = self.config.get("max_holding_days", 15)
        days_held = position_data.get("days_held", 0)

        if days_held < min_holding:
            return False

        if days_held >= max_holding:
            logger.info(
                "Time exit: %s held %d days (max=%d)", symbol, days_held, max_holding
            )
            return True

        df = self.bars_cache.get(symbol)
        if df is None or df.empty:
            return False

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        current_price = float(close.iloc[-1])

        vwap_values = indicators.vwap(high, low, close, volume)
        current_vwap = float(vwap_values[-1])

        if np.isnan(current_vwap):
            return False

        if current_price < current_vwap:
            logger.info(
                "VWAP exit: %s price=$%.2f < VWAP=$%.2f",
                symbol,
                current_price,
                current_vwap,
            )
            return True

        return False

    def get_exit_signals(self, open_positions: List[dict]) -> List[SignalResult]:
        """Generate exit signals for open momentum positions.

        Args:
            open_positions: List of position dicts.

        Returns:
            List[SignalResult]: EXIT signals for positions to close.
        """
        exit_signals = []
        for pos in open_positions:
            symbol = pos["symbol"]
            layer = pos.get("layer", "")
            if layer != "layer2":
                continue
            if self.should_exit(symbol, pos):
                current_price = pos.get("current_price", 0.0)
                exit_signals.append(
                    SignalResult(
                        symbol=symbol,
                        signal="EXIT",
                        confidence=1.0,
                        signal_price=current_price,
                        metadata={
                            "trade_id": pos.get("trade_id"),
                            "days_held": pos.get("days_held", 0),
                            "exit_type": "SIGNAL",
                            "layer": "layer2",
                        },
                    )
                )
        return exit_signals
