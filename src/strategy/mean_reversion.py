"""Volatility-adjusted mean reversion strategy with regime overlay.

Primary alpha engine that identifies oversold S&P 500 equities using
z-score and RSI, with regime filtering to avoid trending markets.
Structurally prevents shorts and enforces minimum holding periods.
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.data import indicators
from src.data.market_data import fetch_bars_bulk
from src.strategy.base_strategy import BaseStrategy, SignalResult
from src.strategy.regime_filter import RegimeFilter
from src.utils.logger import get_logger

logger = get_logger("strategy.mean_reversion")


def _load_config() -> dict:
    """Load strategy configuration from config.yml.

    Returns:
        dict: Strategy configuration parameters.

    Example:
        >>> config = _load_config()
        >>> config["zscore_entry_threshold"]
        -2.0
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("strategy", {})


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using z-score and RSI with regime overlay.

    Legacy daily strategy — for bar-by-bar trading use VWAPMeanReversionStrategy.

    Entry logic:
        Z-score < -2.0 AND Z-score > -4.0 (not in freefall)
        AND RSI(14) < 35 -> BUY

    Exit logic:
        Z-score > -0.5 OR RSI(14) > 65 OR held > MAX_HOLDING_DAYS -> EXIT

    Structural constraints:
        - No shorts (only BUY signals are generated)
        - PDT check: EXIT signals require position age >= MIN_HOLDING_DAYS
        - All signals pass through regime filter before returning

    Attributes:
        config: Strategy parameters from config.yml.
        regime_filter: RegimeFilter instance for market regime detection.
        bars_cache: Cached bar data from the latest fetch.
    """

    def __init__(self, regime_filter: Optional[RegimeFilter] = None):
        """Initialize the mean reversion strategy.

        Args:
            regime_filter: Optional pre-configured RegimeFilter. Creates one
                if not provided.

        Example:
            >>> strategy = MeanReversionStrategy()
        """
        self.config = _load_config()
        self.regime_filter = regime_filter or RegimeFilter()
        self.bars_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # BaseStrategy abstract method implementations (legacy stubs)
    # ------------------------------------------------------------------

    @property
    def layer_name(self) -> str:
        """Return the layer name for this legacy strategy."""
        return self.config.get("layer1_name", "MEAN_REVERSION_V1")

    def evaluate_bar(self, symbol: str, bar: dict, bar_store, open_position=None):
        """Not implemented for legacy strategy — use VWAPMeanReversionStrategy."""
        return []

    def generate_signals(self, universe: List[str]) -> List[SignalResult]:
        """Generate mean reversion signals for all symbols in universe.

        Fetches 60 days of OHLCV data, computes z-scores and RSI,
        and applies entry criteria. All BUY signals are filtered
        through the regime filter before returning.

        Args:
            universe: List of ticker symbols to evaluate.

        Returns:
            List[SignalResult]: List of actionable signals (BUY or HOLD).
                EXIT signals are generated separately via should_exit().

        Example:
            >>> signals = strategy.generate_signals(["AAPL", "MSFT"])
            >>> [s.signal for s in signals if s.signal == "BUY"]
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

        Example:
            >>> signals = await strategy._generate_signals_async(["AAPL"])
        """
        lookback = self.config.get("layer1_lookback_bars", 60)
        timeframe = self.config.get("layer1_timeframe", "1Min")
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
            "Generated signals: %d total, %d BUY (regime=%s)",
            len(filtered_signals),
            buy_count,
            self.regime_filter.current_regime,
        )
        return filtered_signals

    def _evaluate_symbol(self, symbol: str) -> Optional[SignalResult]:
        """Evaluate a single symbol for mean reversion entry.

        Args:
            symbol: Ticker symbol to evaluate.

        Returns:
            Optional[SignalResult]: BUY signal if criteria met, HOLD otherwise.
                Returns None if insufficient data.

        Example:
            >>> signal = strategy._evaluate_symbol("AAPL")
            >>> signal.signal in ("BUY", "HOLD") if signal else True
        """
        df = self.bars_cache.get(symbol)
        if df is None or len(df) < self.config.get("layer1_lookback_bars", 60) * 0.5:
            return None

        close = df["close"]
        current_price = float(close.iloc[-1])

        zscore_period = self.config.get("zscore_period", 20)
        rsi_period = self.config.get("rsi_period", 14)

        z_scores = indicators.rolling_zscore(close, zscore_period)
        rsi_values = indicators.rsi(close, rsi_period)

        current_z = z_scores[-1]
        current_rsi = rsi_values[-1]

        if np.isnan(current_z) or np.isnan(current_rsi):
            return None

        entry_threshold = self.config.get("zscore_entry_threshold", -2.0)
        freefall_threshold = self.config.get("zscore_entry_freefall", -4.0)
        rsi_oversold = self.config.get("rsi_oversold", 35)

        is_oversold = current_z < entry_threshold
        not_freefall = current_z > freefall_threshold
        rsi_confirms = current_rsi < rsi_oversold

        metadata = {
            "z_score": round(float(current_z), 4),
            "rsi": round(float(current_rsi), 2),
            "strategy": self.config.get("layer1_name", "mean_reversion_v1"),
            "layer": "layer1",
        }

        if is_oversold and not_freefall and rsi_confirms:
            confidence = min(abs(float(current_z)) / 4.0, 1.0)
            logger.info(
                "BUY signal: %s z=%.2f, RSI=%.1f, confidence=%.2f",
                symbol,
                current_z,
                current_rsi,
                confidence,
            )
            return SignalResult(
                symbol=symbol,
                signal="BUY",
                confidence=round(confidence, 4),
                signal_price=current_price,
                layer_name=self.layer_name,
                metadata=metadata,
            )

        return SignalResult(
            symbol=symbol,
            signal="HOLD",
            confidence=0.0,
            signal_price=current_price,
            layer_name=self.layer_name,
            metadata=metadata,
        )

    def should_exit(self, symbol: str, position_data: dict) -> bool:
        """Determine if an open position should be exited.

        Exit criteria:
        1. Z-score > -0.5 (mean reverted)
        2. RSI(14) > 65 (overbought)
        3. Held > MAX_HOLDING_DAYS (time-based exit)

        PDT protection: Never returns True if position held fewer
        than MIN_HOLDING_DAYS trading days.

        Args:
            symbol: Ticker symbol of the open position.
            position_data: Dictionary containing:
                - days_held (int): Number of trading days held
                - entry_price (float): Entry price
                - current_price (float): Current price

        Returns:
            bool: True if the position should be exited, False otherwise.

        Example:
            >>> strategy.should_exit("AAPL", {"days_held": 5,
            ...     "entry_price": 150.0, "current_price": 155.0})
            True
        """
        min_holding = self.config.get("min_holding_days", 3)
        max_holding = self.config.get("max_holding_days", 15)
        days_held = position_data.get("days_held", 0)

        if days_held < min_holding:
            logger.debug(
                "PDT protection: %s held %d/%d days - cannot exit",
                symbol,
                days_held,
                min_holding,
            )
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
        zscore_period = self.config.get("zscore_period", 20)
        rsi_period = self.config.get("rsi_period", 14)

        z_scores = indicators.rolling_zscore(close, zscore_period)
        rsi_values = indicators.rsi(close, rsi_period)

        current_z = z_scores[-1]
        current_rsi = rsi_values[-1]

        if np.isnan(current_z) or np.isnan(current_rsi):
            return False

        exit_z_threshold = self.config.get("zscore_exit_threshold", -0.5)
        rsi_overbought = self.config.get("rsi_overbought", 65)

        if float(current_z) > exit_z_threshold:
            logger.info(
                "Z-score exit: %s z=%.2f > %.1f", symbol, current_z, exit_z_threshold
            )
            return True

        if float(current_rsi) > rsi_overbought:
            logger.info(
                "RSI exit: %s RSI=%.1f > %d", symbol, current_rsi, rsi_overbought
            )
            return True

        return False

    def get_exit_signals(
        self, open_positions: List[dict]
    ) -> List[SignalResult]:
        """Generate exit signals for all open positions.

        Args:
            open_positions: List of position dicts with keys:
                symbol, days_held, entry_price, current_price, trade_id.

        Returns:
            List[SignalResult]: EXIT signals for positions that should be closed.

        Example:
            >>> exits = strategy.get_exit_signals([
            ...     {"symbol": "AAPL", "days_held": 5, "entry_price": 150.0,
            ...      "current_price": 155.0, "trade_id": 1}])
        """
        exit_signals = []
        for pos in open_positions:
            symbol = pos["symbol"]
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
                        },
                    )
                )
        return exit_signals
