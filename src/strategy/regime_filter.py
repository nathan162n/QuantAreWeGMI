"""Market regime detection using SPY SMA and VIXY volatility proxy.

Simplified regime filter: reduces position size in BEAR conditions
instead of halting all trading. Only halts completely at extreme panic
(VIXY > 60, e.g., March 2020 levels).

Previous version blocked all entries in BEAR — this caused the system
to sit idle for extended periods. Now it reduces size to 50% instead,
keeping all four layers active (except Layer 2 ORB which is disabled in BEAR).
"""

from typing import Dict, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("strategy.regime_filter")

_REGIME_INSTANCE: Optional["RegimeFilter"] = None


class RegimeState:
    """Enum-like constants for VIXY-based regime states (Phase 5)."""
    HALT_ALL = "HALT_ALL"
    REDUCE_SIZE = "REDUCE_SIZE"
    NORMAL = "NORMAL"


class RegimeFilter:
    """Determines market regime and returns position size scalar.

    Regime rules:
        EXTREME: VIXY > 60
            → size_scalar = 0.0 (full halt)
        BEAR: SPY < 200-day SMA AND VIXY > 40
            → size_scalar = 0.50 (half size), Layer 2 disabled
        BULL: SPY >= 200-day SMA
            → size_scalar = 1.0
        NEUTRAL: All other conditions
            → size_scalar = 1.0

    Attributes:
        spy_sma_period: Period for SPY SMA filter. Default 200.
        vix_size_reduction_threshold: VIXY level triggering 50% size reduction. Default 40.
        vix_halt_threshold: VIXY level triggering full halt. Default 60.
        vix_size_reduction_scalar: Multiplier applied when in BEAR. Default 0.50.

    Example:
        >>> rf = RegimeFilter()
        >>> regime, scalar = rf.get_regime(spy_prices, 45.0)
        >>> regime
        'BEAR'
    """

    def __init__(
        self,
        spy_sma_period: int = 200,
        vix_size_reduction_threshold: float = 40.0,
        vix_halt_threshold: float = 60.0,
        vix_size_reduction_scalar: float = 0.50,
    ):
        """Initialize regime filter.

        Args:
            spy_sma_period: Period for SPY SMA. Default 200.
            vix_size_reduction_threshold: VIXY above this triggers size reduction. Default 40.
            vix_halt_threshold: VIXY above this halts all trading. Default 60.
            vix_size_reduction_scalar: Size multiplier in BEAR. Default 0.50.

        Example:
            >>> rf = RegimeFilter(spy_sma_period=200, vix_halt_threshold=60)
        """
        self.spy_sma_period = spy_sma_period
        self.vix_size_reduction_threshold = vix_size_reduction_threshold
        self.vix_halt_threshold = vix_halt_threshold
        self.vix_size_reduction_scalar = vix_size_reduction_scalar
        self._current_regime = "NEUTRAL"
        self._current_scalar = 1.0

    def get_regime(
        self,
        spy_prices,
        vixy_current_price: float,
    ) -> Tuple[str, float]:
        """Compute the current market regime and size scalar.

        Args:
            spy_prices: pd.Series or list of SPY close prices, most recent last.
                Must have at least spy_sma_period values for SMA comparison.
            vixy_current_price: Current price of VIXY (VIX proxy ETF).

        Returns:
            Tuple[str, float]: (regime, size_scalar)
                regime is one of: "BULL", "NEUTRAL", "BEAR", "EXTREME"
                size_scalar is 0.0, 0.5, or 1.0.

        Example:
            >>> import pandas as pd
            >>> spy = pd.Series([100.0] * 200 + [95.0])
            >>> regime, scalar = rf.get_regime(spy, 45.0)
            >>> (regime, scalar)
            ('BEAR', 0.5)
        """
        import pandas as pd

        # EXTREME panic — full halt regardless of SPY
        if vixy_current_price > self.vix_halt_threshold:
            self._current_regime = "EXTREME"
            self._current_scalar = 0.0
            logger.warning(
                "EXTREME regime: VIXY=%.2f > %.1f halt threshold. All entries halted.",
                vixy_current_price,
                self.vix_halt_threshold,
            )
            return "EXTREME", 0.0

        # Compute SPY 200-day SMA
        prices = pd.Series(spy_prices, dtype=float)
        if len(prices) < self.spy_sma_period:
            logger.debug(
                "Insufficient SPY data (%d < %d bars) — defaulting NEUTRAL",
                len(prices),
                self.spy_sma_period,
            )
            spy_above_sma = True
        else:
            sma_200 = float(prices.rolling(window=self.spy_sma_period).mean().iloc[-1])
            spy_current = float(prices.iloc[-1])
            spy_above_sma = spy_current >= sma_200
            logger.debug(
                "SPY: current=%.2f, SMA200=%.2f, above_sma=%s",
                spy_current,
                sma_200,
                spy_above_sma,
            )

        # BEAR: SPY below 200-SMA AND VIXY elevated
        if not spy_above_sma and vixy_current_price > self.vix_size_reduction_threshold:
            self._current_regime = "BEAR"
            self._current_scalar = self.vix_size_reduction_scalar
            logger.info(
                "BEAR regime: SPY below 200-SMA, VIXY=%.2f. Size reduced to %.0f%%.",
                vixy_current_price,
                self.vix_size_reduction_scalar * 100,
            )
            return "BEAR", self.vix_size_reduction_scalar

        # BULL: SPY above 200-SMA
        if spy_above_sma:
            self._current_regime = "BULL"
            self._current_scalar = 1.0
            logger.debug("BULL regime.")
            return "BULL", 1.0

        # NEUTRAL: SPY below 200-SMA but VIXY not elevated
        self._current_regime = "NEUTRAL"
        self._current_scalar = 1.0
        logger.debug(
            "NEUTRAL regime: SPY below SMA but VIXY=%.2f < %.1f.",
            vixy_current_price,
            self.vix_size_reduction_threshold,
        )
        return "NEUTRAL", 1.0

    @property
    def current_regime(self) -> str:
        """Return the last computed regime string.

        Returns:
            str: One of "BULL", "NEUTRAL", "BEAR", "EXTREME".

        Example:
            >>> rf.current_regime
            'BULL'
        """
        return self._current_regime

    @property
    def current_size_scalar(self) -> float:
        """Return the last computed size scalar.

        Returns:
            float: 0.0, 0.5, or 1.0.

        Example:
            >>> rf.current_size_scalar
            1.0
        """
        return self._current_scalar

    def is_layer2_enabled(self) -> bool:
        """Check if Layer 2 (ORB) is allowed in current regime.

        Layer 2 is disabled in BEAR and EXTREME regimes.

        Returns:
            bool: True if ORB entries are permitted.

        Example:
            >>> rf.is_layer2_enabled()
            False  # in BEAR regime
        """
        return self._current_regime not in ("BEAR", "EXTREME")

    def is_trading_halted(self) -> bool:
        """Check if all trading is halted (EXTREME regime only).

        Returns:
            bool: True only when VIXY > 60.

        Example:
            >>> rf.is_trading_halted()
            False
        """
        return self._current_regime == "EXTREME"

    @staticmethod
    def rank_by_zscore(symbols: list, bar_store) -> Dict[str, float]:
        """Rank symbols by their recent price z-score using last 20 1-min closes.

        Phase 4C: Used by BarDispatcher to filter signals by momentum rank.
        Symbols with high z-scores are trending; low/negative z-scores suggest
        mean-reversion candidates.

        Args:
            symbols: List of ticker symbols to rank.
            bar_store: BarStore instance providing get_bars().

        Returns:
            Dict[str, float]: Map of symbol → z-score. Symbols with
                insufficient data are excluded from the result.

        Example:
            >>> scores = RegimeFilter.rank_by_zscore(["AAPL", "MSFT"], store)
            >>> scores["AAPL"]
            1.23
        """
        import numpy as np

        scores: Dict[str, float] = {}
        for sym in symbols:
            try:
                df = bar_store.get_bars(sym, "1Min", 20)
                if df is None or len(df) < 5:
                    continue
                closes = df["close"].values.astype(float)
                mean = float(np.mean(closes))
                std = float(np.std(closes))
                if std == 0:
                    scores[sym] = 0.0
                else:
                    scores[sym] = round((closes[-1] - mean) / std, 4)
            except Exception:
                continue
        return scores

    def get_vixy_regime(self, vixy_price: float, config: dict) -> str:
        """Determine the VIXY-specific regime gate for BUY signal filtering.

        Phase 5: Uses dedicated VIXY thresholds separate from the existing
        get_regime() method, which requires SPY data. This allows a fast
        real-time gate using only the VIXY price.

        Thresholds (from config.regime):
            vixy_hard_halt_threshold (default 35.0): HALT_ALL
            vixy_size_reduction_threshold (default 25.0): REDUCE_SIZE
            Otherwise: NORMAL

        Args:
            vixy_price: Current VIXY price.
            config: Full config dict with 'regime' subsection.

        Returns:
            str: One of RegimeState.HALT_ALL, RegimeState.REDUCE_SIZE,
                or RegimeState.NORMAL.

        Example:
            >>> rf.get_vixy_regime(38.0, config)
            'HALT_ALL'
        """
        regime_cfg = config.get("regime", {})
        halt_threshold = float(regime_cfg.get("vixy_hard_halt_threshold", 35.0))
        reduce_threshold = float(regime_cfg.get("vixy_size_reduction_threshold", 25.0))

        if vixy_price >= halt_threshold:
            logger.warning(
                "VIXY regime=HALT_ALL: price=%.2f >= halt_threshold=%.1f",
                vixy_price, halt_threshold,
            )
            return RegimeState.HALT_ALL

        if vixy_price >= reduce_threshold:
            logger.info(
                "VIXY regime=REDUCE_SIZE: price=%.2f >= reduce_threshold=%.1f",
                vixy_price, reduce_threshold,
            )
            return RegimeState.REDUCE_SIZE

        return RegimeState.NORMAL


def get_regime_filter() -> RegimeFilter:
    """Get or create the singleton RegimeFilter instance.

    Returns:
        RegimeFilter: The singleton instance.

    Example:
        >>> rf = get_regime_filter()
    """
    global _REGIME_INSTANCE
    if _REGIME_INSTANCE is None:
        _REGIME_INSTANCE = RegimeFilter()
    return _REGIME_INSTANCE
