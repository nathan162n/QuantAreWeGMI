"""Abstract base class for trading strategy layers.

Defines the interface that all four strategy layers must follow.
The BarDispatcher calls evaluate_bar() on each layer for every
incoming WebSocket bar event.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SignalResult:
    """Trading signal produced by a strategy layer for a single symbol.

    Attributes:
        symbol: Ticker symbol the signal applies to.
        signal: Signal type — 'BUY', 'EXIT', or 'HOLD'.
        confidence: Signal confidence from 0.0 to 1.0.
        signal_price: Price at the time the signal was generated.
        layer_name: Which strategy layer produced this signal.
        stop_price: Computed stop-loss price for this signal.
        metadata: Additional signal details stored for analysis.

    Example:
        >>> signal = SignalResult(
        ...     symbol="AAPL", signal="BUY", confidence=0.75,
        ...     signal_price=150.0, layer_name="L1_VWAP_MR",
        ...     stop_price=149.0, metadata={"vwap": 151.0, "rsi": 38})
    """

    symbol: str
    signal: str  # 'BUY', 'EXIT', 'HOLD'
    confidence: float
    signal_price: float
    layer_name: str
    stop_price: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """5-bar forward forecast attached to every BUY signal.

    Attributes:
        direction: Human-readable forecast direction.
        score: Integer score from -100 to +100 (positive = bullish).
        confidence_pct: Score magnitude as a percentage (0-100).
        estimated_low: Lower price estimate in 5 bars.
        estimated_high: Upper price estimate in 5 bars.
        key_driver: Plain English dominant factor description.
        bars_horizon: Always 5 for the 5-bar forecast.
        disclaimer: Standard forecast disclaimer.

    Example:
        >>> pred = PredictionResult(
        ...     direction="↑ BULLISH", score=65, confidence_pct=65.0,
        ...     estimated_low=149.0, estimated_high=153.0,
        ...     key_driver="Price below VWAP with RSI recovering",
        ...     bars_horizon=5,
        ...     disclaimer="Statistical estimate...")
    """

    direction: str
    score: int
    confidence_pct: float
    estimated_low: float
    estimated_high: float
    key_driver: str
    bars_horizon: int = 5
    disclaimer: str = (
        "Statistical estimate based on historical pattern similarity. "
        "Not a guaranteed prediction. Past patterns do not guarantee future price movements."
    )


class BaseStrategy(ABC):
    """Abstract base class for all four strategy layers.

    Each layer receives a bar and the current BarStore and must decide
    whether to generate a BUY signal, an EXIT signal for an open position,
    or return HOLD. Layers are stateless per-bar — any intraday state
    (e.g., opening range, bar count for time stops) must be maintained
    internally in the layer instance.

    Layer naming convention:
        L1_VWAP_MR  — VWAP Mean Reversion
        L2_ORB      — Opening Range Breakout
        L3_RSI_SCALP — RSI(7) Reversal Scalp
        L4_VOL_SURGE — Volume Surge Momentum

    Example:
        >>> class MyLayer(BaseStrategy):
        ...     @property
        ...     def layer_name(self): return "MY_LAYER"
        ...     def evaluate_bar(self, symbol, bar, bar_store,
        ...                      open_position): return []
        ...     def should_exit(self, symbol, bar, bar_store,
        ...                     position_data): return False
    """

    @property
    @abstractmethod
    def layer_name(self) -> str:
        """Unique identifier for this strategy layer.

        Returns:
            str: Layer name (e.g., "L1_VWAP_MR").

        Example:
            >>> layer.layer_name
            'L1_VWAP_MR'
        """
        ...

    @abstractmethod
    def evaluate_bar(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        open_position: Optional[dict] = None,
    ) -> List[SignalResult]:
        """Evaluate a bar and return signals.

        Called by BarDispatcher on every incoming bar (subject to cadence
        rules set by the dispatcher). Returns a list of SignalResult objects:
        - Empty list: no action
        - SignalResult(signal='BUY'): open a new position
        - SignalResult(signal='EXIT'): close the open_position
        - SignalResult(signal='HOLD'): explicit hold (rare, usually just return [])

        Args:
            symbol: Ticker symbol.
            bar: Current bar dict with open, high, low, close, volume, timestamp.
            bar_store: BarStore instance with historical bar data.
            open_position: If not None, dict with entry_price, entry_time,
                qty, stop_price, bars_held, layer_name. The layer should
                check if its own open position needs to exit.

        Returns:
            List[SignalResult]: Signals to process. Usually 0 or 1 element.

        Example:
            >>> signals = layer.evaluate_bar("AAPL", bar, store, None)
            >>> if signals and signals[0].signal == "BUY":
            ...     order_manager.submit_entry(signals[0])
        """
        ...

    @abstractmethod
    def should_exit(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        position_data: dict,
    ) -> bool:
        """Determine if an open position should be exited.

        Called directly by StreamEngine on stop-loss checks and by
        EODManager at 15:30 ET. Distinct from evaluate_bar() which
        handles the primary signal evaluation loop.

        Args:
            symbol: Ticker symbol of the open position.
            bar: Current bar dict.
            bar_store: BarStore instance.
            position_data: Dict with entry_price, entry_time, qty,
                stop_price, bars_held, layer_name.

        Returns:
            bool: True if the position should be closed.

        Example:
            >>> should_close = layer.should_exit(
            ...     "AAPL", current_bar, store,
            ...     {"entry_price": 150.0, "stop_price": 149.0,
            ...      "bars_held": 15, "entry_time": datetime(...)})
            True
        """
        ...

    # Legacy compatibility — kept for tests that use generate_signals
    def generate_signals(self, universe: List[str]) -> List[SignalResult]:
        """Legacy interface — not used in the new WebSocket-driven system.

        Returns:
            List[SignalResult]: Always empty in the new architecture.

        Example:
            >>> layer.generate_signals(["AAPL"])
            []
        """
        return []
