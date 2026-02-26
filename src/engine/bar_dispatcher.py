"""Bar dispatcher — routes incoming bars to strategy layers with deduplication.

Manages which strategy layers are called on each bar and enforces the
global deduplication rules: no duplicate entries per symbol per layer,
no entries when at max positions.

Layer routing:
    - Layer 1 (VWAP MR):    fires on every 1-min bar
    - Layer 2 (ORB):        fires on 1-min bars only when opening_range is available
    - Layer 3 (RSI Scalp):  fires on every 3rd 1-min bar (bar_count % cadence == 0)
    - Layer 4 (Vol Surge):  fires on every 1-min bar

Deduplication rules:
    - If a position is already open for a symbol in Layer X, do not generate
      a new entry signal for that symbol from Layer X
    - If total open positions >= max_positions (3), do not generate any entry signals
    - If an open order already exists for a symbol, do not generate new orders

Example:
    >>> dispatcher = BarDispatcher(layers=[l1, l2, l3, l4], max_positions=3)
    >>> signals = dispatcher.dispatch(
    ...     "AAPL", bar, bar_store, open_positions, open_orders
    ... )
    >>> for sig in signals:
    ...     print(sig.signal, sig.layer_name)
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from src.strategy.base_strategy import BaseStrategy, SignalResult
from src.data.bar_store import BarStore
from src.utils.logger import get_logger

logger = get_logger("engine.bar_dispatcher")

# Layer name constants matching convention used across the system
_LAYER_NAMES = {
    0: "L1_VWAP_MR",
    1: "L2_ORB",
    2: "L3_RSI_SCALP",
    3: "L4_VOL_SURGE",
}


def _load_config() -> dict:
    """Load full config.yml. Returns {} on error."""
    try:
        import os
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
        )
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class BarDispatcher:
    """Routes bars to strategy layers and enforces deduplication.

    Deduplication rules:
      - If a position is already open for a symbol in Layer X, do not generate
        a new entry signal for that symbol from Layer X.
      - If total open positions >= max_positions (3), do not generate any entry
        signals (only EXIT signals are passed through).
      - If an open order already exists for a symbol, do not generate new orders.

    Layer routing:
      - Layer 1 (VWAP MR):    fires on every 1-min bar.
      - Layer 2 (ORB):        fires on 1-min bars only when opening_range is
                              available AND layer2_enabled is True.
      - Layer 3 (RSI Scalp):  fires on every 3rd 1-min bar
                              (bar_count % l3_cadence == 0).
      - Layer 4 (Vol Surge):  fires on every 1-min bar.

    Attributes:
        layers: List of BaseStrategy instances (all four layers).
        max_positions: Maximum simultaneous open positions.
        l3_cadence: Bar cadence for Layer 3 evaluation (default 3).

    Example:
        >>> dispatcher = BarDispatcher(layers=[l1, l2, l3, l4], max_positions=3)
        >>> signals = dispatcher.dispatch(
        ...     symbol="AAPL",
        ...     bar={"open": 150.0, "close": 151.0, ...},
        ...     bar_store=store,
        ...     open_positions={"AAPL_L1_VWAP_MR": {...}},
        ...     open_orders=set(),
        ... )
    """

    def __init__(
        self,
        layers: List[BaseStrategy],
        max_positions: int = 3,
        l3_cadence: int = 3,
        config: Optional[dict] = None,
    ):
        """Initialize the BarDispatcher.

        Args:
            layers: List of BaseStrategy instances. All four layers should be
                provided, ordered as [L1, L2, L3, L4].
            max_positions: Maximum simultaneous open positions. Default 3.
            l3_cadence: Bar cadence for Layer 3. Layer 3 only fires when
                bar_count % l3_cadence == 0. Default 3.
            config: Optional pre-loaded config dict. Loaded from config.yml
                if not provided.

        Example:
            >>> dispatcher = BarDispatcher(
            ...     layers=[vwap_mr, orb, rsi_scalp, vol_surge],
            ...     max_positions=3,
            ...     l3_cadence=3,
            ... )
        """
        self.layers = layers
        self.max_positions = max_positions
        self.l3_cadence = l3_cadence
        self._config: dict = config if config is not None else _load_config()

        # Bug 5: stop-out cooldown tracker — maps symbol → UTC datetime of last stop
        self._last_stop_out: Dict[str, datetime] = {}

    def dispatch(
        self,
        symbol: str,
        bar: dict,
        bar_store: BarStore,
        open_positions: Dict[str, dict],
        open_orders: Set[str],
        layer2_enabled: bool = True,
        vixy_price: Optional[float] = None,
    ) -> List[SignalResult]:
        """Evaluate all applicable layers for this bar and return signals.

        Iterates over all four layers, applying cadence and deduplication rules.
        EXIT signals always pass through. BUY signals are filtered by dedup rules,
        stop-out cooldown, confidence gate, and VIXY regime gate.

        Args:
            symbol: Ticker symbol that received the bar.
            bar: Bar dict with open, high, low, close, volume, timestamp.
            bar_store: BarStore instance with history.
            open_positions: Dict[symbol_layerkey, position_dict] — ALL open
                positions. Key format: "{symbol}_{layer_name}" e.g.
                "AAPL_L1_VWAP_MR".
            open_orders: Set of symbols that have pending open orders.
            layer2_enabled: Whether Layer 2 (ORB) is permitted (regime gate).
                Default True.
            vixy_price: Optional current VIXY price for Phase 5 regime gate.
                If provided and above halt threshold, all BUY signals are blocked.

        Returns:
            List[SignalResult]: Signals from all layers. Empty list if no
                actionable signals.

        Example:
            >>> signals = dispatcher.dispatch(
            ...     "AAPL", bar, store,
            ...     {"AAPL_L1_VWAP_MR": pos_dict}, {"MSFT"}
            ... )
            >>> [s.signal for s in signals]
            ['BUY']
        """
        signals: List[SignalResult] = []
        now = datetime.now(timezone.utc)

        # Phase 5: VIXY regime gate — halt ALL new BUY signals
        vixy_halt = False
        if vixy_price is not None:
            try:
                from src.strategy.regime_filter import RegimeFilter, RegimeState
                vixy_state = RegimeFilter().get_vixy_regime(vixy_price, self._config)
                if vixy_state == RegimeState.HALT_ALL:
                    vixy_halt = True
                    logger.debug(
                        "Phase5 VIXY halt for %s: price=%.2f", symbol, vixy_price
                    )
            except Exception:
                pass

        # Bug 5: stop-out cooldown check for this symbol
        cooldown_minutes = self._config.get("risk", {}).get("stop_out_cooldown_minutes", 30)
        in_cooldown = False
        if symbol in self._last_stop_out:
            elapsed = now - self._last_stop_out[symbol]
            if elapsed < timedelta(minutes=cooldown_minutes):
                in_cooldown = True
                logger.debug(
                    "Cooldown active for %s: %.0fs remaining",
                    symbol,
                    (timedelta(minutes=cooldown_minutes) - elapsed).total_seconds(),
                )

        # Phase 4D: confidence gate threshold (stored as integer 0-100)
        min_conf_pct = self._config.get("prediction", {}).get("min_confidence_to_trade", 0)
        min_conf = min_conf_pct / 100.0 if min_conf_pct > 0 else 0.0

        # Count current open positions once for the whole dispatch call
        position_count = self.get_open_position_count(open_positions)
        at_max = position_count >= self.max_positions

        # Determine the bar count for this symbol (used for L3 cadence)
        bar_count = bar_store.get_bar_count(symbol)

        for layer in self.layers:
            layer_name = layer.layer_name

            # ----------------------------------------------------------------
            # Layer-specific firing conditions
            # ----------------------------------------------------------------
            if layer_name == "L2_ORB":
                # Layer 2 only fires when explicitly enabled (regime) AND the
                # opening range has been computed for this symbol.
                if not layer2_enabled:
                    logger.debug(
                        "L2_ORB skipped for %s: layer2_enabled=False", symbol
                    )
                    continue
                opening_range = bar_store.get_opening_range(symbol)
                if opening_range is None:
                    logger.debug(
                        "L2_ORB skipped for %s: opening range not yet computed",
                        symbol,
                    )
                    continue

            elif layer_name == "L3_RSI_SCALP":
                # Layer 3 fires on every l3_cadence-th bar. cadence <= 1 = every bar.
                if self.l3_cadence > 1 and (bar_count == 0 or (bar_count % self.l3_cadence != 0)):
                    logger.debug(
                        "L3_RSI_SCALP skipped for %s: bar_count=%d cadence=%d",
                        symbol,
                        bar_count,
                        self.l3_cadence,
                    )
                    continue

            # ----------------------------------------------------------------
            # Retrieve this layer's open position for the symbol (may be None)
            # ----------------------------------------------------------------
            position_key = f"{symbol}_{layer_name}"
            open_position = open_positions.get(position_key)

            # ----------------------------------------------------------------
            # Evaluate the layer
            # ----------------------------------------------------------------
            try:
                layer_signals = layer.evaluate_bar(
                    symbol=symbol,
                    bar=bar,
                    bar_store=bar_store,
                    open_position=open_position,
                )
            except Exception as exc:
                logger.error(
                    "Layer %s raised exception for symbol %s: %s",
                    layer_name,
                    symbol,
                    str(exc),
                    exc_info=True,
                )
                continue

            if not layer_signals:
                continue

            # ----------------------------------------------------------------
            # Apply deduplication rules per signal
            # ----------------------------------------------------------------
            for signal in layer_signals:
                if signal.signal == "EXIT":
                    # EXIT signals always pass through, no dedup applied.
                    signals.append(signal)
                    # Bug 5: record stop-out time so cooldown blocks re-entry
                    exit_reason = (signal.metadata or {}).get("exit_reason", "")
                    if exit_reason and (
                        "stop" in exit_reason.lower() or "stop_breach" in exit_reason.lower()
                        or exit_reason.startswith("stop")
                    ):
                        self._last_stop_out[symbol] = now
                        logger.info(
                            "Stop-out recorded for %s — cooldown=%dmin active",
                            symbol, cooldown_minutes,
                        )

                elif signal.signal == "BUY":
                    # Phase 5: VIXY halt gate
                    if vixy_halt:
                        logger.debug(
                            "BUY blocked for %s/%s: VIXY halt", symbol, layer_name
                        )
                        continue

                    # Bug 5: stop-out cooldown gate
                    if in_cooldown:
                        logger.debug(
                            "BUY blocked for %s/%s: stop-out cooldown",
                            symbol, layer_name,
                        )
                        continue

                    # Reject if at max positions.
                    if at_max:
                        logger.debug(
                            "BUY dedup for %s/%s: at max_positions=%d",
                            symbol,
                            layer_name,
                            self.max_positions,
                        )
                        continue

                    # Reject if this symbol already has a pending open order.
                    if symbol in open_orders:
                        logger.debug(
                            "BUY dedup for %s/%s: open order already pending",
                            symbol,
                            layer_name,
                        )
                        continue

                    # Reject if this layer already has an open position for
                    # this symbol (per-layer deduplication).
                    if open_position is not None:
                        logger.debug(
                            "BUY dedup for %s/%s: position already open",
                            symbol,
                            layer_name,
                        )
                        continue

                    # Phase 4D: confidence gate
                    if min_conf > 0 and signal.confidence is not None:
                        if signal.confidence < min_conf:
                            logger.debug(
                                "BUY blocked for %s/%s: confidence=%.3f < min=%.3f",
                                symbol, layer_name, signal.confidence, min_conf,
                            )
                            continue

                    signals.append(signal)

                else:
                    # HOLD or any other signal type — never routed to order mgr.
                    logger.debug(
                        "Signal %s/%s HOLD — not forwarded",
                        symbol,
                        layer_name,
                    )

        if signals:
            logger.debug(
                "Dispatched %d signal(s) for %s: %s",
                len(signals),
                symbol,
                [f"{s.signal}/{s.layer_name}" for s in signals],
            )

        return signals

    def get_open_position_count(self, open_positions: Dict[str, dict]) -> int:
        """Count number of currently open positions (unique symbols).

        Counts distinct symbols across all layer keys in open_positions. The
        key format is "{symbol}_{layer_name}", so we extract the symbol prefix
        by stripping the known layer suffixes, then count unique symbols.

        Args:
            open_positions: Dict of all open positions keyed by
                "{symbol}_{layer_name}".

        Returns:
            int: Count of unique open position symbols, capped at max_positions
                to avoid any double-counting edge cases.

        Example:
            >>> count = dispatcher.get_open_position_count({
            ...     "AAPL_L1_VWAP_MR": {...},
            ...     "MSFT_L4_VOL_SURGE": {...},
            ... })
            >>> count
            2
        """
        if not open_positions:
            return 0

        # Known layer name suffixes used in key construction.
        _layer_suffixes = (
            "_L1_VWAP_MR",
            "_L2_ORB",
            "_L3_RSI_SCALP",
            "_L4_VOL_SURGE",
        )

        unique_symbols: Set[str] = set()
        for key, pos_dict in open_positions.items():
            # Prefer the explicit "symbol" field in the position dict when
            # available, as it avoids any string-parsing ambiguity.
            sym = pos_dict.get("symbol") if isinstance(pos_dict, dict) else None
            if sym:
                unique_symbols.add(sym)
            else:
                # Fallback: strip known layer suffix from key.
                raw = key
                for suffix in _layer_suffixes:
                    if raw.endswith(suffix):
                        raw = raw[: len(raw) - len(suffix)]
                        break
                unique_symbols.add(raw)

        return min(len(unique_symbols), self.max_positions)
