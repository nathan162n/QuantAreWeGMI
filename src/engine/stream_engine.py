"""Stream engine — WebSocket event-driven core of the live trading system.

Manages the Alpaca WebSocket connection, drives all intraday logic from
real-time bar events, and coordinates signal generation, order submission,
and stop-loss monitoring.

This replaces all APScheduler polling during market hours. Every incoming
1-minute bar triggers signal evaluation for that symbol immediately.

WebSocket reconnection:
    On disconnect, reconnect with exponential backoff (1s, 2s, 4s ... max 60s).
    Re-subscribe to all symbols after each reconnection. Never raise on
    transient disconnect.

Concurrency model:
    - WebSocket callbacks run in the asyncio event loop.
    - Signal evaluation is synchronous within each callback.
    - Order submission is async (awaited within callback).
    - Discord notifications are fire-and-forget (create_task, never awaited).

Example:
    >>> engine = StreamEngine(
    ...     alpaca_client=client,
    ...     bar_store=store,
    ...     bar_dispatcher=dispatcher,
    ...     order_manager=order_mgr,
    ...     pdt_guard=pdt,
    ...     circuit_breaker=cb,
    ...     notification_queue=nq,
    ...     regime_filter=rf,
    ...     symbols=["AAPL", "MSFT", "NVDA"],
    ... )
    >>> await engine.start()
"""

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

from src.risk.position_sizer import PositionSizer
from src.utils.logger import get_logger

logger = get_logger("engine.stream_engine")

# Exponential backoff configuration for WebSocket reconnection.
_RECONNECT_BASE_DELAY_S: float = 1.0
_RECONNECT_MAX_DELAY_S: float = 60.0
_RECONNECT_BACKOFF_FACTOR: float = 2.0


class StreamEngine:
    """Drives all intraday trading from WebSocket bar events.

    On each incoming bar:
      1. Update BarStore for that symbol/timeframe.
      2. Check stop-losses for any open position in that symbol.
      3. Dispatch to BarDispatcher for signal evaluation.
      4. Process BUY and EXIT signals.
      5. Check circuit breaker.

    Attributes:
        alpaca_client: AlpacaClient with stream_bars / subscribe_bars method.
        bar_store: BarStore instance.
        bar_dispatcher: BarDispatcher instance.
        order_manager: OrderManager instance.
        pdt_guard: PDTGuard instance.
        circuit_breaker: CircuitBreaker instance.
        notification_queue: NotificationQueue instance.
        regime_filter: RegimeFilter instance.
        symbols: List of symbols to subscribe to.
        max_positions: Maximum simultaneous open positions.
        position_sizer: PositionSizer instance (created internally).

    Example:
        >>> engine = StreamEngine(
        ...     alpaca_client=client,
        ...     bar_store=store,
        ...     bar_dispatcher=dispatcher,
        ...     order_manager=order_mgr,
        ...     pdt_guard=pdt,
        ...     circuit_breaker=cb,
        ...     notification_queue=nq,
        ...     regime_filter=rf,
        ...     symbols=["AAPL"],
        ... )
        >>> await engine.start()
    """

    def __init__(
        self,
        alpaca_client,
        bar_store,
        bar_dispatcher,
        order_manager,
        pdt_guard,
        circuit_breaker,
        notification_queue,
        regime_filter,
        symbols: List[str],
        max_positions: int = 3,
    ):
        """Initialize the stream engine.

        Args:
            alpaca_client: AlpacaClient instance. Must expose stream_bars(symbols,
                callback) coroutine.
            bar_store: BarStore instance for in-memory bar history.
            bar_dispatcher: BarDispatcher instance for routing bars to strategies.
            order_manager: OrderManager instance for order submission.
            pdt_guard: PDTGuard instance for PDT rule enforcement.
            circuit_breaker: CircuitBreaker instance for portfolio-level safety.
            notification_queue: NotificationQueue instance for Discord delivery.
            regime_filter: RegimeFilter instance for market regime detection.
            symbols: List of ticker symbols to subscribe to.
            max_positions: Maximum simultaneous open positions. Default 3.

        Example:
            >>> engine = StreamEngine(client, store, dispatcher, om, pdt,
            ...                       cb, nq, rf, ["AAPL", "NVDA"])
        """
        self.alpaca_client = alpaca_client
        self.bar_store = bar_store
        self.bar_dispatcher = bar_dispatcher
        self.order_manager = order_manager
        self.pdt_guard = pdt_guard
        self.circuit_breaker = circuit_breaker
        self.notification_queue = notification_queue
        self.regime_filter = regime_filter
        self.symbols = symbols
        self.max_positions = max_positions

        # Internal state
        # Keyed by "{symbol}_{layer_name}", value is the position dict.
        self._open_positions: Dict[str, dict] = {}
        # Set of symbols with a pending open order.
        self._open_orders: Set[str] = set()
        # Flag set by stop() to break the reconnect loop.
        self._running: bool = False
        # Track equity value at open for circuit breaker daily-loss check.
        self._equity_at_open: float = 0.0

        # PositionSizer is instantiated here — reads from config risk section.
        import yaml, os
        _cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml")
        with open(_cfg_path, "r") as _f:
            _cfg = yaml.safe_load(_f)
        _risk = _cfg.get("risk", {})
        self.position_sizer = PositionSizer(
            dollar_risk_per_trade=float(_risk.get("dollar_risk_per_trade", 150.00)),
            max_position_pct=float(_risk.get("max_position_pct", 0.10)),
            max_positions=max_positions,
        )

    async def start(self) -> None:
        """Start the WebSocket stream. Runs until shutdown signal.

        Connects to the Alpaca WebSocket stream and subscribes to all
        configured symbols. Reconnects automatically with exponential
        backoff on disconnect. The loop exits only when stop() has been
        called.

        Example:
            >>> await engine.start()
        """
        self._running = True
        delay = _RECONNECT_BASE_DELAY_S

        # Fetch initial equity for circuit breaker reference at market open.
        try:
            account = await asyncio.get_event_loop().run_in_executor(
                None, self.alpaca_client.get_account
            )
            self._equity_at_open = float(account.get("equity", 0.0))
            logger.info(
                "StreamEngine starting — equity_at_open=$%.2f, symbols=%s",
                self._equity_at_open,
                self.symbols,
            )
        except Exception as exc:
            logger.error(
                "StreamEngine: failed to fetch initial equity: %s", str(exc)
            )

        while self._running:
            try:
                logger.info(
                    "StreamEngine: connecting WebSocket for %d symbol(s)",
                    len(self.symbols),
                )
                # stream_bars is a long-running coroutine that calls _on_bar
                # for every arriving 1-minute bar.
                await self.alpaca_client.stream_bars(
                    symbols=self.symbols,
                    callback=self._on_bar,
                )
                # If stream_bars returns normally (not via exception),
                # exit the reconnect loop only if stop() was called.
                if not self._running:
                    break
                logger.warning(
                    "StreamEngine: stream ended unexpectedly — reconnecting"
                )
            except asyncio.CancelledError:
                logger.info("StreamEngine: received CancelledError — shutting down")
                self._running = False
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.error(
                    "StreamEngine: WebSocket error (will reconnect in %.0fs): %s",
                    delay,
                    str(exc),
                )

            if not self._running:
                break

            # Exponential backoff before reconnecting.
            await asyncio.sleep(delay)
            delay = min(delay * _RECONNECT_BACKOFF_FACTOR, _RECONNECT_MAX_DELAY_S)

            if self._running:
                await self._on_reconnect()
                # Reset delay on successful reconnect start.
                delay = _RECONNECT_BASE_DELAY_S

        logger.info("StreamEngine: stopped.")

    async def _on_bar(self, bar: dict) -> None:
        """Handle every incoming 1-minute bar.

        This is the hot path. It must complete within ~500ms to avoid
        falling behind the live stream. All heavy work (order submission)
        is awaited inline. Discord notifications are dispatched via
        create_task (fire-and-forget) to stay non-blocking.

        Flow:
          1. Update BarStore(symbol, "1Min", bar).
          2. Check stop-losses for any open position in this symbol.
          3. If circuit breaker is triggered, skip signal generation.
          4. Dispatch to BarDispatcher for signal evaluation.
          5. For each BUY signal: compute position size, submit order.
          6. For each EXIT signal: PDT check, submit sell.
          7. Check circuit breaker with current equity.

        Args:
            bar: Bar dict with keys: symbol, open, high, low, close,
                volume, timestamp.

        Example:
            >>> await engine._on_bar({
            ...     "symbol": "AAPL", "open": 150.0, "high": 151.5,
            ...     "low": 149.8, "close": 151.2, "volume": 8500,
            ...     "timestamp": datetime.now(timezone.utc),
            ... })
        """
        symbol: str = bar.get("symbol", "")
        if not symbol:
            logger.warning("_on_bar received bar with no symbol — skipping")
            return

        # ----------------------------------------------------------------
        # 1. Update BarStore
        # ----------------------------------------------------------------
        try:
            self.bar_store.update(symbol, "1Min", bar)
        except Exception as exc:
            logger.error(
                "_on_bar: BarStore update failed for %s: %s", symbol, str(exc)
            )
            return

        # ----------------------------------------------------------------
        # 2. Stop-loss check for any open position in this symbol
        # ----------------------------------------------------------------
        positions_for_symbol = {
            k: v
            for k, v in self._open_positions.items()
            if isinstance(v, dict) and v.get("symbol") == symbol
        }
        for pos_key, position in list(positions_for_symbol.items()):
            layer_name = position.get("layer_name", "")
            # Find the matching strategy layer instance.
            matching_layer = None
            for layer in self.bar_dispatcher.layers:
                if layer.layer_name == layer_name:
                    matching_layer = layer
                    break

            if matching_layer is None:
                continue

            try:
                should_exit = matching_layer.should_exit(
                    symbol=symbol,
                    bar=bar,
                    bar_store=self.bar_store,
                    position_data=position,
                )
            except Exception as exc:
                logger.error(
                    "_on_bar: should_exit raised for %s/%s: %s",
                    symbol,
                    layer_name,
                    str(exc),
                )
                should_exit = False

            if should_exit:
                logger.info(
                    "Stop-loss triggered for %s/%s — submitting exit",
                    symbol,
                    layer_name,
                )
                # Build a minimal EXIT signal for the position.
                from src.strategy.base_strategy import SignalResult

                exit_signal = SignalResult(
                    symbol=symbol,
                    signal="EXIT",
                    confidence=1.0,
                    signal_price=float(bar.get("close", 0.0)),
                    layer_name=layer_name,
                    stop_price=float(position.get("stop_price", 0.0)),
                    metadata={"exit_reason": "STOP_LOSS"},
                )
                await self._process_exit_signal(exit_signal, position)

        # ----------------------------------------------------------------
        # 3. Circuit breaker check before generating new signals
        # ----------------------------------------------------------------
        if self.circuit_breaker.is_active():
            logger.debug(
                "_on_bar: circuit breaker active — skipping signal generation for %s",
                symbol,
            )
            return

        # ----------------------------------------------------------------
        # 4. Dispatch to BarDispatcher
        # ----------------------------------------------------------------
        layer2_enabled: bool = self.regime_filter.is_layer2_enabled()
        try:
            signals = self.bar_dispatcher.dispatch(
                symbol=symbol,
                bar=bar,
                bar_store=self.bar_store,
                open_positions=self._open_positions,
                open_orders=self._open_orders,
                layer2_enabled=layer2_enabled,
            )
        except Exception as exc:
            logger.error(
                "_on_bar: BarDispatcher.dispatch raised for %s: %s",
                symbol,
                str(exc),
                exc_info=True,
            )
            signals = []

        # ----------------------------------------------------------------
        # 5 & 6. Process signals
        # ----------------------------------------------------------------
        for signal in signals:
            try:
                if signal.signal == "BUY":
                    await self._process_buy_signal(signal)
                elif signal.signal == "EXIT":
                    # Locate the open position for this symbol/layer.
                    pos_key = f"{symbol}_{signal.layer_name}"
                    position = self._open_positions.get(pos_key)
                    if position is not None:
                        await self._process_exit_signal(signal, position)
                    else:
                        logger.warning(
                            "_on_bar: EXIT signal for %s but no open position found "
                            "(key=%s)",
                            symbol,
                            pos_key,
                        )
            except Exception as exc:
                logger.error(
                    "_on_bar: error processing signal %s/%s: %s",
                    signal.signal,
                    signal.layer_name,
                    str(exc),
                    exc_info=True,
                )

        # ----------------------------------------------------------------
        # 7. Post-bar circuit breaker check with current equity
        # ----------------------------------------------------------------
        try:
            account = await asyncio.get_event_loop().run_in_executor(
                None, self.alpaca_client.get_account
            )
            current_equity = float(account.get("equity", self._equity_at_open))
            cb_reason = self.circuit_breaker.check_all_conditions(
                portfolio_now=current_equity,
                portfolio_at_open=self._equity_at_open,
            )
            if cb_reason:
                logger.critical(
                    "_on_bar: circuit breaker condition met for %s — triggering: %s",
                    symbol,
                    cb_reason,
                )
                self.circuit_breaker.trigger(cb_reason)
                # Send Discord alert (fire-and-forget).
                asyncio.get_event_loop().create_task(
                    self._send_circuit_breaker_alert(cb_reason)
                )
        except Exception as exc:
            logger.error(
                "_on_bar: circuit breaker equity check failed: %s", str(exc)
            )

    async def _send_circuit_breaker_alert(self, reason: str) -> None:
        """Send a circuit breaker alert via the notification queue.

        Args:
            reason: Human-readable trigger reason.

        Example:
            >>> await engine._send_circuit_breaker_alert("Daily loss -3.1%")
        """
        try:
            self.notification_queue.enqueue_alert(
                alert_type="CIRCUIT_BREAKER_TRIGGERED",
                message=reason,
                data={"reason": reason},
            )
        except Exception as exc:
            logger.error("_send_circuit_breaker_alert failed: %s", str(exc))

    async def _on_reconnect(self) -> None:
        """Called after WebSocket reconnection. Re-subscribe, log, Discord alert.

        Logs the reconnection event, sends a system restart Discord alert,
        and resets any transient state that should not persist across
        reconnections.

        Example:
            >>> await engine._on_reconnect()
        """
        logger.info(
            "StreamEngine: reconnected — re-subscribing to %d symbol(s)",
            len(self.symbols),
        )
        try:
            self.notification_queue.enqueue_alert(
                alert_type="SYSTEM_RESTART",
                message=(
                    f"StreamEngine reconnected at "
                    f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}. "
                    f"Re-subscribed to {len(self.symbols)} symbol(s)."
                ),
                data={"symbols": ", ".join(self.symbols)},
            )
        except Exception as exc:
            logger.error("_on_reconnect: notification failed: %s", str(exc))

    async def _process_exit_signal(
        self, signal, position: dict
    ) -> None:
        """Process an EXIT signal for an open position.

        Checks PDT guard before submitting. If PDT would be violated, a
        Discord PDT block alert is sent and the exit is skipped. If the
        exit is allowed, a market sell is submitted via the order manager,
        the position is removed from _open_positions, a day trade is
        recorded if applicable, and a Discord trade exit embed is sent.

        Args:
            signal: SignalResult with signal="EXIT", symbol, signal_price,
                and layer_name populated.
            position: Open position dict with at minimum: symbol, entry_time,
                qty, entry_price, stop_price, layer_name, trade_id (optional).

        Example:
            >>> await engine._process_exit_signal(exit_signal, position_dict)
        """
        symbol = position.get("symbol", signal.symbol)
        entry_time = position.get("entry_time")
        qty = position.get("qty", 0)
        entry_price = position.get("entry_price", 0.0)
        layer_name = position.get("layer_name", signal.layer_name)
        trade_id = position.get("trade_id")
        pos_key = f"{symbol}_{layer_name}"

        if qty <= 0:
            logger.warning(
                "_process_exit_signal: position %s has qty=%d — skipping",
                pos_key,
                qty,
            )
            return

        # PDT check
        can_exit = self.pdt_guard.can_exit(symbol, entry_time)
        if not can_exit:
            logger.warning(
                "PDT BLOCK: cannot exit %s (layer=%s) — would exceed 3 day trades",
                symbol,
                layer_name,
            )
            # Fire-and-forget Discord alert
            asyncio.get_event_loop().create_task(
                self._send_pdt_block_alert(symbol, layer_name, qty, entry_price)
            )
            return

        is_day_trade = self.pdt_guard.is_day_trade(symbol, entry_time)
        exit_reason = signal.metadata.get("exit_reason", "SIGNAL")

        try:
            if trade_id is not None:
                success = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda tid=trade_id, sym=symbol, q=qty, r=exit_reason, dt=is_day_trade: (
                        self.order_manager.submit_exit_order(
                            trade_id=tid,
                            symbol=sym,
                            qty=q,
                            exit_reason=r,
                            is_day_trade=dt,
                        )
                    ),
                )
            else:
                order_id = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda sym=symbol, q=qty: (
                        self.alpaca_client.submit_market_order(sym, q, "sell")
                    ),
                )
                success = order_id is not None

            if success:
                # Remove from open positions
                self._open_positions.pop(pos_key, None)
                self._open_orders.discard(symbol)

                # Record day trade if applicable
                if is_day_trade and trade_id is None:
                    # submit_exit_order handles PDT recording when trade_id is set.
                    self.pdt_guard.record_day_trade(symbol)

                logger.info(
                    "EXIT executed: %s %d shares (layer=%s, reason=%s, day_trade=%s)",
                    symbol,
                    qty,
                    layer_name,
                    exit_reason,
                    is_day_trade,
                )

                # Fire-and-forget Discord exit embed
                exit_price = float(signal.signal_price) if signal.signal_price else 0.0
                realized_pnl = round((exit_price - entry_price) * qty, 4)
                asyncio.get_event_loop().create_task(
                    self._send_trade_exit_notification(
                        symbol=symbol,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=qty,
                        realized_pnl=realized_pnl,
                        exit_reason=exit_reason,
                        layer_name=layer_name,
                        entry_time=position.get("entry_time"),
                    )
                )
            else:
                logger.error(
                    "EXIT order not submitted for %s (layer=%s)", symbol, layer_name
                )

        except Exception as exc:
            logger.error(
                "_process_exit_signal: exception for %s/%s: %s",
                symbol,
                layer_name,
                str(exc),
                exc_info=True,
            )

    async def _send_pdt_block_alert(
        self,
        symbol: str,
        layer_name: str,
        qty: int,
        entry_price: float,
    ) -> None:
        """Send a PDT block Discord alert (fire-and-forget helper).

        Args:
            symbol: Ticker symbol.
            layer_name: Strategy layer that owns the position.
            qty: Number of shares held.
            entry_price: Entry price of the position.

        Example:
            >>> asyncio.create_task(engine._send_pdt_block_alert("AAPL", "L1_VWAP_MR", 5, 150.0))
        """
        try:
            rolling_count = self.pdt_guard.get_rolling_count()
            self.notification_queue.enqueue_alert(
                alert_type="PDT_BLOCK",
                message=(
                    f"EXIT blocked for {symbol} (layer={layer_name}). "
                    f"Closing now would trigger day trade #{rolling_count + 1} "
                    f"(limit=3). Position held."
                ),
                data={
                    "symbol": symbol,
                    "layer": layer_name,
                    "qty": qty,
                    "entry_price": f"${entry_price:.2f}",
                    "rolling_day_trades": rolling_count,
                },
            )
        except Exception as exc:
            logger.error("_send_pdt_block_alert failed: %s", str(exc))

    async def _send_trade_exit_notification(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        qty: int,
        realized_pnl: float,
        exit_reason: str,
        layer_name: str,
        entry_time=None,
    ) -> None:
        """Send a Discord trade exit embed (fire-and-forget helper).

        Args:
            symbol: Ticker symbol.
            entry_price: Entry fill price.
            exit_price: Exit price (signal_price; actual fill may differ).
            qty: Number of shares.
            realized_pnl: Estimated realized P&L.
            exit_reason: Reason for exit (SIGNAL, STOP, TRAIL, EOD, etc.).
            layer_name: Strategy layer name.
            entry_time: Optional entry datetime for hold duration calculation.

        Example:
            >>> asyncio.create_task(engine._send_trade_exit_notification(...))
        """
        try:
            from datetime import datetime, timezone
            hold_minutes = 0.0
            exit_time = datetime.now(timezone.utc)
            if entry_time is not None:
                hold_minutes = (exit_time - entry_time).total_seconds() / 60.0

            self.notification_queue.enqueue_trade_exit(
                trade_data={
                    "symbol": symbol,
                    "layer_name": layer_name,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": qty,
                    "pnl": realized_pnl,
                    "hold_minutes": hold_minutes,
                    "exit_reason": exit_reason,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "slippage": 0.0,
                    "open_positions": len(self._open_positions),
                }
            )
        except Exception as exc:
            logger.error("_send_trade_exit_notification failed: %s", str(exc))

    async def _process_buy_signal(self, signal) -> None:
        """Process a BUY signal. Computes shares, submits order, sends Discord embed.

        Performs final position count and duplicate order guards before
        submitting. Calls PositionSizer to determine share quantity. On
        successful order submission adds the position to _open_positions and
        the symbol to _open_orders, then fires a Discord BUY embed.

        Args:
            signal: SignalResult with signal="BUY", symbol, signal_price,
                stop_price, layer_name, confidence, and metadata populated.

        Example:
            >>> await engine._process_buy_signal(buy_signal)
        """
        symbol = signal.symbol
        layer_name = signal.layer_name
        pos_key = f"{symbol}_{layer_name}"

        # Final guard: position count may have changed since dispatch.
        current_count = self.bar_dispatcher.get_open_position_count(
            self._open_positions
        )
        if current_count >= self.max_positions:
            logger.info(
                "_process_buy_signal: skipping %s — at max positions (%d)",
                symbol,
                self.max_positions,
            )
            return

        if symbol in self._open_orders:
            logger.info(
                "_process_buy_signal: skipping %s — open order already pending",
                symbol,
            )
            return

        if pos_key in self._open_positions:
            logger.info(
                "_process_buy_signal: skipping %s — position already exists (%s)",
                symbol,
                pos_key,
            )
            return

        # Fetch current account equity and buying power for position sizing.
        try:
            account = await asyncio.get_event_loop().run_in_executor(
                None, self.alpaca_client.get_account
            )
            equity = float(account.get("equity", 0.0))
            buying_power = float(account.get("buying_power", 0.0))
        except Exception as exc:
            logger.error(
                "_process_buy_signal: failed to fetch account for %s: %s",
                symbol,
                str(exc),
            )
            return

        if equity <= 0 or buying_power <= 0:
            logger.warning(
                "_process_buy_signal: zero equity/buying_power for %s — skipping",
                symbol,
            )
            return

        # Compute shares via PositionSizer.
        regime_scalar = self.regime_filter.current_size_scalar
        shares = self.position_sizer.compute_shares(
            signal=signal,
            equity=equity,
            buying_power=buying_power,
            current_open_positions=current_count,
            regime_scalar=regime_scalar,
        )

        if shares < 1:
            logger.info(
                "_process_buy_signal: shares=0 for %s/%s — skipping (price=%.2f)",
                symbol,
                layer_name,
                signal.signal_price,
            )
            return

        # Submit entry order through the order manager.
        stop_price = float(signal.stop_price) if signal.stop_price else 0.0
        # Conservative take-profit: 2× the risk distance above entry.
        risk_distance = abs(signal.signal_price - stop_price) if stop_price else 0.0
        take_profit_price = (
            round(signal.signal_price + 2.0 * risk_distance, 2)
            if risk_distance > 0
            else round(signal.signal_price * 1.02, 2)
        )

        try:
            trade_id = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda sym=symbol, q=shares, ln=layer_name, sp=stop_price,
                       tp=take_profit_price, sgp=signal.signal_price: (
                    self.order_manager.submit_entry_order(
                        symbol=sym,
                        qty=q,
                        side="buy",
                        strategy_name=ln,
                        stop_price=sp,
                        take_profit_price=tp,
                        signal_metadata=signal.metadata,
                        signal_price=sgp,
                    )
                ),
            )
        except Exception as exc:
            logger.error(
                "_process_buy_signal: submit_entry_order raised for %s/%s: %s",
                symbol,
                layer_name,
                str(exc),
                exc_info=True,
            )
            return

        if trade_id is None:
            logger.error(
                "_process_buy_signal: order submission returned None for %s/%s",
                symbol,
                layer_name,
            )
            return

        # Record position and pending order.
        now_utc = datetime.now(timezone.utc)
        self._open_positions[pos_key] = {
            "symbol": symbol,
            "layer_name": layer_name,
            "entry_price": float(signal.signal_price),
            "entry_time": now_utc,
            "qty": shares,
            "stop_price": stop_price,
            "trade_id": trade_id,
            "bars_held": 0,
        }
        self._open_orders.add(symbol)

        logger.info(
            "BUY submitted: %d %s @ ~$%.2f (layer=%s, stop=$%.2f, "
            "regime_scalar=%.2f, trade_id=%s)",
            shares,
            symbol,
            signal.signal_price,
            layer_name,
            stop_price,
            regime_scalar,
            trade_id,
        )

        # Fire-and-forget Discord BUY embed.
        asyncio.get_event_loop().create_task(
            self._send_trade_entry_notification(
                signal=signal,
                shares=shares,
                stop_price=stop_price,
                equity=equity,
                regime_scalar=regime_scalar,
            )
        )

    async def _send_trade_entry_notification(
        self,
        signal,
        shares: int,
        stop_price: float,
        equity: float,
        regime_scalar: float,
    ) -> None:
        """Send a Discord trade entry embed (fire-and-forget helper).

        Args:
            signal: SignalResult with symbol, signal_price, layer_name, metadata.
            shares: Number of shares purchased.
            stop_price: Stop loss price.
            equity: Account equity at time of entry.
            regime_scalar: Regime size scalar applied.

        Example:
            >>> asyncio.create_task(engine._send_trade_entry_notification(...))
        """
        try:
            self.notification_queue.enqueue_trade_entry(
                trade_data={
                    "symbol": signal.symbol,
                    "entry_price": float(signal.signal_price),
                    "shares": shares,
                    "stop_price": stop_price,
                    "strategy_name": signal.layer_name,
                    "regime": self.regime_filter.current_regime,
                    "z_score": signal.metadata.get("z_score", 0),
                    "rsi": signal.metadata.get("rsi", 0),
                    "total_equity": equity,
                    "open_positions": len(self._open_positions),
                    "cash_remaining": max(0.0, equity - shares * float(signal.signal_price or 0.0)),
                    "regime_scalar": regime_scalar,
                }
            )
        except Exception as exc:
            logger.error("_send_trade_entry_notification failed: %s", str(exc))

    def stop(self) -> None:
        """Signal the stream engine to stop gracefully.

        Sets _running to False so that the reconnect loop in start() exits
        after the current WebSocket operation completes.

        Example:
            >>> engine.stop()
        """
        self._running = False
        logger.info("StreamEngine: stop() called — will exit after current stream")
