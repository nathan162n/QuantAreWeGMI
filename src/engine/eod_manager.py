"""EOD Manager — forced position close and post-market cleanup.

Scheduled to run at 15:30 ET. Closes all positions that are not PDT-blocked.
PDT-blocked positions are held overnight with a Discord alert sent via the
notification queue.

Actions at 15:30 ET:
    1. For every open position:
       a. Check if closing would be a day trade AND rolling_day_trades == 3
       b. If PDT-blocked: hold overnight, log WARNING, send Discord PDT alert
       c. If not blocked: submit market sell, log, Discord trade exit embed
    2. Cancel all open orders that did not fill
    3. Log EOD summary to database / logger
    4. Enqueue daily summary notification

Example:
    >>> manager = EODManager(
    ...     alpaca_client=client,
    ...     pdt_guard=pdt,
    ...     order_manager=order_mgr,
    ...     notification_queue=nq,
    ...     repository=repo,
    ... )
    >>> summary = await manager.run_eod_close(open_positions)
    >>> print(summary["closed"], summary["pdt_blocked"])
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger("engine.eod_manager")


class EODManager:
    """Manages end-of-day position close at 15:30 ET.

    Actions at 15:30 ET:
      1. For every open position:
         a. Check if closing would be a day trade AND rolling_day_trades == 3
         b. If PDT-blocked: hold overnight, log WARNING, send Discord PDT alert
         c. If not blocked: submit market sell, log, Discord trade exit embed
      2. Cancel all open orders that did not fill
      3. Log EOD summary to database
      4. Enqueue daily summary notification

    Attributes:
        alpaca_client: AlpacaClient instance for direct broker calls.
        pdt_guard: PDTGuard instance for PDT rule enforcement.
        order_manager: OrderManager instance for order submission.
        notification_queue: NotificationQueue instance for Discord alerts.
        repository: Optional database repository for logging EOD state.

    Example:
        >>> manager = EODManager(
        ...     alpaca_client=client,
        ...     pdt_guard=pdt_guard,
        ...     order_manager=order_mgr,
        ...     notification_queue=nq,
        ... )
        >>> result = await manager.run_eod_close(open_positions)
        >>> result["closed"]
        2
    """

    def __init__(
        self,
        alpaca_client,
        pdt_guard,
        order_manager,
        notification_queue,
        repository=None,
    ):
        """Initialize the EOD manager.

        Args:
            alpaca_client: AlpacaClient instance with cancel_all_orders and
                get_account methods.
            pdt_guard: PDTGuard instance with can_exit, is_day_trade, and
                get_rolling_count methods.
            order_manager: OrderManager instance with submit_exit_order method.
            notification_queue: NotificationQueue with enqueue_alert,
                enqueue_trade_exit, and enqueue_daily_summary methods.
            repository: Optional database repository for trade lookups and
                EOD state persistence. If None, DB logging is skipped.

        Example:
            >>> mgr = EODManager(client, pdt, om, nq, repo)
        """
        self.alpaca_client = alpaca_client
        self.pdt_guard = pdt_guard
        self.order_manager = order_manager
        self.notification_queue = notification_queue
        self.repository = repository

    async def run_eod_close(
        self,
        open_positions: Dict[str, dict],
    ) -> dict:
        """Execute EOD close sequence for all open positions.

        Iterates open_positions and attempts to exit each one. PDT-blocked
        exits are skipped with a WARNING and a Discord alert. After processing
        all positions, open orders are cancelled and an EOD summary is logged.

        The return dict may be used by the caller to drive the daily summary
        notification.

        Args:
            open_positions: Dict of open positions keyed by
                "{symbol}_{layer_name}". Each value must contain at minimum:
                    - symbol (str): Ticker symbol.
                    - entry_time (datetime): UTC datetime of entry.
                    - qty (int): Number of shares.
                    - stop_price (float): Stop loss price.
                    - layer_name (str): Layer that opened the position.
                    - entry_price (float): Entry fill price (optional, used for
                      Discord embed).
                    - trade_id (int, optional): DB trade record ID.

        Returns:
            dict: Summary with keys:
                - closed (int): Number of positions successfully closed.
                - pdt_blocked (int): Number of positions held overnight due
                  to PDT limit.
                - cancelled_orders (int): Number of open orders cancelled.

        Example:
            >>> result = await manager.run_eod_close({
            ...     "AAPL_L1_VWAP_MR": {
            ...         "symbol": "AAPL", "entry_time": datetime(...),
            ...         "qty": 5, "stop_price": 148.0,
            ...         "layer_name": "L1_VWAP_MR", "entry_price": 150.0,
            ...     }
            ... })
            >>> result["closed"]
            1
        """
        closed = 0
        pdt_blocked = 0

        logger.info(
            "EOD close starting — %d open position(s) to evaluate",
            len(open_positions),
        )

        for position_key, pos in list(open_positions.items()):
            symbol = pos.get("symbol", "UNKNOWN")
            entry_time = pos.get("entry_time")
            qty = pos.get("qty", 0)
            entry_price = pos.get("entry_price", 0.0)
            layer_name = pos.get("layer_name", "UNKNOWN")
            trade_id = pos.get("trade_id")

            if qty <= 0:
                logger.warning(
                    "EOD: position %s has qty=%d — skipping", position_key, qty
                )
                continue

            # ------------------------------------------------------------------
            # PDT check: determine if closing would violate the PDT rule
            # ------------------------------------------------------------------
            can_exit = self.pdt_guard.can_exit(symbol, entry_time)

            if not can_exit:
                rolling_count = self.pdt_guard.get_rolling_count()
                logger.warning(
                    "EOD PDT BLOCK: %s (layer=%s) — holding overnight. "
                    "Rolling day trades=%d (limit=3). entry_time=%s",
                    symbol,
                    layer_name,
                    rolling_count,
                    entry_time.isoformat() if entry_time else "N/A",
                )

                # Send Discord PDT block alert (non-blocking)
                self.notification_queue.enqueue_alert(
                    alert_type="PDT_BLOCK",
                    message=(
                        f"EOD close BLOCKED for {symbol} (layer={layer_name}). "
                        f"Closing this position would trigger a 4th day trade "
                        f"({rolling_count}/3 used). Position held overnight."
                    ),
                    data={
                        "symbol": symbol,
                        "layer": layer_name,
                        "qty": qty,
                        "entry_price": f"${entry_price:.2f}",
                        "rolling_day_trades": rolling_count,
                    },
                )
                pdt_blocked += 1
                continue

            # ------------------------------------------------------------------
            # Submit market sell to close the position
            # ------------------------------------------------------------------
            is_day_trade = self.pdt_guard.is_day_trade(symbol, entry_time)

            try:
                # Bug 3 fix: capture current price BEFORE submitting the exit
                # order, because the position disappears once the sell executes.
                fill_price = None
                try:
                    positions = await asyncio.get_event_loop().run_in_executor(
                        None, self.alpaca_client.get_positions
                    )
                    fill_price = next(
                        (p["current_price"] for p in positions if p["symbol"] == symbol),
                        None,
                    )
                except Exception as price_exc:
                    logger.debug(
                        "EOD: could not fetch pre-exit price for %s: %s",
                        symbol, str(price_exc),
                    )

                if trade_id is not None:
                    # Use the order manager's exit flow (DB update + PDT record)
                    success = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda tid=trade_id, sym=symbol, q=qty, dt=is_day_trade: (
                            self.order_manager.submit_exit_order(
                                trade_id=tid,
                                symbol=sym,
                                qty=q,
                                exit_reason="EOD",
                                is_day_trade=dt,
                            )
                        ),
                    )
                    # If fill_price was captured and exit succeeded, patch it in
                    # (submit_exit_order fetches it from positions post-sell, but
                    # the position may already be gone by then).
                    if success and fill_price and fill_price > 0:
                        try:
                            from src.database import repository as _repo
                            _repo.update_trade(trade_id, {"exit_price": fill_price})
                        except Exception:
                            pass
                else:
                    # No DB trade_id available — submit market order directly
                    order_id = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda sym=symbol, q=qty: (
                            self.alpaca_client.submit_market_order(sym, q, "sell")
                        ),
                    )
                    success = order_id is not None
                    if success and is_day_trade:
                        self.pdt_guard.record_day_trade(symbol)

                if success:
                    logger.info(
                        "EOD close: sold %d %s @ market (layer=%s, day_trade=%s, fill_price=%s)",
                        qty,
                        symbol,
                        layer_name,
                        is_day_trade,
                        f"${fill_price:.4f}" if fill_price else "unknown",
                    )

                    # Send Discord exit embed
                    self.notification_queue.enqueue_trade_exit(
                        trade_data={
                            "symbol": symbol,
                            "entry_price": entry_price,
                            "exit_price": fill_price or 0.0,
                            "shares": qty,
                            "realized_pnl": (
                                round((fill_price - entry_price) * qty, 6)
                                if fill_price and entry_price
                                else 0.0
                            ),
                            "exit_reason": "EOD",
                            "strategy_name": layer_name,
                            "hold_duration": "EOD forced",
                        }
                    )
                    closed += 1
                else:
                    logger.error(
                        "EOD close FAILED for %s (layer=%s): order not submitted",
                        symbol,
                        layer_name,
                    )

            except Exception as exc:
                logger.error(
                    "EOD close exception for %s (layer=%s): %s",
                    symbol,
                    layer_name,
                    str(exc),
                    exc_info=True,
                )

        # ------------------------------------------------------------------
        # Cancel remaining unfilled open orders
        # ------------------------------------------------------------------
        cancelled_orders = await self.cancel_open_orders()

        # ------------------------------------------------------------------
        # Log EOD summary
        # ------------------------------------------------------------------
        summary_msg = self._build_eod_summary(closed, pdt_blocked, cancelled_orders)
        logger.info(summary_msg)

        if self.repository is not None:
            try:
                self.repository.save_system_log(
                    level="INFO",
                    module="eod_manager",
                    message=summary_msg,
                    extra_data={
                        "closed": closed,
                        "pdt_blocked": pdt_blocked,
                        "cancelled_orders": cancelled_orders,
                    },
                )
            except Exception as exc:
                logger.error("EOD: failed to save system log: %s", str(exc))

        # ------------------------------------------------------------------
        # Enqueue daily summary Discord notification
        # ------------------------------------------------------------------
        try:
            portfolio_state: dict = {}
            trades_today: list = []
            metrics: dict = {}

            if self.repository is not None:
                try:
                    account = await asyncio.get_event_loop().run_in_executor(
                        None, self.alpaca_client.get_account
                    )
                    portfolio_value = account.get("equity", 0.0)
                    portfolio_state = {
                        "portfolio_value": portfolio_value,
                        "starting_capital": 100000.0,
                        "regime": "UNKNOWN",
                        "pdt_count": self.pdt_guard.get_rolling_count(),
                    }
                    trades_today = self.repository.get_trades_closed_today() or []
                    metrics = {}
                except Exception as exc:
                    logger.warning(
                        "EOD: could not fetch account/portfolio for summary: %s",
                        str(exc),
                    )

            self.notification_queue.enqueue_daily_summary(
                portfolio_state=portfolio_state,
                trades_today=trades_today,
                metrics=metrics,
            )
        except Exception as exc:
            logger.error(
                "EOD: failed to enqueue daily summary notification: %s", str(exc)
            )

        return {
            "closed": closed,
            "pdt_blocked": pdt_blocked,
            "cancelled_orders": cancelled_orders,
        }

    async def cancel_open_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled.

        Delegates to alpaca_client.cancel_all_orders() which is a blocking
        REST call, run in the executor to avoid blocking the event loop.

        Returns:
            int: Number of orders cancelled (0 if an error occurs).

        Example:
            >>> cancelled = await manager.cancel_open_orders()
            >>> cancelled
            3
        """
        try:
            count = await asyncio.get_event_loop().run_in_executor(
                None, self.alpaca_client.cancel_all_orders
            )
            if count:
                logger.info("EOD: cancelled %d open order(s)", count)
            else:
                logger.debug("EOD: no open orders to cancel")
            return count or 0
        except Exception as exc:
            logger.error("EOD: failed to cancel open orders: %s", str(exc))
            return 0

    def _build_eod_summary(
        self,
        closed: int,
        pdt_blocked: int,
        cancelled_orders: int,
    ) -> str:
        """Build EOD summary string for logging.

        Constructs a human-readable, single-line summary of the EOD close
        sequence result, including timestamp, positions closed, PDT blocks,
        and orders cancelled.

        Args:
            closed: Number of positions successfully closed via market order.
            pdt_blocked: Number of positions held overnight due to PDT limit.
            cancelled_orders: Number of open orders cancelled.

        Returns:
            str: Human-readable EOD summary string.

        Example:
            >>> msg = manager._build_eod_summary(2, 1, 3)
            >>> print(msg)
            'EOD close complete at 15:30 ET — 2 position(s) closed, ...'
        """
        now_utc = datetime.now(timezone.utc)
        timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

        pdt_note = (
            f" [{pdt_blocked} held overnight — PDT limit]"
            if pdt_blocked > 0
            else ""
        )

        return (
            f"EOD close complete at {timestamp_str} — "
            f"{closed} position(s) closed, "
            f"{pdt_blocked} PDT-blocked (held overnight), "
            f"{cancelled_orders} order(s) cancelled."
            f"{pdt_note}"
        )
