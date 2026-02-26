"""Order lifecycle management for the trading system.

Handles the full lifecycle of orders from submission through fill
confirmation, including bracket orders, dry-run mode, and
integration with the PDT guard and risk modules.
"""

import os
from datetime import datetime, timezone
from typing import List, Optional

from src.broker.alpaca_client import get_client
from src.database import repository
from src.utils.logger import get_logger

logger = get_logger("broker.order_manager")


def _load_config() -> dict:
    """Load full config.yml as a dict. Returns {} on any error."""
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
        )
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class OrderManager:
    """Manages order submission, tracking, and lifecycle.

    Provides a high-level interface for submitting orders with
    risk checks, recording trades in the database, and handling
    dry-run mode.

    Attributes:
        client: The Alpaca client instance.
        dry_run: If True, log orders but do not submit them.
    """

    def __init__(self, dry_run: bool = False):
        """Initialize the order manager.

        Args:
            dry_run: If True, simulate order submission without executing.
                Default False.

        Example:
            >>> om = OrderManager(dry_run=False)
        """
        self.client = get_client()
        self.dry_run = dry_run

    def submit_entry_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        strategy_name: str,
        stop_price: float,
        take_profit_price: float,
        signal_metadata: Optional[dict] = None,
        signal_price: float = 0.0,
    ) -> Optional[int]:
        """Submit an entry order and record the trade in the database.

        Submits a market order for the entry leg and a separate stop order
        for protection. The take_profit_price is recorded in the DB only —
        it is not submitted as a bracket leg, which prevents the 3bp stop-out
        bug caused by bracket order price conflicts.

        Phase 4A: Checks reward:risk ratio before submission. Rejects the
        trade if the computed R:R is below the configured minimum.

        Args:
            symbol: Ticker symbol to trade.
            qty: Number of shares.
            side: Trade direction ('buy' or 'sell').
            strategy_name: Name of the strategy generating the trade.
            stop_price: Initial stop loss price.
            take_profit_price: Take profit target price (DB record only).
            signal_metadata: Optional strategy signal details for logging.
            signal_price: Entry signal price used for R:R calculation.

        Returns:
            Optional[int]: Database trade ID on success, None on failure.

        Example:
            >>> trade_id = om.submit_entry_order("AAPL", 3, "buy",
            ...     "mean_reversion_v1", 145.0, 160.0, {"z_score": -2.5},
            ...     signal_price=150.0)
        """
        if not self.client.is_paper_trading():
            trading_mode = os.environ.get("TRADING_MODE", "paper")
            if trading_mode != "live":
                logger.error(
                    "SAFETY: Attempted non-paper order without live mode. Blocked."
                )
                return None

        # Phase 4A: Reward:Risk ratio gate
        if signal_price > 0 and stop_price > 0 and take_profit_price > 0:
            config = _load_config()
            min_rr = config.get("risk", {}).get("min_reward_risk_ratio", 1.5)
            actual_risk = abs(signal_price - stop_price)
            expected_reward = abs(take_profit_price - signal_price)
            if actual_risk > 0:
                rr = expected_reward / actual_risk
                if rr < min_rr:
                    logger.info(
                        "REJECTED %s: R:R=%.2f below min=%.2f (signal=%.4f stop=%.4f tp=%.4f)",
                        symbol, rr, min_rr, signal_price, stop_price, take_profit_price,
                    )
                    return None

        if self.dry_run:
            logger.info(
                "[DRY RUN] Would submit %s %d %s (stop=%.2f, tp=%.2f)",
                side,
                qty,
                symbol,
                stop_price,
                take_profit_price,
            )
            trade = repository.save_trade({
                "symbol": symbol,
                "strategy_name": strategy_name,
                "side": side,
                "qty": qty,
                "entry_price": signal_price if signal_price > 0 else 0.0,
                "entry_time": datetime.now(timezone.utc),
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "signal_metadata": signal_metadata or {},
                "status": "OPEN",
                "was_day_trade": False,
                "alpaca_order_id": "DRY_RUN",
            })
            return trade.id if trade else None

        try:
            # Bug 2 fix: submit market entry + separate stop order instead of
            # bracket order. Bracket orders with 3bp stops cause near-100%
            # stop-outs due to Alpaca's price conflict validation.
            market_order_id = self.client.submit_market_order(symbol, qty, side)
            if market_order_id is None:
                logger.error("Market order submission returned None for %s", symbol)
                return None

            # Attach a stop-loss sell order immediately after the entry fill.
            try:
                self.client.submit_stop_order(symbol, qty, stop_price)
            except Exception as stop_exc:
                logger.warning(
                    "Stop order failed for %s after market fill: %s — position unprotected!",
                    symbol, str(stop_exc),
                )

            # Use signal_price as entry price; fall back to estimated from account.
            entry_price = signal_price
            if entry_price <= 0:
                account = self.client.get_account()
                entry_price = account.get("portfolio_value", 0) / max(qty, 1)

            trade = repository.save_trade({
                "symbol": symbol,
                "strategy_name": strategy_name,
                "side": side,
                "qty": qty,
                "entry_price": entry_price,
                "entry_time": datetime.now(timezone.utc),
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "highest_price_since_entry": entry_price,
                "trailing_stop_price": stop_price,
                "signal_metadata": signal_metadata or {},
                "status": "OPEN",
                "was_day_trade": False,
                "alpaca_order_id": market_order_id,
            })
            logger.info(
                "Entry order filled: %s %d %s @ est. $%.2f (trade_id=%d, stop=%.2f)",
                side, qty, symbol, entry_price, trade.id, stop_price,
            )
            return trade.id

        except Exception as e:
            logger.error(
                "Failed to submit entry order for %s: %s", symbol, str(e)
            )
            return None

    def submit_exit_order(
        self,
        trade_id: int,
        symbol: str,
        qty: int,
        exit_reason: str,
        is_day_trade: bool = False,
    ) -> bool:
        """Submit an exit order and update the trade record.

        Args:
            trade_id: Database ID of the trade to close.
            symbol: Ticker symbol.
            qty: Number of shares to sell.
            exit_reason: Reason for exit (STOP, TRAIL, SIGNAL, TIME, CIRCUIT_BREAKER).
            is_day_trade: Whether this exit constitutes a day trade.

        Returns:
            bool: True if the exit was successful, False otherwise.

        Example:
            >>> success = om.submit_exit_order(1, "AAPL", 3, "SIGNAL")
        """
        if self.dry_run:
            logger.info(
                "[DRY RUN] Would exit %d shares of %s (reason=%s)",
                qty,
                symbol,
                exit_reason,
            )
            # Try to get current price even in dry_run (paper positions exist)
            exit_price = None
            try:
                positions = self.client.get_positions()
                for pos in positions:
                    if pos["symbol"] == symbol:
                        exit_price = pos["current_price"]
                        break
            except Exception:
                pass
            update = {
                "status": "CLOSED",
                "exit_reason": exit_reason,
                "exit_time": datetime.now(timezone.utc),
                "realized_pnl": 0.0,
                "was_day_trade": is_day_trade,
            }
            if exit_price is not None and exit_price > 0:
                update["exit_price"] = exit_price
            repository.update_trade(trade_id, update)
            return True

        try:
            # 1. Get current price BEFORE selling (position disappears after sell)
            positions = self.client.get_positions()
            exit_price = 0.0
            broker_qty = 0
            for pos in positions:
                if pos["symbol"] == symbol:
                    exit_price = pos["current_price"]
                    broker_qty = pos["qty"]
                    break

            # 2. Cancel existing orders for this symbol (bracket stop/TP legs
            #    hold shares and block new sell orders)
            try:
                self.client.cancel_orders_for_symbol(symbol)
            except Exception as cancel_exc:
                logger.warning(
                    "Could not cancel existing orders for %s before exit: %s",
                    symbol,
                    str(cancel_exc),
                )

            # 3. Use actual broker qty if it differs from internal tracker
            actual_qty = broker_qty if broker_qty > 0 else qty
            if broker_qty > 0 and broker_qty != qty:
                logger.warning(
                    "Qty mismatch for %s: internal=%d, broker=%d — using broker qty",
                    symbol,
                    qty,
                    broker_qty,
                )

            order_id = self.client.submit_market_order(symbol, actual_qty, "sell")
            if order_id is None:
                logger.error("Exit order submission returned None for %s", symbol)
                return False

            trade = repository._get_trade_by_id(trade_id)
            entry_price = float(trade.entry_price) if trade else 0.0
            realized_pnl = round((exit_price - entry_price) * actual_qty, 6)

            repository.update_trade(trade_id, {
                "status": "CLOSED",
                "exit_reason": exit_reason,
                "exit_time": datetime.now(timezone.utc),
                "exit_price": exit_price,
                "realized_pnl": realized_pnl,
                "was_day_trade": is_day_trade,
            })

            if is_day_trade:
                from src.risk.pdt_guard import get_pdt_guard
                pdt = get_pdt_guard()
                pdt.record_day_trade(symbol)

            logger.info(
                "Exit order filled: sell %d %s @ $%.2f (pnl=$%.2f, reason=%s)",
                actual_qty,
                symbol,
                exit_price,
                realized_pnl,
                exit_reason,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to submit exit order for %s: %s", symbol, str(e)
            )
            return False

    def emergency_liquidate(self) -> int:
        """Emergency liquidation of all positions (circuit breaker).

        Bypasses all risk checks and PDT guard. Cancels all open orders
        first, then liquidates all positions.

        Returns:
            int: Number of positions liquidated.

        Example:
            >>> count = om.emergency_liquidate()
            >>> count
            3
        """
        logger.critical("EMERGENCY LIQUIDATION initiated")

        if self.dry_run:
            logger.critical("[DRY RUN] Would liquidate all positions")
            open_trades = repository.get_open_trades()
            for trade in open_trades:
                repository.update_trade(trade.id, {
                    "status": "CLOSED",
                    "exit_reason": "CIRCUIT_BREAKER",
                    "exit_time": datetime.now(timezone.utc),
                    "exit_price": 0.0,
                    "realized_pnl": 0.0,
                })
            return len(open_trades)

        try:
            self.client.cancel_all_orders()
            count = self.client.liquidate_all_positions()

            open_trades = repository.get_open_trades()
            for trade in open_trades:
                repository.update_trade(trade.id, {
                    "status": "CLOSED",
                    "exit_reason": "CIRCUIT_BREAKER",
                    "exit_time": datetime.now(timezone.utc),
                    "was_day_trade": False,
                })

            logger.critical("Emergency liquidation complete: %d positions closed", count)
            return count

        except Exception as e:
            logger.critical("Emergency liquidation FAILED: %s", str(e))
            return 0

    def sync_positions_with_broker(self) -> None:
        """Synchronize local trade records with actual broker positions.

        Fetches current positions from Alpaca and updates local trade
        records with current prices and unrealized P&L.

        Example:
            >>> om.sync_positions_with_broker()
        """
        try:
            positions = self.client.get_positions()
            position_map = {p["symbol"]: p for p in positions}

            open_trades = repository.get_open_trades()
            for trade in open_trades:
                if trade.symbol in position_map:
                    pos = position_map[trade.symbol]
                    current_price = pos["current_price"]
                    highest = max(
                        float(trade.highest_price_since_entry or 0),
                        current_price,
                    )
                    repository.update_trade(trade.id, {
                        "highest_price_since_entry": highest,
                    })
                else:
                    logger.warning(
                        "Trade %d (%s) has no matching broker position",
                        trade.id,
                        trade.symbol,
                    )

            logger.info(
                "Position sync: %d trades, %d broker positions",
                len(open_trades),
                len(positions),
            )

        except Exception as e:
            logger.error("Position sync failed: %s", str(e))
