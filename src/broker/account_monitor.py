"""AccountMonitor: polls Alpaca account state every 10 seconds.
Provides always-current equity, buying power, positions, and daily P&L.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger("algotrader.account_monitor")


class AccountMonitor:

    def __init__(self, alpaca_client, poll_interval_seconds: int = 10):
        self._client = alpaca_client
        self._interval = poll_interval_seconds
        self._task = None
        self.equity = 0.0
        self.portfolio_value = 0.0
        self.buying_power = 0.0
        self.cash = 0.0
        self.daytrade_count = 0
        self.open_positions = []
        self.equity_at_open = 0.0
        self.daily_pnl = 0.0
        self.daily_pnl_pct = 0.0
        self.last_updated = None
        self._session_started = False

    async def start(self):
        await self._poll()
        self.equity_at_open = self.equity
        self._session_started = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"AccountMonitor started — equity=${self.equity:,.2f}, "
            f"buying_power=${self.buying_power:,.2f}, "
            f"positions={self.open_position_count}, "
            f"day_trades={self.daytrade_count}"
        )

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        while True:
            await asyncio.sleep(self._interval)
            await self._poll()

    async def _poll(self):
        try:
            account = self._client.get_account()
            positions = self._client.get_positions()
            self.equity = float(account["equity"])
            self.portfolio_value = float(account["portfolio_value"])
            self.buying_power = float(account["buying_power"])
            self.cash = float(account["cash"])
            self.daytrade_count = int(account.get("daytrade_count", 0))
            self.open_positions = [
                {
                    "symbol": p["symbol"],
                    "qty": int(p["qty"]),
                    "side": p.get("side", "long"),
                    "market_value": float(p["market_value"]),
                    "avg_entry_price": float(p["avg_entry_price"]),
                    "unrealized_pl": float(p["unrealized_pl"]),
                    "unrealized_plpc": float(p["unrealized_plpc"]),
                    "current_price": float(p["current_price"]),
                }
                for p in positions
            ]
            if self._session_started and self.equity_at_open:
                self.daily_pnl = self.equity - self.equity_at_open
                self.daily_pnl_pct = self.daily_pnl / self.equity_at_open
            self.last_updated = datetime.utcnow()
        except Exception as e:
            logger.warning(f"AccountMonitor poll failed — keeping last values: {e}")

    def get_position(self, symbol):
        return next((p for p in self.open_positions if p["symbol"] == symbol), None)

    def get_unrealized_pnl(self, symbol):
        pos = self.get_position(symbol)
        return pos["unrealized_pl"] if pos else 0.0

    def get_short_symbols(self):
        return [p["symbol"] for p in self.open_positions if p["qty"] < 0]

    @property
    def open_position_count(self):
        return len(self.open_positions)

    @property
    def is_over_daily_loss_limit(self):
        try:
            from src.utils.config_loader import load_config
            limit = load_config()["risk"]["max_daily_loss_dollars"]
            return self.daily_pnl < -abs(limit)
        except Exception:
            return False
