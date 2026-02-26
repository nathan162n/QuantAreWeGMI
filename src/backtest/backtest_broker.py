"""Simulated order broker for backtest mode.

Applies 0.05% slippage each way on fills. Tracks portfolio equity.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("backtest.broker")


@dataclass
class SimulatedFill:
    """Result of a simulated order fill.

    Attributes:
        symbol: Ticker symbol.
        side: 'buy' or 'sell'.
        requested_price: Price before slippage.
        fill_price: Actual fill price with slippage applied.
        qty: Number of shares.
        slippage_cost: Total slippage cost in dollars.
        timestamp: Fill timestamp.
    """

    symbol: str
    side: str
    requested_price: float
    fill_price: float
    qty: int
    slippage_cost: float
    timestamp: datetime


class BacktestBroker:
    """Simulated order execution for backtesting.

    Applies 0.05% slippage on buys (fill at price * 1.0005) and
    sells (fill at price * 0.9995). Tracks portfolio equity and
    open positions.

    Args:
        initial_equity: Starting capital. Default 500.0.
        slippage_pct: Slippage percentage per side. Default 0.0005 (0.05%).
    """

    def __init__(self, initial_equity: float = 500.0, slippage_pct: float = 0.0005):
        self._initial_equity = initial_equity
        self._cash: float = initial_equity
        self._slippage_pct = slippage_pct
        # Keyed as "{symbol}_{layer_name}"
        self._open_positions: Dict[str, dict] = {}
        self._trade_history: List[dict] = []

    def submit_buy(
        self,
        symbol: str,
        qty: int,
        price: float,
        timestamp: datetime,
        layer_name: str = "",
        stop_price: float = 0.0,
    ) -> Optional[SimulatedFill]:
        """Submit a buy order with slippage.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares.
            price: Current market price.
            timestamp: Current bar timestamp.
            layer_name: Strategy layer name for tracking.
            stop_price: Stop loss price for this position.

        Returns:
            Optional[SimulatedFill]: Fill result or None if insufficient buying power.
        """
        if qty <= 0:
            logger.debug("submit_buy: qty=%d <= 0 for %s, skipping", qty, symbol)
            return None

        fill_price = price * (1.0 + self._slippage_pct)
        cost = fill_price * qty

        if cost > self._cash:
            logger.debug(
                "submit_buy: insufficient cash for %s — need $%.2f, have $%.2f",
                symbol,
                cost,
                self._cash,
            )
            return None

        slippage_cost = (fill_price - price) * qty
        self._cash -= cost

        pos_key = f"{symbol}_{layer_name}" if layer_name else symbol
        self._open_positions[pos_key] = {
            "symbol": symbol,
            "layer_name": layer_name,
            "entry_price": price,
            "fill_price": fill_price,
            "qty": qty,
            "timestamp": timestamp,
            "entry_time": timestamp,
            "stop_price": stop_price,
            "current_price": price,
        }

        logger.debug(
            "BUY filled: %s qty=%d @ $%.4f (slip $%.4f) | cash=$%.2f",
            pos_key,
            qty,
            fill_price,
            slippage_cost,
            self._cash,
        )

        return SimulatedFill(
            symbol=symbol,
            side="buy",
            requested_price=price,
            fill_price=fill_price,
            qty=qty,
            slippage_cost=slippage_cost,
            timestamp=timestamp,
        )

    def submit_sell(
        self,
        symbol: str,
        qty: int,
        price: float,
        timestamp: datetime,
        exit_reason: str = "SIGNAL",
    ) -> Optional[SimulatedFill]:
        """Submit a sell order with slippage.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares.
            price: Current market price.
            timestamp: Current bar timestamp.
            exit_reason: Reason for exit (SIGNAL, STOP, EOD, etc.).

        Returns:
            Optional[SimulatedFill]: Fill result or None if no open position.
        """
        # Find the first open position matching this symbol
        pos_key = None
        for key, pos in self._open_positions.items():
            if pos.get("symbol") == symbol:
                pos_key = key
                break

        if pos_key is None:
            logger.debug("submit_sell: no open position found for %s", symbol)
            return None

        pos = self._open_positions[pos_key]
        sell_qty = min(qty, pos["qty"])

        fill_price = price * (1.0 - self._slippage_pct)
        entry_fill_price = pos["fill_price"]

        # Slippage on both legs
        buy_slippage = (entry_fill_price - pos["entry_price"]) * sell_qty
        sell_slippage = (price - fill_price) * sell_qty
        total_slippage = buy_slippage + sell_slippage

        gross_pnl = (fill_price - entry_fill_price) * sell_qty
        pnl = gross_pnl  # slippage already baked into fill prices

        proceeds = fill_price * sell_qty
        self._cash += proceeds

        entry_time = pos.get("entry_time") or pos.get("timestamp")
        if entry_time is None:
            entry_time = timestamp

        hold_minutes = 0.0
        if isinstance(entry_time, datetime) and isinstance(timestamp, datetime):
            delta = timestamp - entry_time
            hold_minutes = delta.total_seconds() / 60.0

        trade_record = {
            "symbol": symbol,
            "layer_name": pos.get("layer_name", ""),
            "entry_time": entry_time,
            "exit_time": timestamp,
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "fill_entry_price": entry_fill_price,
            "fill_exit_price": fill_price,
            "qty": sell_qty,
            "pnl": round(pnl, 4),
            "slippage": round(total_slippage, 4),
            "exit_reason": exit_reason,
            "hold_minutes": round(hold_minutes, 2),
        }
        self._trade_history.append(trade_record)

        del self._open_positions[pos_key]

        logger.debug(
            "SELL filled: %s qty=%d @ $%.4f | pnl=$%.4f | cash=$%.2f",
            pos_key,
            sell_qty,
            fill_price,
            pnl,
            self._cash,
        )

        return SimulatedFill(
            symbol=symbol,
            side="sell",
            requested_price=price,
            fill_price=fill_price,
            qty=sell_qty,
            slippage_cost=sell_slippage,
            timestamp=timestamp,
        )

    @property
    def equity(self) -> float:
        """Current portfolio equity (cash + open position market values)."""
        open_value = sum(
            pos.get("current_price", pos.get("fill_price", 0.0)) * pos["qty"]
            for pos in self._open_positions.values()
        )
        return self._cash + open_value

    @property
    def cash(self) -> float:
        """Available cash."""
        return self._cash

    def get_open_positions(self) -> Dict[str, dict]:
        """Return dict of open positions keyed by symbol_layername."""
        return dict(self._open_positions)

    def get_trade_history(self) -> List[dict]:
        """Return list of all closed trade dicts."""
        return list(self._trade_history)

    def update_prices(self, symbol: str, current_price: float) -> None:
        """Update the current price for an open position (for equity calc)."""
        for pos in self._open_positions.values():
            if pos.get("symbol") == symbol:
                pos["current_price"] = current_price

    def force_close_all(
        self,
        prices: Dict[str, float],
        timestamp: datetime,
        reason: str = "EOD",
    ) -> List[SimulatedFill]:
        """Close all open positions at given prices (EOD forced close).

        Args:
            prices: Dict[symbol, current_price].
            timestamp: Current timestamp.
            reason: Exit reason. Default "EOD".

        Returns:
            List[SimulatedFill]: All fill results.
        """
        fills = []
        # Snapshot keys to avoid mutation during iteration
        keys = list(self._open_positions.keys())
        for key in keys:
            if key not in self._open_positions:
                continue
            pos = self._open_positions[key]
            sym = pos["symbol"]
            price = prices.get(sym, pos.get("current_price", pos.get("fill_price", 0.0)))
            if price <= 0:
                logger.warning("force_close_all: no valid price for %s, skipping", sym)
                continue
            fill = self.submit_sell(sym, pos["qty"], price, timestamp, reason)
            if fill is not None:
                fills.append(fill)

        logger.info(
            "force_close_all: closed %d positions (reason=%s) | equity=$%.2f",
            len(fills),
            reason,
            self.equity,
        )
        return fills
