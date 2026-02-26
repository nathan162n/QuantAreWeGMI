"""Alpaca API wrapper providing unified access to REST and WebSocket.

Single-responsibility wrapper that isolates all Alpaca SDK usage.
No other module in the system imports alpaca-py directly.
Includes an internal token bucket rate limiter (200 req/min).
"""

import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger("broker.alpaca_client")

_client_instance: Optional["AlpacaClient"] = None


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter for API request throttling.

    Implements the token bucket algorithm to enforce a maximum request
    rate of 200 requests per minute to the Alpaca API.

    Attributes:
        max_tokens: Maximum number of tokens (requests) in the bucket.
        refill_rate: Tokens added per second.
        tokens: Current number of available tokens.
    """

    def __init__(self, max_tokens: int = 200, refill_period: float = 60.0):
        """Initialize the rate limiter.

        Args:
            max_tokens: Maximum burst size. Default 200.
            refill_period: Period in seconds to fully refill. Default 60.0.

        Example:
            >>> limiter = TokenBucketRateLimiter(200, 60.0)
        """
        self.max_tokens = max_tokens
        self.refill_rate = max_tokens / refill_period
        self.tokens = float(max_tokens)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire a token, blocking until one is available or timeout.

        Args:
            timeout: Maximum seconds to wait. Default 30.0.

        Returns:
            bool: True if a token was acquired, False if timed out.

        Example:
            >>> limiter = TokenBucketRateLimiter()
            >>> limiter.acquire()
            True
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill.

        Example:
            >>> limiter = TokenBucketRateLimiter()
            >>> limiter._refill()  # Internal use
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


class AlpacaClient:
    """Unified Alpaca API client for trading operations and market data.

    Wraps the alpaca-py SDK to provide a clean interface for account
    management, order submission, position tracking, and market data.
    All API calls pass through an internal rate limiter.

    Attributes:
        api_key: Alpaca API key.
        secret_key: Alpaca secret key.
        base_url: Alpaca API base URL.
        trading_client: Alpaca TradingClient instance.
        data_client: Alpaca StockHistoricalDataClient instance.
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        base_url: str = "",
    ):
        """Initialize the Alpaca client from provided or environment credentials.

        Args:
            api_key: Alpaca API key. Falls back to ALPACA_API_KEY env var.
            secret_key: Alpaca secret key. Falls back to ALPACA_SECRET_KEY env var.
            base_url: Alpaca base URL. Falls back to ALPACA_BASE_URL env var.

        Example:
            >>> client = AlpacaClient()  # Uses env vars
        """
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = base_url or os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self._rate_limiter = TokenBucketRateLimiter(max_tokens=200, refill_period=60.0)

        self.trading_client = None
        self.data_client = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-initialize Alpaca SDK clients on first use.

        Example:
            >>> client = AlpacaClient()
            >>> client._ensure_initialized()
        """
        if self._initialized:
            return
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.is_paper_trading(),
            )
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            self._initialized = True
            logger.info(
                "Alpaca client initialized (paper=%s)", self.is_paper_trading()
            )
        except Exception as e:
            logger.error("Failed to initialize Alpaca client: %s", str(e))
            raise

    def _throttle(self) -> None:
        """Acquire a rate limiter token before making an API call.

        Example:
            >>> client = AlpacaClient()
            >>> client._throttle()
        """
        if not self._rate_limiter.acquire(timeout=30.0):
            logger.warning("Rate limiter timeout - proceeding anyway")

    def is_paper_trading(self) -> bool:
        """Check if the client is configured for paper trading.

        Derived from the base URL containing 'paper'.

        Returns:
            bool: True if using paper trading endpoint.

        Example:
            >>> client = AlpacaClient(base_url="https://paper-api.alpaca.markets")
            >>> client.is_paper_trading()
            True
        """
        return "paper" in self.base_url.lower()

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_account(self) -> dict:
        """Fetch account information including equity, buying power, and PDT count.

        Returns:
            dict: Account data with keys: equity, buying_power, portfolio_value,
                cash, daytrade_count, pattern_day_trader, trading_blocked.

        Example:
            >>> account = client.get_account()
            >>> account["equity"]
            500.0
        """
        self._ensure_initialized()
        self._throttle()
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "daytrade_count": int(account.daytrade_count),
            "pattern_day_trader": bool(account.pattern_day_trader),
            "trading_blocked": bool(account.trading_blocked),
        }

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_positions(self) -> List[dict]:
        """Fetch all currently open positions.

        Returns:
            List[dict]: List of position dicts with keys: symbol, qty,
                avg_entry_price, current_price, market_value, unrealized_pl,
                unrealized_plpc.

        Example:
            >>> positions = client.get_positions()
            >>> positions[0]["symbol"]
            'AAPL'
        """
        self._ensure_initialized()
        self._throttle()
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def submit_market_order(self, symbol: str, qty: int, side: str) -> Optional[str]:
        """Submit a market order with paper trading safety assertion.

        Args:
            symbol: Ticker symbol to trade.
            qty: Number of shares.
            side: 'buy' or 'sell'.

        Returns:
            Optional[str]: Alpaca order ID on success, None on failure.

        Example:
            >>> order_id = client.submit_market_order("AAPL", 5, "buy")
        """
        self._ensure_initialized()
        self._throttle()

        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = self.trading_client.submit_order(request)
        logger.info(
            "Submitted market order: %s %d %s (order_id=%s)",
            side,
            qty,
            symbol,
            order.id,
        )
        return str(order.id)

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_price: float,
        take_profit_price: float,
    ) -> Optional[str]:
        """Submit a bracket order with stop loss and take profit.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares.
            side: 'buy' or 'sell'.
            stop_price: Stop loss price.
            take_profit_price: Take profit price.

        Returns:
            Optional[str]: Alpaca order ID on success, None on failure.

        Example:
            >>> order_id = client.submit_bracket_order("AAPL", 5, "buy", 145.0, 160.0)
        """
        self._ensure_initialized()
        self._throttle()

        from alpaca.trading.requests import (
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
        )

        order = self.trading_client.submit_order(request)
        logger.info(
            "Submitted bracket order: %s %d %s (stop=%.2f, tp=%.2f, order_id=%s)",
            side,
            qty,
            symbol,
            stop_price,
            take_profit_price,
            order.id,
        )
        return str(order.id)

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def submit_stop_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
    ) -> Optional[str]:
        """Submit a stop (stop-market) sell order to protect a long position.

        Used as the protective leg after a market entry order is filled.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares to sell on stop trigger.
            stop_price: Stop trigger price.

        Returns:
            Optional[str]: Alpaca order ID on success, None on failure.

        Example:
            >>> order_id = client.submit_stop_order("AAPL", 5, 145.0)
        """
        self._ensure_initialized()
        self._throttle()

        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        request = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            stop_price=round(stop_price, 2),
        )

        order = self.trading_client.submit_order(request)
        logger.info(
            "Submitted stop order: sell %d %s @ stop=%.2f (order_id=%s)",
            qty,
            symbol,
            stop_price,
            order.id,
        )
        return str(order.id)

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def cancel_orders_for_symbol(self, symbol: str) -> int:
        """Cancel all open orders for a specific symbol.

        Used before submitting an exit market order to release shares
        held by bracket order children (stop-loss / take-profit legs).

        Args:
            symbol: Ticker symbol whose orders should be cancelled.

        Returns:
            int: Number of orders cancelled.
        """
        self._ensure_initialized()
        self._throttle()

        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
        )
        open_orders = self.trading_client.get_orders(filter=request)
        cancelled = 0
        for order in open_orders:
            try:
                self.trading_client.cancel_order_by_id(order.id)
                cancelled += 1
            except Exception as exc:
                logger.warning(
                    "Failed to cancel order %s for %s: %s",
                    order.id,
                    symbol,
                    str(exc),
                )
        if cancelled:
            logger.info("Cancelled %d open orders for %s", cancelled, symbol)
        return cancelled

    def get_position_qty(self, symbol: str) -> int:
        """Get the actual broker-side position quantity for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            int: Number of shares held, or 0 if no position exists.
        """
        positions = self.get_positions()
        for pos in positions:
            if pos["symbol"] == symbol:
                return pos["qty"]
        return 0

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def cancel_all_orders(self) -> int:
        """Cancel all open orders.

        Used by the circuit breaker during emergency shutdown.

        Returns:
            int: Number of orders cancelled.

        Example:
            >>> cancelled = client.cancel_all_orders()
            >>> cancelled
            3
        """
        self._ensure_initialized()
        self._throttle()
        statuses = self.trading_client.cancel_orders()
        count = len(statuses) if statuses else 0
        logger.info("Cancelled %d open orders", count)
        return count

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def liquidate_all_positions(self) -> int:
        """Market sell all open positions immediately.

        Used by the circuit breaker. Bypasses all risk checks.

        Returns:
            int: Number of positions liquidated.

        Example:
            >>> liquidated = client.liquidate_all_positions()
        """
        self._ensure_initialized()
        self._throttle()
        result = self.trading_client.close_all_positions(cancel_orders=True)
        count = len(result) if result else 0
        logger.critical("Liquidated %d positions (emergency)", count)
        return count

    def get_day_trade_count(self) -> int:
        """Get the rolling 5-day PDT count from the Alpaca account.

        Returns:
            int: Number of day trades in the rolling window.

        Example:
            >>> client.get_day_trade_count()
            2
        """
        account = self.get_account()
        return account["daytrade_count"]

    def get_bars(
        self, symbol: str, timeframe: str = "1Day", limit: int = 60, start=None
    ) -> Optional[pd.DataFrame]:
        """Fetch historical bar data for a single symbol.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe ('1Day', '1Hour', '5Min'). Default '1Day'.
            limit: Number of bars. Default 60. Ignored when start is provided.
            start: Optional datetime to fetch bars from. When provided, fetches
                all bars from start to now instead of using limit.

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns [open, high, low, close,
                volume, vwap] and UTC datetime index, or None on failure.

        Example:
            >>> df = client.get_bars("AAPL", "1Day", 60)
            >>> df.columns.tolist()
            ['open', 'high', 'low', 'close', 'volume', 'vwap']
        """
        self._ensure_initialized()
        self._throttle()

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        tf_map = {
            "1Day": TimeFrame.Day,
            "1Hour": TimeFrame.Hour,
            "1Min": TimeFrame.Minute,
        }
        if timeframe == "5Min":
            try:
                tf = TimeFrame(5, TimeFrame.TimeFrameUnit.Minute)
            except (AttributeError, TypeError):
                tf = TimeFrame.Minute
        else:
            tf = tf_map.get(timeframe, TimeFrame.Day)

        if start is not None:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
            )
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                limit=limit,
            )

        bars = self.data_client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return None

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "vwap": "vwap",
        })

        needed_cols = ["open", "high", "low", "close", "volume", "vwap"]
        for col in needed_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df[needed_cols]

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """Fetch the latest bid/ask quote for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Optional[dict]: Quote dict with keys bid, ask, bid_size, ask_size.

        Example:
            >>> quote = client.get_latest_quote("AAPL")
            >>> quote["bid"]
            150.25
        """
        self._ensure_initialized()
        self._throttle()

        from alpaca.data.requests import StockLatestQuoteRequest

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self.data_client.get_stock_latest_quote(request)

        if symbol in quotes:
            q = quotes[symbol]
            return {
                "bid": float(q.bid_price),
                "ask": float(q.ask_price),
                "bid_size": int(q.bid_size),
                "ask_size": int(q.ask_size),
            }
        return None

    async def stream_bars(
        self, symbols: List[str], callback: Callable
    ) -> None:
        """Start a WebSocket stream for real-time bar updates.

        Args:
            symbols: List of symbols to subscribe to.
            callback: Async callable invoked with each bar update dict.

        Example:
            >>> async def on_bar(bar):
            ...     print(f"{bar['symbol']}: {bar['close']}")
            >>> await client.stream_bars(["AAPL", "MSFT"], on_bar)
        """
        try:
            from alpaca.data.live import StockDataStream

            stream = StockDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )

            async def _bar_handler(bar):
                """Handle incoming bar data from WebSocket.

                Args:
                    bar: Alpaca bar object.
                """
                bar_dict = {
                    "symbol": bar.symbol,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "timestamp": bar.timestamp,
                }
                await callback(bar_dict)

            stream.subscribe_bars(_bar_handler, *symbols)
            logger.info("Starting WebSocket stream for %d symbols", len(symbols))
            # stream.run() is synchronous and tries to start its own event loop,
            # which crashes inside an already-running asyncio loop.
            # _run_forever() is the underlying async coroutine — await it directly.
            await stream._run_forever()
        except Exception as e:
            logger.error("WebSocket stream error: %s", str(e))


def get_client() -> AlpacaClient:
    """Get or create the singleton AlpacaClient instance.

    Returns:
        AlpacaClient: The initialized client.

    Example:
        >>> client = get_client()
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = AlpacaClient()
    return _client_instance


def reset_client() -> None:
    """Reset the singleton client (for testing).

    Example:
        >>> reset_client()
    """
    global _client_instance
    _client_instance = None
