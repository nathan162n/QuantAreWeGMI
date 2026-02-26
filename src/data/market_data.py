"""Async multi-ticker market data fetcher with caching.

Provides concurrent data fetching with semaphore-based concurrency control,
in-memory caching with 60-second TTL, and exponential backoff on errors.
"""

import asyncio
import time
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger("data.market_data")

_cache: Dict[str, dict] = {}
_CACHE_TTL_SECONDS = 60
_SEMAPHORE_LIMIT = 10


def _cache_key(symbol: str, timeframe: str, limit: int) -> str:
    """Generate a cache key for bar data.

    Args:
        symbol: Ticker symbol.
        timeframe: Bar timeframe (e.g. '1Day').
        limit: Number of bars.

    Returns:
        str: Cache key string.

    Example:
        >>> _cache_key("AAPL", "1Day", 60)
        'bars:AAPL:1Day:60'
    """
    return f"bars:{symbol}:{timeframe}:{limit}"


def _is_cache_valid(key: str) -> bool:
    """Check if a cached entry is still within its TTL.

    Args:
        key: Cache key to check.

    Returns:
        bool: True if the cache entry exists and is not expired.

    Example:
        >>> _is_cache_valid("bars:AAPL:1Day:60")
        True
    """
    if key not in _cache:
        return False
    return (time.time() - _cache[key]["timestamp"]) < _CACHE_TTL_SECONDS


def _get_alpaca_client():
    """Lazy import of the Alpaca client to avoid circular imports.

    Returns:
        AlpacaClient: The initialized Alpaca client instance.

    Example:
        >>> client = _get_alpaca_client()
    """
    from src.broker.alpaca_client import get_client
    return get_client()


async def fetch_bars_bulk(
    symbols: List[str],
    timeframe: str = "1Day",
    limit: int = 60,
) -> Dict[str, pd.DataFrame]:
    """Fetch historical bars for multiple symbols concurrently.

    Uses a semaphore to limit concurrent requests to 10 simultaneous.
    Results are cached with a 60-second TTL.

    Args:
        symbols: List of ticker symbols to fetch.
        timeframe: Bar timeframe ('1Day', '1Hour', '5Min'). Default '1Day'.
        limit: Number of bars per symbol. Default 60.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of symbol to DataFrame with columns
            [open, high, low, close, volume, vwap] and UTC datetime index.
            Symbols that failed to fetch are omitted from the result.

    Example:
        >>> bars = await fetch_bars_bulk(["AAPL", "MSFT", "GOOGL"], "1Day", 60)
        >>> bars["AAPL"].columns.tolist()
        ['open', 'high', 'low', 'close', 'volume', 'vwap']
    """
    semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    results: Dict[str, pd.DataFrame] = {}

    async def _fetch_one(symbol: str) -> None:
        """Fetch bars for a single symbol with semaphore control.

        Args:
            symbol: Ticker symbol to fetch.
        """
        async with semaphore:
            cache_k = _cache_key(symbol, timeframe, limit)
            if _is_cache_valid(cache_k):
                results[symbol] = _cache[cache_k]["data"]
                return
            try:
                df = await _fetch_bars_from_alpaca(symbol, timeframe, limit)
                if df is not None and not df.empty:
                    _cache[cache_k] = {"data": df, "timestamp": time.time()}
                    results[symbol] = df
            except Exception as e:
                logger.error("Failed to fetch bars for %s: %s", symbol, str(e))

    tasks = [_fetch_one(s) for s in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(
        "Fetched bars for %d/%d symbols (timeframe=%s, limit=%d)",
        len(results),
        len(symbols),
        timeframe,
        limit,
    )
    return results


async def _fetch_bars_from_alpaca(
    symbol: str, timeframe: str, limit: int
) -> Optional[pd.DataFrame]:
    """Fetch bar data from Alpaca API for a single symbol.

    Runs the synchronous Alpaca SDK call in a thread executor
    to avoid blocking the event loop.

    Args:
        symbol: Ticker symbol to fetch.
        timeframe: Bar timeframe string.
        limit: Number of bars to request.

    Returns:
        Optional[pd.DataFrame]: DataFrame of bar data, or None on failure.

    Example:
        >>> df = await _fetch_bars_from_alpaca("AAPL", "1Day", 60)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _fetch_bars_sync, symbol, timeframe, limit
    )


@retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(Exception,))
def _fetch_bars_sync(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Synchronous bar fetch from Alpaca with retry.

    Args:
        symbol: Ticker symbol.
        timeframe: Bar timeframe.
        limit: Number of bars.

    Returns:
        Optional[pd.DataFrame]: Bar data DataFrame or None on failure.

    Example:
        >>> df = _fetch_bars_sync("AAPL", "1Day", 60)
    """
    client = _get_alpaca_client()
    bars = client.get_bars(symbol, timeframe, limit)
    if bars is None or bars.empty:
        return None
    return bars


async def fetch_latest_quotes(
    symbols: List[str],
) -> Dict[str, dict]:
    """Fetch latest bid/ask quotes for multiple symbols.

    Used by the spread filter in the universe screener.

    Args:
        symbols: List of ticker symbols.

    Returns:
        Dict[str, dict]: Mapping of symbol to quote dict with keys
            'bid', 'ask', 'bid_size', 'ask_size'.

    Example:
        >>> quotes = await fetch_latest_quotes(["AAPL", "MSFT"])
        >>> quotes["AAPL"]["bid"]
        150.25
    """
    semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    results: Dict[str, dict] = {}

    async def _fetch_one(symbol: str) -> None:
        """Fetch quote for a single symbol.

        Args:
            symbol: Ticker symbol.
        """
        async with semaphore:
            try:
                quote = await _fetch_quote_from_alpaca(symbol)
                if quote is not None:
                    results[symbol] = quote
            except Exception as e:
                logger.error("Failed to fetch quote for %s: %s", symbol, str(e))

    tasks = [_fetch_one(s) for s in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def _fetch_quote_from_alpaca(symbol: str) -> Optional[dict]:
    """Fetch latest quote from Alpaca for a single symbol.

    Args:
        symbol: Ticker symbol.

    Returns:
        Optional[dict]: Quote dict or None on failure.

    Example:
        >>> quote = await _fetch_quote_from_alpaca("AAPL")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_quote_sync, symbol)


@retry_with_backoff(max_retries=2, base_delay=0.5)
def _fetch_quote_sync(symbol: str) -> Optional[dict]:
    """Synchronous quote fetch from Alpaca with retry.

    Args:
        symbol: Ticker symbol.

    Returns:
        Optional[dict]: Quote data or None.

    Example:
        >>> _fetch_quote_sync("AAPL")
        {'bid': 150.25, 'ask': 150.30, 'bid_size': 100, 'ask_size': 200}
    """
    client = _get_alpaca_client()
    return client.get_latest_quote(symbol)


async def fetch_single_bar(
    symbol: str, timeframe: str = "1Day", limit: int = 200
) -> Optional[pd.DataFrame]:
    """Fetch bar data for a single symbol (e.g. SPY or VIXY for regime).

    Args:
        symbol: Ticker symbol.
        timeframe: Bar timeframe. Default '1Day'.
        limit: Number of bars. Default 200 (for 200-day SMA).

    Returns:
        Optional[pd.DataFrame]: Bar data or None on failure.

    Example:
        >>> spy_bars = await fetch_single_bar("SPY", "1Day", 200)
    """
    result = await fetch_bars_bulk([symbol], timeframe, limit)
    return result.get(symbol)


def clear_cache() -> None:
    """Clear the in-memory bar data cache.

    Example:
        >>> clear_cache()
    """
    global _cache
    _cache.clear()
    logger.info("Market data cache cleared")
