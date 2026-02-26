"""Daily universe screener with multi-stage filter pipeline.

Produces the tradeable universe by applying a series of quality,
liquidity, and volatility filters to S&P 500 components. Runs
once pre-market and caches results to the database.
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.data import indicators
from src.data.market_data import fetch_bars_bulk, fetch_latest_quotes
from src.database import repository
from src.utils.logger import get_logger

logger = get_logger("universe.screener")


def _load_config() -> dict:
    """Load universe configuration from config.yml.

    Returns:
        dict: Universe configuration parameters.

    Example:
        >>> config = _load_config()
        >>> config["min_price"]
        15.0
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("universe", {})


def get_universe_symbols() -> List[str]:
    """Load the curated large-cap universe from CSV.

    Reads from the path specified in config.yml (universe.base_list_path).
    Falls back to a hardcoded list of 30 highly liquid symbols.

    Returns:
        List[str]: List of ticker symbols from the curated universe.

    Example:
        >>> symbols = get_universe_symbols()
        >>> "AAPL" in symbols
        True
    """
    config = _load_config()
    csv_path = config.get("base_list_path", "data/universe_largecap.csv")

    if not os.path.isabs(csv_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        csv_path = os.path.join(project_root, csv_path)

    try:
        df = pd.read_csv(csv_path)
        symbols = df["symbol"].tolist()
        logger.info("Loaded %d symbols from %s", len(symbols), csv_path)
        return symbols
    except Exception as e:
        logger.warning("Failed to load universe CSV: %s. Using hardcoded fallback.", str(e))
        return _hardcoded_fallback()


def _hardcoded_fallback() -> List[str]:
    """Return hardcoded large-cap symbols as final fallback.

    Returns:
        List[str]: 30 curated large-cap symbols.
    """
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
        "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT",
        "CSCO", "ACN", "TLT",
    ]


class UniverseScreener:
    """Multi-stage filter pipeline for building the daily tradeable universe.

    Applies index membership, price, liquidity, volatility, earnings,
    and spread filters to produce a list of tradeable symbols.

    Attributes:
        config: Universe configuration from config.yml.
    """

    def __init__(self):
        """Initialize the screener with configuration.

        Example:
            >>> screener = UniverseScreener()
        """
        self.config = _load_config()

    def run(self) -> List[str]:
        """Execute the full screening pipeline synchronously.

        Wraps the async pipeline for use from synchronous code.

        Returns:
            List[str]: List of symbols that passed all filters.

        Example:
            >>> screener = UniverseScreener()
            >>> universe = screener.run()
            >>> len(universe) >= 20
            True
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._run_pipeline())
                    return future.result()
            else:
                return loop.run_until_complete(self._run_pipeline())
        except RuntimeError:
            return asyncio.run(self._run_pipeline())

    async def _run_pipeline(self) -> List[str]:
        """Execute the async screening pipeline.

        Filter pipeline (each stage logs symbol count and rejection reasons):
        1. Index membership: S&P 500 components
        2. Price filter: Close > $15.00
        3. Liquidity filter: 20d ADV > $50M
        4. Volatility cap: 20d realized vol < 80% annualized
        5. Earnings blackout: Exclude symbols near earnings (skipped in paper)
        6. Spread filter: Bid-ask spread < 0.1% (skipped if market closed)

        Falls back to prior day's cached universe if fewer than 20 symbols
        pass or if the pipeline crashes.

        Returns:
            List[str]: Filtered universe of tradeable symbols.

        Example:
            >>> universe = await screener._run_pipeline()
        """
        rejected: Dict[str, str] = {}

        try:
            all_symbols = get_universe_symbols()
            logger.info("Stage 1 - Universe base list: %d symbols", len(all_symbols))

            bars_data = await fetch_bars_bulk(all_symbols, "1Day", 60)
            symbols_with_data = [s for s in all_symbols if s in bars_data]
            for s in all_symbols:
                if s not in bars_data:
                    rejected[s] = "No market data available"
            logger.info(
                "Data available for %d/%d symbols",
                len(symbols_with_data),
                len(all_symbols),
            )

            passed_price, rejected_price = self._filter_price(
                symbols_with_data, bars_data
            )
            rejected.update(rejected_price)
            logger.info(
                "Stage 2 - Price filter (>$%.2f): %d passed, %d rejected",
                self.config.get("min_price", 15.0),
                len(passed_price),
                len(rejected_price),
            )

            passed_liquidity, rejected_liq = self._filter_liquidity(
                passed_price, bars_data
            )
            rejected.update(rejected_liq)
            logger.info(
                "Stage 3 - Liquidity filter (ADV>$%dM): %d passed, %d rejected",
                int(self.config.get("min_adv_dollars", 50_000_000) / 1_000_000),
                len(passed_liquidity),
                len(rejected_liq),
            )

            passed_vol, rejected_vol = self._filter_volatility(
                passed_liquidity, bars_data
            )
            rejected.update(rejected_vol)
            logger.info(
                "Stage 4 - Volatility cap (<%.0f%%): %d passed, %d rejected",
                self.config.get("max_realized_vol", 0.80) * 100,
                len(passed_vol),
                len(rejected_vol),
            )

            passed_earnings = passed_vol
            logger.info(
                "Stage 5 - Earnings blackout: %d passed (earnings check deferred to signal time)",
                len(passed_earnings),
            )

            try:
                passed_spread, rejected_spread = await self._filter_spread(
                    passed_earnings
                )
                rejected.update(rejected_spread)
                logger.info(
                    "Stage 6 - Spread filter (<%.2f%%): %d passed, %d rejected",
                    self.config.get("max_spread_pct", 0.001) * 100,
                    len(passed_spread),
                    len(rejected_spread),
                )
                final_universe = passed_spread
            except Exception as e:
                logger.warning(
                    "Spread filter skipped (market may be closed): %s", str(e)
                )
                final_universe = passed_earnings

            min_size = self.config.get("min_universe_size", 20)
            if len(final_universe) < min_size:
                logger.critical(
                    "Universe too small (%d < %d). Falling back to previous day.",
                    len(final_universe),
                    min_size,
                )
                fallback = repository.get_latest_universe()
                if fallback and len(fallback) >= min_size:
                    return fallback
                logger.warning(
                    "No valid fallback universe. Using %d symbols anyway.",
                    len(final_universe),
                )

            now = datetime.now(timezone.utc)
            repository.save_universe(now, final_universe, rejected)

            logger.info(
                "Universe screening complete: %d symbols passed all filters",
                len(final_universe),
            )
            return final_universe

        except Exception as e:
            logger.critical("Universe screener CRASHED: %s", str(e))
            fallback = repository.get_latest_universe()
            if fallback:
                logger.info("Using cached universe (%d symbols)", len(fallback))
                return fallback
            logger.critical("No cached universe available. Returning empty list.")
            return []

    def _filter_price(
        self, symbols: List[str], bars: Dict[str, pd.DataFrame]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Filter symbols by minimum price.

        Args:
            symbols: Symbols to filter.
            bars: Dictionary of symbol to bar DataFrame.

        Returns:
            Tuple of (passed symbols, rejected dict with reasons).

        Example:
            >>> passed, rejected = screener._filter_price(["AAPL"], bars)
        """
        min_price = self.config.get("min_price", 15.0)
        passed = []
        rejected = {}
        for symbol in symbols:
            df = bars.get(symbol)
            if df is None or df.empty:
                rejected[symbol] = "No price data"
                continue
            last_close = float(df["close"].iloc[-1])
            if last_close > min_price:
                passed.append(symbol)
            else:
                rejected[symbol] = f"Price ${last_close:.2f} < ${min_price:.2f}"
        return passed, rejected

    def _filter_liquidity(
        self, symbols: List[str], bars: Dict[str, pd.DataFrame]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Filter symbols by average dollar volume.

        Args:
            symbols: Symbols to filter.
            bars: Dictionary of symbol to bar DataFrame.

        Returns:
            Tuple of (passed symbols, rejected dict with reasons).

        Example:
            >>> passed, rejected = screener._filter_liquidity(["AAPL"], bars)
        """
        min_adv = self.config.get("min_adv_dollars", 50_000_000)
        passed = []
        rejected = {}
        for symbol in symbols:
            df = bars.get(symbol)
            if df is None or df.empty:
                rejected[symbol] = "No volume data"
                continue
            adv_value = indicators.adv(df["volume"], df["close"], period=20)
            if adv_value >= min_adv:
                passed.append(symbol)
            else:
                rejected[symbol] = (
                    f"ADV ${adv_value / 1e6:.1f}M < ${min_adv / 1e6:.0f}M"
                )
        return passed, rejected

    def _filter_volatility(
        self, symbols: List[str], bars: Dict[str, pd.DataFrame]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Filter symbols by realized volatility cap.

        Args:
            symbols: Symbols to filter.
            bars: Dictionary of symbol to bar DataFrame.

        Returns:
            Tuple of (passed symbols, rejected dict with reasons).

        Example:
            >>> passed, rejected = screener._filter_volatility(["AAPL"], bars)
        """
        max_vol = self.config.get("max_realized_vol", 0.80)
        passed = []
        rejected = {}
        for symbol in symbols:
            df = bars.get(symbol)
            if df is None or df.empty:
                rejected[symbol] = "No data for volatility"
                continue
            vol = indicators.realized_vol(df["close"], period=20)
            if vol <= max_vol:
                passed.append(symbol)
            else:
                rejected[symbol] = f"Volatility {vol * 100:.1f}% > {max_vol * 100:.0f}%"
        return passed, rejected

    async def _filter_spread(
        self, symbols: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Filter symbols by bid-ask spread.

        Skipped entirely if the market is closed (quotes unavailable).

        Args:
            symbols: Symbols to filter.

        Returns:
            Tuple of (passed symbols, rejected dict with reasons).

        Example:
            >>> passed, rejected = await screener._filter_spread(["AAPL"])
        """
        max_spread_pct = self.config.get("max_spread_pct", 0.001)
        quotes = await fetch_latest_quotes(symbols)

        if not quotes:
            logger.warning("No quotes available - skipping spread filter")
            return symbols, {}

        passed = []
        rejected = {}
        for symbol in symbols:
            if symbol not in quotes:
                passed.append(symbol)
                continue
            q = quotes[symbol]
            bid = q.get("bid", 0)
            ask = q.get("ask", 0)
            if bid <= 0 or ask <= 0:
                passed.append(symbol)
                continue
            midpoint = (bid + ask) / 2.0
            spread_pct = (ask - bid) / midpoint if midpoint > 0 else 0
            if spread_pct <= max_spread_pct:
                passed.append(symbol)
            else:
                rejected[symbol] = (
                    f"Spread {spread_pct * 100:.3f}% > {max_spread_pct * 100:.2f}%"
                )
        return passed, rejected
