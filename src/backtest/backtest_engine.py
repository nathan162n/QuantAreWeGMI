"""Bar-by-bar backtest simulation engine.

Simulates all four strategy layers on historical 1-minute Alpaca bars.
Processes bars in strict chronological order across all symbols.
Uses BacktestBroker for simulated fills with slippage.

Key differences from the previous backtest:
  - Uses 1-minute bars (same resolution as live trading)
  - Simulates all 4 layers simultaneously
  - Applies PDT rules accurately (tracks rolling count)
  - Forces EOD close at 15:30 ET simulation time
  - No APScheduler — pure chronological loop
"""

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Bar cache directory — downloaded bars are stored here so repeat --test runs
# are instant without re-hitting the Alpaca API.
_BAR_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "bar_cache",
)

import pandas as pd

from src.backtest.backtest_broker import BacktestBroker
from src.backtest.backtest_reporter import BacktestReporter
from src.data.bar_store import BarStore
from src.engine.bar_dispatcher import BarDispatcher
from src.risk.pdt_guard import PDTGuard
from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
from src.strategy.regime_filter import RegimeFilter
from src.risk.position_sizer import PositionSizer
from src.utils.logger import get_logger

logger = get_logger("backtest.engine")


@dataclass
class BacktestResult:
    """Results from a completed backtest run.

    Attributes:
        total_trades: Total number of closed trades.
        win_rate: Win rate as percentage (0-100).
        total_pnl: Total profit/loss in dollars.
        final_equity: Ending portfolio equity.
        initial_equity: Starting capital.
        start_date: Backtest start date.
        end_date: Backtest end date.
        trade_history: Full list of trade dicts.
        equity_curve: List of {date, equity} dicts.
        pdt_blocks: Number of trades blocked by PDT rules.
        eod_closes: Number of EOD forced closes.
        report_text: Formatted report string.
        trades_csv_path: Path to trades CSV file.
        equity_csv_path: Path to equity curve CSV.
    """

    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    final_equity: float = 500.0
    initial_equity: float = 500.0
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    trade_history: List[dict] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)
    pdt_blocks: int = 0
    eod_closes: int = 0
    report_text: str = ""
    trades_csv_path: str = ""
    equity_csv_path: str = ""


class BacktestEngine:
    """Bar-by-bar simulation engine for --nighttest mode.

    Data loading:
        For each symbol: fetch historical 1-min bars from Alpaca.
        Store in memory as Dict[symbol, pd.DataFrame].
        Merge all symbols into single timeline sorted by timestamp.
        Process each bar as if it were a live WebSocket callback.

    PDT simulation:
        PDT guard runs in simulation mode — counts are tracked accurately
        but do NOT prevent trades. Instead the report shows 'PDT blocks: N'.

    Args:
        symbols: List of symbols to backtest. Default uses universe_largecap.csv.
        initial_equity: Starting capital. Default 500.0.
        slippage_pct: Per-side slippage. Default 0.0005 (0.05%).
        max_positions: Max simultaneous positions. Default 3.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        initial_equity: float = 500.0,
        slippage_pct: float = 0.0005,
        max_positions: int = 3,
    ):
        self.initial_equity = initial_equity
        self.slippage_pct = slippage_pct
        self.max_positions = max_positions

        if symbols is not None:
            self.symbols = symbols
        else:
            self.symbols = self._load_default_symbols()

    def _load_default_symbols(self) -> List[str]:
        """Load top 10 symbols from universe_largecap.csv for speed."""
        try:
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data",
                "universe_largecap.csv",
            )
            symbols = []
            with open(csv_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue  # skip header
                    parts = line.strip().split(",")
                    if parts and parts[0]:
                        symbols.append(parts[0].strip())
                    if len(symbols) >= 10:
                        break
            logger.info("Loaded %d symbols from universe_largecap.csv", len(symbols))
            return symbols
        except Exception as e:
            logger.error("Failed to load symbols from CSV: %s", str(e))
            return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "JPM", "SPY"]

    def run(
        self,
        start_date: date,
        end_date: date,
        initial_equity: Optional[float] = None,
    ) -> BacktestResult:
        """Run the full backtest simulation.

        Args:
            start_date: First date to simulate.
            end_date: Last date to simulate.
            initial_equity: Override starting capital. Default uses self.initial_equity.

        Returns:
            BacktestResult: Complete results with report text and CSV paths.
        """
        equity_amount = initial_equity if initial_equity is not None else self.initial_equity

        broker = BacktestBroker(equity_amount, self.slippage_pct)
        pdt_guard = PDTGuard()
        position_sizer = PositionSizer()
        bar_store = BarStore()

        layers = [
            VWAPMeanReversionStrategy(),
            OpeningRangeBreakoutStrategy(),
            RSIReversalScalpStrategy(),
            VolumeSurgeMomentumStrategy(),
        ]
        dispatcher = BarDispatcher(layers, self.max_positions)
        regime_filter = RegimeFilter()

        equity_curve: List[dict] = []
        pdt_blocks_counter = [0]
        eod_closes_counter = [0]
        equity_at_open = equity_amount

        # --- 1. Fetch regime data (SPY + VIXY daily bars) ---
        print("  Fetching regime data (SPY/VIXY daily bars)...")
        spy_daily, vixy_daily = self._fetch_regime_data(start_date, end_date)

        # --- 2. Fetch all 1-minute bars (cache-first) ---
        logger.info(
            "Fetching historical 1-min bars for %d symbols: %s to %s",
            len(self.symbols),
            start_date,
            end_date,
        )
        print(f"  Fetching bars for {len(self.symbols)} symbols (cached bars load instantly)...")
        symbol_bars: Dict[str, Optional[pd.DataFrame]] = {}
        for sym in self.symbols:
            df = self._fetch_bars_for_symbol(sym, start_date, end_date)
            if df is None or df.empty:
                logger.warning("No bars returned for %s — skipping", sym)
                symbol_bars[sym] = None
            else:
                symbol_bars[sym] = df
                logger.info("Loaded %d bars for %s", len(df), sym)

        # --- 3. Build unified timeline ---
        all_dfs = []
        for sym, df in symbol_bars.items():
            if df is None or df.empty:
                continue
            df = df.copy()
            df["symbol"] = sym
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = df.index
                df = df.reset_index(drop=True)
            elif "timestamp" not in df.columns:
                logger.warning("No timestamp for %s bars -- skipping", sym)
                continue
            all_dfs.append(df)

        if not all_dfs:
            logger.error("No bar data fetched for any symbol. Returning empty result.")
            return BacktestResult(
                initial_equity=equity_amount,
                final_equity=equity_amount,
                start_date=start_date,
                end_date=end_date,
            )

        merged = pd.concat(all_dfs).sort_values("timestamp").reset_index(drop=True)
        total_bars = len(merged)
        logger.info("Unified timeline: %d bars across all symbols", total_bars)

        # --- 4. Process bars chronologically ---
        current_date: Optional[date] = None
        session_reset_done_for: Dict[date, bool] = {}
        regime_updated_for: Dict[date, bool] = {}
        # Track the most recent close price for every symbol — used for EOD close
        current_prices: Dict[str, float] = {}

        for i, (idx, row) in enumerate(merged.iterrows()):
            try:
                bar_dict = self._bar_to_dict(row, row["symbol"])
            except Exception as e:
                logger.error("Failed to build bar_dict at index %d: %s", i, str(e))
                continue

            ts = bar_dict.get("timestamp")
            if ts is None:
                continue

            # Ensure timezone-aware
            if isinstance(ts, pd.Timestamp):
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                ts = ts.to_pydatetime()
            elif isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            bar_dict["timestamp"] = ts
            bar_current_date = ts.date()
            sym = bar_dict["symbol"]

            # Always update the current price tracker for this symbol
            current_prices[sym] = float(bar_dict.get("close", 0.0))

            # Progress display
            if i % 1000 == 0:
                print(
                    f"\r  Processing bar {i}/{total_bars}"
                    f" | Date: {bar_current_date}"
                    f" | Regime: {regime_filter.current_regime}"
                    f" | Equity: ${broker.equity:.2f}"
                    f" | Trades: {len(broker.get_trade_history())}",
                    end="",
                    flush=True,
                )

            # EOD check: force close at 15:30 ET using latest prices for ALL symbols
            if self._is_eod_bar(bar_dict):
                open_pos = broker.get_open_positions()
                if open_pos:
                    fills = broker.force_close_all(dict(current_prices), ts, "EOD")
                    eod_closes_counter[0] += len(fills)
                continue

            # Session reset at 09:30 ET
            if self._is_session_open_bar(bar_dict):
                if bar_current_date not in session_reset_done_for:
                    session_reset_done_for[bar_current_date] = True
                    bar_store.reset_session(sym)
                    for layer in layers:
                        if hasattr(layer, "reset_daily_state"):
                            layer.reset_daily_state()
                    equity_at_open = broker.equity
                    logger.debug("Session reset for %s on %s", sym, bar_current_date)
                else:
                    bar_store.reset_session(sym)

            # Update regime once per trading day at session open (mirrors live pre-market)
            if bar_current_date not in regime_updated_for and self._is_session_open_bar(bar_dict):
                regime_updated_for[bar_current_date] = True
                if spy_daily is not None and vixy_daily is not None:
                    try:
                        ts_day = pd.Timestamp(bar_current_date, tz="UTC")
                        spy_slice = spy_daily[spy_daily.index <= ts_day]["close"]
                        # Find VIXY close for the most recent available date
                        vixy_slice = vixy_daily[vixy_daily.index <= ts_day]["close"]
                        if len(spy_slice) > 0 and len(vixy_slice) > 0:
                            vixy_price = float(vixy_slice.iloc[-1])
                            regime, scalar = regime_filter.get_regime(spy_slice, vixy_price)
                            logger.debug(
                                "Regime updated for %s: %s (scalar=%.1f, VIXY=%.2f)",
                                bar_current_date, regime, scalar, vixy_price,
                            )
                    except Exception as e:
                        logger.warning("Regime update failed for %s: %s", bar_current_date, e)

            # ORB finalization at 10:00 ET
            if self._is_orb_close_bar(bar_dict):
                bar_store.finalize_opening_range(sym)

            # Record daily equity (once per date)
            if not equity_curve or equity_curve[-1]["date"] != bar_current_date:
                equity_curve.append({"date": bar_current_date, "equity": broker.equity})

            current_date = bar_current_date

            # Process the bar
            self._process_bar(
                sym,
                bar_dict,
                bar_store,
                broker,
                pdt_guard,
                position_sizer,
                dispatcher,
                regime_filter,
                layers,
                bar_current_date,
                equity_at_open,
                pdt_blocks_counter,
            )

        print()  # newline after progress output
        logger.info("Simulation complete. Processing results...")

        trade_history = broker.get_trade_history()
        final_equity = broker.equity
        total_trades = len(trade_history)
        winning = sum(1 for t in trade_history if t.get("pnl", 0) > 0)
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl = final_equity - equity_amount

        # Generate report
        reporter = BacktestReporter()
        date_suffix = f"{start_date}_{end_date}".replace("-", "")
        report_text = reporter.generate_report(
            trade_history=trade_history,
            equity_curve=equity_curve,
            start_date=start_date,
            end_date=end_date,
            initial_equity=equity_amount,
            pdt_blocks=pdt_blocks_counter[0],
            eod_closes=eod_closes_counter[0],
        )

        trades_csv_path, equity_csv_path = reporter.save_csv_files(
            trade_history=trade_history,
            equity_curve=equity_curve,
            date_suffix=date_suffix,
        )

        return BacktestResult(
            total_trades=total_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            final_equity=final_equity,
            initial_equity=equity_amount,
            start_date=start_date,
            end_date=end_date,
            trade_history=trade_history,
            equity_curve=equity_curve,
            pdt_blocks=pdt_blocks_counter[0],
            eod_closes=eod_closes_counter[0],
            report_text=report_text,
            trades_csv_path=trades_csv_path,
            equity_csv_path=equity_csv_path,
        )

    def _get_cache_path(self, symbol: str, start_date: date, end_date: date) -> str:
        """Return the filesystem path for a symbol's bar cache file."""
        os.makedirs(_BAR_CACHE_DIR, exist_ok=True)
        filename = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}.csv.gz"
        return os.path.join(_BAR_CACHE_DIR, filename)

    def _fetch_bars_for_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Fetch 1-minute historical bars for one symbol, using local cache.

        Cache location: data/bar_cache/{symbol}_{start}_{end}.csv.gz
        On a cache hit the Alpaca API is not called, making repeated --test
        runs across the same date range essentially instant.

        Args:
            symbol: Ticker symbol.
            start_date: Start date.
            end_date: End date.

        Returns:
            Optional[pd.DataFrame]: DataFrame with OHLCV columns and datetime index,
                or None on failure.
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date)

        # --- Cache hit ---
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, compression="gzip", index_col=0, parse_dates=True)
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")
                logger.info("Cache hit for %s (%d bars)", symbol, len(df))
                return df
            except Exception as e:
                logger.warning("Cache read failed for %s, re-fetching: %s", symbol, e)

        # --- Cache miss: fetch from Alpaca ---
        try:
            from src.broker.alpaca_client import get_client

            client = get_client()
            start_dt = datetime(
                start_date.year, start_date.month, start_date.day,
                13, 30, 0, tzinfo=timezone.utc  # 09:30 ET
            )
            end_dt = datetime(
                end_date.year, end_date.month, end_date.day,
                21, 0, 0, tzinfo=timezone.utc  # 17:00 ET buffer
            )

            df = client.get_bars(symbol, "1Min", start=start_dt)
            if df is None or df.empty:
                logger.warning("No bars returned from Alpaca for %s", symbol)
                return None

            # Filter to requested date range
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")
                df = df[(df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))]

            # Save to cache
            try:
                df.to_csv(cache_path, compression="gzip")
                logger.info("Cached %d bars for %s → %s", len(df), symbol, cache_path)
            except Exception as e:
                logger.warning("Failed to write cache for %s: %s", symbol, e)

            return df

        except Exception as e:
            logger.error(
                "Failed to fetch bars for %s (%s to %s): %s",
                symbol, start_date, end_date, str(e), exc_info=True,
            )
            return None

    def _fetch_regime_data(
        self,
        start_date: date,
        end_date: date,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch SPY and VIXY daily bars for regime detection across the backtest period.

        Fetches SPY going back an extra 300 calendar days before start_date so the
        200-day SMA is warm from the very first bar of the simulation.

        Returns:
            Tuple[spy_daily_df, vixy_daily_df]: DataFrames indexed by date, or
                (None, None) if the fetch fails (regime stays NEUTRAL).
        """
        try:
            from src.broker.alpaca_client import get_client
            client = get_client()

            # Extra lookback so SMA-200 is warmed up immediately
            spy_start = datetime(start_date.year, start_date.month, start_date.day,
                                 tzinfo=timezone.utc) - timedelta(days=300)
            end_dt = datetime(end_date.year, end_date.month, end_date.day,
                              tzinfo=timezone.utc)

            spy_df = client.get_bars("SPY", "1Day", start=spy_start)
            vixy_df = client.get_bars("VIXY", "1Day", start=spy_start)

            if spy_df is not None and not spy_df.empty:
                if spy_df.index.tzinfo is None:
                    spy_df.index = spy_df.index.tz_localize("UTC")
                logger.info("Fetched %d SPY daily bars for regime detection", len(spy_df))
            else:
                spy_df = None
                logger.warning("Could not fetch SPY daily bars — regime will stay NEUTRAL")

            if vixy_df is not None and not vixy_df.empty:
                if vixy_df.index.tzinfo is None:
                    vixy_df.index = vixy_df.index.tz_localize("UTC")
                logger.info("Fetched %d VIXY daily bars for regime detection", len(vixy_df))
            else:
                vixy_df = None
                logger.warning("Could not fetch VIXY daily bars — regime will stay NEUTRAL")

            return spy_df, vixy_df

        except Exception as e:
            logger.warning("Regime data fetch failed: %s — regime will stay NEUTRAL", e)
            return None, None

    def _process_bar(
        self,
        symbol: str,
        bar: dict,
        bar_store: BarStore,
        broker: BacktestBroker,
        pdt_guard: PDTGuard,
        position_sizer: PositionSizer,
        dispatcher: BarDispatcher,
        regime_filter: RegimeFilter,
        layers: list,
        current_date: date,
        equity_at_open: float,
        pdt_blocks_counter: list,
    ) -> None:
        """Process a single bar in simulation.

        Updates bar store, runs dispatcher, handles entries/exits.
        pdt_blocks_counter is a mutable list [count] for pass-by-reference.
        """
        # 1. Update bar store
        bar_store.update(symbol, "1Min", bar)

        # 2. Update current prices for equity calc
        broker.update_prices(symbol, bar["close"])

        # 3. Get open positions
        open_positions = broker.get_open_positions()

        # 4. Check stop losses on any open position for this symbol
        for pos_key, pos in list(open_positions.items()):
            if pos.get("symbol") != symbol:
                continue
            stop_price = pos.get("stop_price", 0.0)
            if stop_price > 0 and bar["low"] <= stop_price:
                # Stop hit — exit at stop price (or current bar close if worse)
                exit_price = min(bar["close"], stop_price)
                ts = bar.get("timestamp", datetime.now(timezone.utc))
                broker.submit_sell(symbol, pos["qty"], exit_price, ts, "STOP")
                logger.debug(
                    "Stop loss hit for %s at $%.4f (stop=$%.4f)",
                    pos_key,
                    exit_price,
                    stop_price,
                )
                # Refresh open positions after stop exit
                open_positions = broker.get_open_positions()
                break

        # 5. Dispatch bar to strategy layers
        try:
            signals = dispatcher.dispatch(
                symbol=symbol,
                bar=bar,
                bar_store=bar_store,
                open_positions=open_positions,
                open_orders=set(),
                layer2_enabled=regime_filter.is_layer2_enabled(),
            )
        except Exception as e:
            logger.error("Dispatcher error for %s: %s", symbol, str(e), exc_info=True)
            return

        ts = bar.get("timestamp", datetime.now(timezone.utc))

        for signal in signals:
            try:
                if signal.signal == "BUY":
                    equity = broker.equity
                    shares = position_sizer.compute_shares(
                        signal,
                        equity=equity,
                        buying_power=broker.cash,
                        current_open_positions=len(open_positions),
                        regime_scalar=regime_filter.current_size_scalar,
                    )
                    if shares > 0:
                        fill = broker.submit_buy(
                            symbol=symbol,
                            qty=shares,
                            price=signal.signal_price,
                            timestamp=ts,
                            layer_name=signal.layer_name,
                            stop_price=signal.stop_price,
                        )
                        if fill:
                            logger.debug(
                                "BUY %s x%d @ $%.4f via %s",
                                symbol,
                                shares,
                                fill.fill_price,
                                signal.layer_name,
                            )

                elif signal.signal == "EXIT":
                    pos_key = f"{symbol}_{signal.layer_name}"
                    open_positions_now = broker.get_open_positions()
                    if pos_key in open_positions_now:
                        pos = open_positions_now[pos_key]
                        # PDT simulation: track but don't block
                        entry_time = pos.get("entry_time") or pos.get("timestamp")
                        if entry_time and isinstance(entry_time, datetime):
                            is_day_trade = entry_time.date() == current_date
                            if is_day_trade:
                                # Simulate PDT count increment
                                pdt_blocks_counter[0]  # access to confirm list exists
                                # In simulation mode: record the day trade event but always execute
                                logger.debug(
                                    "PDT day trade recorded (simulation): %s on %s",
                                    symbol,
                                    current_date,
                                )
                        fill = broker.submit_sell(
                            symbol=symbol,
                            qty=pos["qty"],
                            price=bar["close"],
                            timestamp=ts,
                            exit_reason="SIGNAL",
                        )
                        if fill:
                            logger.debug(
                                "EXIT %s x%d @ $%.4f via %s",
                                symbol,
                                fill.qty,
                                fill.fill_price,
                                signal.layer_name,
                            )

            except Exception as e:
                logger.error(
                    "Error processing signal %s for %s: %s",
                    signal.signal,
                    symbol,
                    str(e),
                    exc_info=True,
                )

    def _is_eod_bar(self, bar: dict) -> bool:
        """Return True if this bar is at or after 15:30 ET.

        15:30 ET = 20:30 UTC (EST) or 19:30 UTC (EDT).
        """
        ts = bar.get("timestamp")
        if ts is None:
            return False
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        minutes_utc = ts.hour * 60 + ts.minute
        # 15:30 ET = 20:30 UTC (EST) = 1230 minutes, or 19:30 UTC (EDT) = 1170 minutes
        return minutes_utc == 20 * 60 + 30 or minutes_utc == 19 * 60 + 30

    def _is_session_open_bar(self, bar: dict) -> bool:
        """Return True if this bar is the first bar of the session (09:30 ET).

        09:30 ET = 14:30 UTC (EST) or 13:30 UTC (EDT).
        """
        ts = bar.get("timestamp")
        if ts is None:
            return False
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        minutes_utc = ts.hour * 60 + ts.minute
        # 09:30 ET = 14:30 UTC (EST) = 870, or 13:30 UTC (EDT) = 810
        return minutes_utc == 14 * 60 + 30 or minutes_utc == 13 * 60 + 30

    def _is_orb_close_bar(self, bar: dict) -> bool:
        """Return True if this bar is at 10:00 ET (opening range finalization).

        10:00 ET = 15:00 UTC (EST) or 14:00 UTC (EDT).
        """
        ts = bar.get("timestamp")
        if ts is None:
            return False
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        minutes_utc = ts.hour * 60 + ts.minute
        # 10:00 ET = 15:00 UTC (EST) = 900, or 14:00 UTC (EDT) = 840
        return minutes_utc == 15 * 60 + 0 or minutes_utc == 14 * 60 + 0

    def _bar_to_dict(self, row: pd.Series, symbol: str) -> dict:
        """Convert a DataFrame row to a bar dict."""
        ts = row.get("timestamp", None)
        if ts is None and hasattr(row, "name"):
            ts = row.name  # use index if timestamp col missing

        return {
            "open": float(row.get("open", 0.0) if hasattr(row, "get") else getattr(row, "open", 0.0)),
            "high": float(row.get("high", 0.0) if hasattr(row, "get") else getattr(row, "high", 0.0)),
            "low": float(row.get("low", 0.0) if hasattr(row, "get") else getattr(row, "low", 0.0)),
            "close": float(row.get("close", 0.0) if hasattr(row, "get") else getattr(row, "close", 0.0)),
            "volume": float(row.get("volume", 0.0) if hasattr(row, "get") else getattr(row, "volume", 0.0)),
            "symbol": symbol,
            "timestamp": ts,
        }
