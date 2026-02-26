"""AlgoTrader — High-Frequency Multi-Strategy Algorithmic Trading System.

Entry point and orchestrator for all run modes. Implements a fully
event-driven WebSocket architecture for live trading and a bar-by-bar
simulation engine for backtesting.

CLI flags:
    --paper            Paper trading, WebSocket-driven live loop (default)
    --live             Live trading (requires TRADING_MODE=live + confirmation)
    --test             Run bar-by-bar replay on historical 1-min Alpaca data
    --days N           Number of trading days for --test (default 365)
    --start YYYY-MM-DD Start date for --test (overrides --days)
    --reset-circuit-breaker  Reset circuit breaker flag and exit
    --report-only      Dashboard only, no trading
    --dry-run          Full signal generation, no real orders
    --help             Show all flags

Startup flow (live mode):
    1. Parse CLI
    2. Init logger
    3. Init DB (Supabase → SQLite fallback)
    4. Start NotificationQueue worker
    5. Check circuit breaker → halt if active
    6. Init AlpacaClient
    7. Assert is_paper_trading() unless --live
    8. Send SYSTEM_RESTART Discord alert
    9. Init BarStore (empty)
    10. Init all four strategy layer instances
    11. Init BarDispatcher with all four layers
    12. Init StreamEngine with BarDispatcher
    13. Pre-market setup (APScheduler, 08:00 ET): screener, bar warm-up, regime
    14. Market open (APScheduler, 09:30 ET): session reset, start StreamEngine
    15. ORB close (APScheduler, 10:00 ET): finalize opening ranges
    16. EOD close (APScheduler, 15:30 ET): EODManager
    17. Post-market (APScheduler, 16:30 ET): daily summary, HTML report
"""

import argparse
import asyncio
import os
import signal
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
)

from src.utils.logger import setup_logger, get_logger

logger = get_logger("main")

_NOTIFICATION_QUEUE = None
_STREAM_ENGINE = None
_SHUTDOWN_EVENT: Optional[asyncio.Event] = None


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load complete configuration from config.yml.

    Returns:
        dict: Full configuration dictionary.

    Example:
        >>> config = load_config()
        >>> config["trading"]["max_positions"]
        3
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_args()
        >>> args.paper
        True
    """
    parser = argparse.ArgumentParser(
        description="AlgoTrader — High-Frequency Multi-Strategy Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --paper                     Start paper trading (default)
  python main.py --dry-run                   Full logic, no real orders
  python main.py --report-only               Generate dashboard and exit
  python main.py --test                      Run 365-day historical bar replay
  python main.py --test --days 30            30-day replay
  python main.py --test --start 2024-01-01   Replay from specific date
  python main.py --reset-circuit-breaker     Reset circuit breaker and exit

Bar cache: downloaded bars are cached in data/bar_cache/ so repeated --test
runs are instant. Delete that folder to force a fresh download.
        """,
    )
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true",
                        help="Live trading (requires TRADING_MODE=live)")
    parser.add_argument("--test", action="store_true",
                        help="Run bar-by-bar historical replay using real Alpaca 1-min bars")
    parser.add_argument("--nighttest", action="store_true",
                        help=argparse.SUPPRESS)  # deprecated alias for --test
    parser.add_argument("--days", type=int, default=365,
                        help="Number of trading days for --test (default 365)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD for --test")
    parser.add_argument("--reset-circuit-breaker", action="store_true",
                        help="Reset the circuit breaker flag and exit")
    parser.add_argument("--report-only", action="store_true",
                        help="Generate dashboard without starting trading loop")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals but don't submit real orders")
    return parser.parse_args()


def load_universe(config: dict) -> List[str]:
    """Load the list of symbols from the universe CSV file.

    Args:
        config: Configuration dictionary.

    Returns:
        List[str]: List of ticker symbols.

    Example:
        >>> symbols = load_universe(config)
        >>> "AAPL" in symbols
        True
    """
    import csv
    base_path = config.get("universe", {}).get("base_list_path", "data/universe_largecap.csv")
    csv_path = os.path.join(os.path.dirname(__file__), base_path)
    symbols = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get("symbol", "").strip()
                if sym:
                    symbols.append(sym)
        logger.info("Loaded universe: %d symbols", len(symbols))
    except Exception as e:
        logger.error("Failed to load universe CSV: %s", str(e))
        symbols = [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
            "JPM", "SPY", "QQQ", "GLD",
        ]
        logger.warning("Using fallback universe of %d symbols", len(symbols))
    return symbols


# ---------------------------------------------------------------------------
# Special run modes
# ---------------------------------------------------------------------------

def run_reset_circuit_breaker() -> None:
    """Reset the circuit breaker flag, log, and send Discord alert.

    Example:
        >>> run_reset_circuit_breaker()
    """
    try:
        from src.risk.circuit_breaker import CircuitBreaker
        CircuitBreaker.reset("Manual reset via --reset-circuit-breaker")
        logger.info("Circuit breaker reset successfully.")
        print("Circuit breaker reset. Trading is now enabled.")
    except Exception as e:
        logger.error("Failed to reset circuit breaker: %s", str(e))
        print(f"ERROR: Failed to reset circuit breaker: {e}")


def run_report_only(config: dict) -> None:
    """Generate dashboard and exit without starting the trading loop.

    Args:
        config: Configuration dictionary.

    Example:
        >>> run_report_only(config)
    """
    try:
        from reports.dashboard import generate_terminal_report
        report_data = generate_terminal_report()
        print(report_data.get("summary", "Report generated successfully."))
    except Exception as e:
        logger.error("Dashboard generation failed: %s", str(e))
        print(f"Dashboard error: {e}")
        traceback.print_exc()


def run_nighttest(args: argparse.Namespace, config: dict) -> None:
    """Run the historical backtest simulation.

    Args:
        args: Parsed CLI arguments.
        config: Configuration dictionary.

    Example:
        >>> run_nighttest(args, config)
    """
    from src.backtest.backtest_engine import BacktestEngine

    # Compute date range
    if args.start:
        try:
            start_date = date.fromisoformat(args.start)
        except ValueError:
            print(f"ERROR: Invalid start date format: {args.start}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        # Go back args.days trading days from today
        # Rough approximation: N days * (7/5) calendar days to account for weekends
        calendar_days = int(args.days * 1.4) + 10
        start_date = date.today() - timedelta(days=calendar_days)

    end_date = date.today()

    print(f"\nAlgoTrader -- Historical Bar Replay (--test)")
    print(f"Period: {start_date} to {end_date} ({args.days} requested trading days)")
    print(f"Capital: ${config['reporting']['starting_capital']:.2f}")
    print(f"Symbols: loading from universe CSV...")
    print()

    symbols = load_universe(config)
    initial_equity = float(config["reporting"]["starting_capital"])
    slippage_pct = float(config.get("backtest", {}).get("default_slippage_pct", 0.0005))
    max_positions = int(config["risk"]["max_positions"])

    engine = BacktestEngine(
        symbols=symbols,
        initial_equity=initial_equity,
        slippage_pct=slippage_pct,
        max_positions=max_positions,
    )

    try:
        result = engine.run(
            start_date=start_date,
            end_date=end_date,
            initial_equity=initial_equity,
        )
    except Exception as e:
        logger.error("Backtest failed: %s\n%s", str(e), traceback.format_exc())
        print(f"\nERROR: Backtest failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Print the formatted report
    print(result.report_text)

    # Check against minimum trade count target
    target = int(config.get("backtest", {}).get("target_trades_per_year", 300))
    actual_days = (end_date - start_date).days
    annualized_pace = int(result.total_trades * 365 / max(actual_days, 1))

    if annualized_pace < target:
        logger.warning(
            "Trade count below target: %d trades (%d annualized pace) vs target %d/year",
            result.total_trades,
            annualized_pace,
            target,
        )
        print(
            f"\n⚠  WARNING: Trade count {result.total_trades} ({annualized_pace}/yr pace) "
            f"is below the target of {target}/year. "
            f"Signal thresholds may still be too restrictive."
        )
    else:
        print(f"\n✓  Trade count OK: {result.total_trades} trades ({annualized_pace}/yr pace)")

    print(f"\nEquity curve: {result.equity_csv_path}")
    print(f"Trade log:    {result.trades_csv_path}")


# ---------------------------------------------------------------------------
# Live trading orchestration
# ---------------------------------------------------------------------------

async def run_live_trading(args: argparse.Namespace, config: dict) -> None:
    """Run the full live trading system — WebSocket-driven intraday loop.

    Sets up APScheduler for pre-market, ORB, EOD, and post-market tasks.
    The intraday signal generation is driven entirely by WebSocket bar events.

    Args:
        args: Parsed CLI arguments.
        config: Configuration dictionary.

    Example:
        >>> asyncio.run(run_live_trading(args, config))
    """
    global _NOTIFICATION_QUEUE, _STREAM_ENGINE, _SHUTDOWN_EVENT

    _SHUTDOWN_EVENT = asyncio.Event()

    # --- Step 3: Init DB ---
    try:
        from src.database.engine import init_db
        init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.critical("Database init failed: %s", str(e))
        # Non-fatal — SQLite fallback should handle it

    # --- Step 4: Start NotificationQueue ---
    # The background worker thread starts automatically inside NotificationQueue.__init__().
    # Do NOT call start_worker() — no such method exists.
    try:
        from src.notifications.notification_queue import NotificationQueue
        queue_size = int(config.get("notifications", {}).get("queue_max_size", 1000))
        _NOTIFICATION_QUEUE = NotificationQueue(max_size=queue_size)
        logger.info("NotificationQueue started.")
    except Exception as e:
        logger.error("NotificationQueue failed to start: %s", str(e))

    # --- Step 5: Check circuit breaker ---
    from src.risk.circuit_breaker import CircuitBreaker, TRADING_HALTED
    cb = CircuitBreaker()
    if cb.is_active():
        logger.critical(
            "Circuit breaker is ACTIVE. Trading halted. "
            "Run: python main.py --reset-circuit-breaker"
        )
        if _NOTIFICATION_QUEUE:
            _NOTIFICATION_QUEUE.enqueue_alert(
                "ERROR",
                "Circuit breaker is active. Trading halted.",
            )
        return

    # --- Step 6: Init AlpacaClient ---
    from src.broker.alpaca_client import get_client, reset_client
    reset_client()
    client = get_client()

    # --- Step 7: Assert paper trading unless --live ---
    if args.live:
        if os.environ.get("TRADING_MODE", "paper").lower() != "live":
            print("ERROR: --live requires TRADING_MODE=live in .env")
            sys.exit(1)
        confirm = input("WARNING: LIVE TRADING MODE. Real money at risk. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(0)
        logger.warning("LIVE TRADING MODE ACTIVE")
    else:
        if not client.is_paper_trading():
            logger.critical("Non-paper URL detected without --live flag. Aborting.")
            sys.exit(1)
        logger.info("Paper trading mode confirmed.")

    # --- AccountMonitor: real-time account state polling ---
    from src.broker.account_monitor import AccountMonitor
    account_monitor = AccountMonitor(client, poll_interval_seconds=10)
    await account_monitor.start()
    logger.info(
        f"Paper account live — equity=${account_monitor.equity:,.2f}, "
        f"buying_power=${account_monitor.buying_power:,.2f}"
    )

    # Close any inherited short positions
    short_symbols = account_monitor.get_short_symbols()
    if short_symbols:
        logger.critical(f"Short positions found — closing immediately: {short_symbols}")
        for sym in short_symbols:
            pos = account_monitor.get_position(sym)
            qty = abs(pos["qty"])
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, client.submit_market_order, sym, qty, "buy"
                )
                logger.critical(f"Closed short {sym}: bought {qty} shares")
            except Exception as e:
                logger.error(f"Failed to close short {sym}: {e}")
        await asyncio.sleep(12)
        await account_monitor._poll()
    else:
        logger.info("Startup check passed — no short positions")

    # --- Step 8: System restart Discord alert ---
    mode_str = "LIVE" if args.live else ("DRY RUN" if args.dry_run else "PAPER")
    if _NOTIFICATION_QUEUE:
        try:
            _NOTIFICATION_QUEUE.enqueue_alert(
                "SYSTEM_RESTART",
                f"AlgoTrader restarted in {mode_str} mode. "
                f"Universe: {len(load_universe(config))} symbols. "
                f"4 strategy layers active.",
            )
        except Exception as e:
            logger.error("Failed to send startup Discord alert: %s", str(e))

    # --- Step 9: Init BarStore ---
    from src.data.bar_store import get_bar_store, reset_bar_store
    reset_bar_store()
    bar_store = get_bar_store()
    logger.info("BarStore initialized.")

    # --- Step 10: Init strategy layers ---
    from src.strategy.vwap_mean_reversion import VWAPMeanReversionStrategy
    from src.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy
    from src.strategy.rsi_reversal_scalp import RSIReversalScalpStrategy
    from src.strategy.volume_surge_momentum import VolumeSurgeMomentumStrategy
    from src.strategy.regime_filter import get_regime_filter

    layers = [
        VWAPMeanReversionStrategy(),
        OpeningRangeBreakoutStrategy(),
        RSIReversalScalpStrategy(),
        VolumeSurgeMomentumStrategy(),
    ]
    regime_filter = get_regime_filter()
    logger.info("Strategy layers initialized: %s", [l.layer_name for l in layers])

    # --- Step 11: Init BarDispatcher ---
    from src.engine.bar_dispatcher import BarDispatcher
    max_positions = int(config["risk"]["max_positions"])
    l3_cadence = int(config.get("strategy", {}).get("l3_bar_cadence", 1))
    dispatcher = BarDispatcher(layers, max_positions=max_positions, l3_cadence=l3_cadence)

    # --- Step 12: Init StreamEngine ---
    from src.risk.pdt_guard import PDTGuard
    from src.risk.position_sizer import PositionSizer
    from src.engine.stream_engine import StreamEngine
    from src.broker.order_manager import OrderManager

    pdt_guard = PDTGuard(max_day_trades=3)
    position_sizer = PositionSizer(
        dollar_risk_per_trade=float(config["risk"]["dollar_risk_per_trade"]),
        max_position_pct=float(config["risk"]["max_position_pct"]),
        max_positions=max_positions,
        max_position_value=float(config["risk"].get("max_position_value", 10000.0)),
    )
    order_manager = OrderManager(dry_run=args.dry_run)

    # Attach account_monitor to circuit breaker for live equity data
    cb.account_monitor = account_monitor
    symbols = load_universe(config)

    _STREAM_ENGINE = StreamEngine(
        alpaca_client=client,
        bar_store=bar_store,
        bar_dispatcher=dispatcher,
        order_manager=order_manager,
        pdt_guard=pdt_guard,
        circuit_breaker=cb,
        notification_queue=_NOTIFICATION_QUEUE,
        regime_filter=regime_filter,
        symbols=symbols,
        max_positions=max_positions,
    )

    # --- APScheduler setup ---
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        import pytz

        tz_name = config.get("scheduler", {}).get("timezone", "America/New_York")
        tz = pytz.timezone(tz_name)
        scheduler = AsyncIOScheduler(timezone=tz)

        sched_cfg = config.get("scheduler", {})

        # Pre-market: 08:00 ET
        scheduler.add_job(
            _premarket_task,
            "cron",
            hour=sched_cfg.get("premarket_run_hour", 8),
            minute=sched_cfg.get("premarket_run_minute", 0),
            args=[client, bar_store, regime_filter, symbols, config],
            id="premarket",
            replace_existing=True,
        )

        # Market open: 09:30 ET — reset sessions and start stream
        scheduler.add_job(
            _market_open_task,
            "cron",
            hour=9,
            minute=30,
            args=[bar_store, layers, symbols],
            id="market_open",
            replace_existing=True,
        )

        # ORB finalization: 10:00 ET
        scheduler.add_job(
            _orb_finalize_task,
            "cron",
            hour=sched_cfg.get("opening_range_close_hour", 10),
            minute=sched_cfg.get("opening_range_close_minute", 0),
            args=[bar_store, symbols],
            id="orb_close",
            replace_existing=True,
        )

        # EOD close: 15:30 ET
        scheduler.add_job(
            _eod_task,
            "cron",
            hour=sched_cfg.get("eod_close_hour", 15),
            minute=sched_cfg.get("eod_close_minute", 30),
            args=[_STREAM_ENGINE, client, pdt_guard, order_manager, _NOTIFICATION_QUEUE],
            id="eod_close",
            replace_existing=True,
        )

        # Post-market summary: 16:30 ET
        scheduler.add_job(
            _post_market_task,
            "cron",
            hour=sched_cfg.get("post_market_summary_hour", 16),
            minute=sched_cfg.get("post_market_summary_minute", 30),
            args=[client, config, _NOTIFICATION_QUEUE],
            id="post_market",
            replace_existing=True,
        )

        scheduler.start()
        logger.info("APScheduler started with 5 scheduled tasks.")
    except ImportError:
        logger.error(
            "APScheduler not installed. Scheduled tasks will not run. "
            "Install with: pip install apscheduler pytz"
        )

    # --- Start WebSocket stream ---
    logger.info("Starting WebSocket stream for %d symbols...", len(symbols))
    try:
        await _STREAM_ENGINE.start()
    except asyncio.CancelledError:
        logger.info("Stream engine cancelled — shutting down.")
    except Exception as e:
        logger.critical("Stream engine crashed: %s\n%s", str(e), traceback.format_exc())
        if _NOTIFICATION_QUEUE:
            try:
                _NOTIFICATION_QUEUE.enqueue_alert(
                    "ERROR",
                    f"Stream engine crashed: {e}",
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Scheduled task callbacks
# ---------------------------------------------------------------------------

async def _premarket_task(
    client, bar_store, regime_filter, symbols: List[str], config: dict
) -> None:
    """Pre-market tasks at 08:00 ET.

    1. Run universe screener
    2. Warm up BarStore with 200 1-min bars per symbol
    3. Detect market regime (SPY + VIXY)

    Args:
        client: AlpacaClient instance.
        bar_store: BarStore instance.
        regime_filter: RegimeFilter instance.
        symbols: Universe symbol list.
        config: Configuration dict.

    Example:
        >>> await _premarket_task(client, store, rf, symbols, config)
    """
    logger.info("Pre-market task starting...")
    try:
        # Universe screener
        from src.universe.screener import run_screener
        passed = await asyncio.get_event_loop().run_in_executor(
            None, run_screener, symbols, config
        )
        logger.info("Screener passed: %d/%d symbols", len(passed), len(symbols))
    except Exception as e:
        logger.error("Screener failed: %s", str(e))

    # Warm up BarStore — fetch last 200 1-min bars for each symbol
    warm_start = datetime.now(timezone.utc) - timedelta(hours=8)
    for symbol in symbols:
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None, client.get_bars, symbol, "1Min", 200, warm_start
            )
            if df is not None and not df.empty:
                bars = []
                for idx, row in df.iterrows():
                    bars.append({
                        "open": float(row.get("open", 0)),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                        "close": float(row.get("close", 0)),
                        "volume": float(row.get("volume", 0)),
                        "timestamp": idx if hasattr(idx, "tzinfo") else datetime.now(timezone.utc),
                        "symbol": symbol,
                    })
                bar_store.preload_bars(symbol, "1Min", bars)
        except Exception as e:
            logger.warning("Failed to warm up %s: %s", symbol, str(e))

    # Detect regime
    try:
        spy_df = await asyncio.get_event_loop().run_in_executor(
            None, client.get_bars, "SPY", "1Day", 210, None
        )
        vixy_df = await asyncio.get_event_loop().run_in_executor(
            None, client.get_bars, "VIXY", "1Day", 2, None
        )
        if spy_df is not None and vixy_df is not None:
            vixy_price = float(vixy_df["close"].iloc[-1])
            regime, scalar = regime_filter.get_regime(spy_df["close"], vixy_price)
            logger.info(
                "Market regime: %s | VIXY=%.2f | Size scalar=%.1fx",
                regime, vixy_price, scalar
            )
    except Exception as e:
        logger.warning("Regime detection failed: %s", str(e))

    logger.info("Pre-market task complete.")


async def _market_open_task(bar_store, layers: list, symbols: List[str]) -> None:
    """Market open tasks at 09:30 ET.

    Resets session VWAP and opening range for all symbols.
    Resets Layer 2 daily state (opening range not yet triggered).

    Args:
        bar_store: BarStore instance.
        layers: List of strategy layer instances.
        symbols: Universe symbol list.

    Example:
        >>> await _market_open_task(store, layers, symbols)
    """
    logger.info("Market open: resetting session data for %d symbols", len(symbols))
    bar_store.reset_all_sessions()
    for layer in layers:
        if hasattr(layer, "reset_daily_state"):
            layer.reset_daily_state()
    logger.info("Session VWAP and opening range data reset.")


async def _orb_finalize_task(bar_store, symbols: List[str]) -> None:
    """Opening range finalization at 10:00 ET.

    Computes and stores the opening range (09:30–10:00 bars) for each symbol.
    After this runs, Layer 2 (ORB) becomes active.

    Args:
        bar_store: BarStore instance.
        symbols: Universe symbol list.

    Example:
        >>> await _orb_finalize_task(store, symbols)
    """
    logger.info("Finalizing opening ranges for %d symbols", len(symbols))
    computed = 0
    for symbol in symbols:
        try:
            result = bar_store.finalize_opening_range(symbol)
            if result:
                computed += 1
        except Exception as e:
            logger.warning("Failed to finalize ORB for %s: %s", symbol, str(e))
    logger.info("Opening ranges computed: %d/%d symbols", computed, len(symbols))


async def _eod_task(
    stream_engine, client, pdt_guard, order_manager, notification_queue
) -> None:
    """EOD forced close at 15:30 ET.

    Closes all positions that are not PDT-blocked.
    Cancels all open orders.

    Args:
        stream_engine: StreamEngine instance (to get open positions).
        client: AlpacaClient instance.
        pdt_guard: PDTGuard instance.
        order_manager: OrderManager instance.
        notification_queue: NotificationQueue instance.

    Example:
        >>> await _eod_task(engine, client, guard, om, queue)
    """
    logger.info("EOD task starting at 15:30 ET")
    try:
        from src.engine.eod_manager import EODManager
        eod = EODManager(
            alpaca_client=client,
            pdt_guard=pdt_guard,
            order_manager=order_manager,
            notification_queue=notification_queue,
        )
        open_positions = {}
        if stream_engine is not None and hasattr(stream_engine, "_open_positions"):
            open_positions = stream_engine._open_positions

        summary = await eod.run_eod_close(open_positions)
        logger.info("EOD close complete: %s", summary)
    except Exception as e:
        logger.error("EOD task failed: %s\n%s", str(e), traceback.format_exc())


async def _post_market_task(client, config: dict, notification_queue) -> None:
    """Post-market tasks at 16:30 ET.

    Saves DailyPortfolioState, generates daily summary, and sends to Discord.

    Args:
        client: AlpacaClient instance.
        config: Configuration dict.
        notification_queue: NotificationQueue instance.

    Example:
        >>> await _post_market_task(client, config, queue)
    """
    logger.info("Post-market task starting")
    try:
        account = await asyncio.get_event_loop().run_in_executor(
            None, client.get_account
        )
        equity = account["equity"]

        from src.database import repository
        from src.database.models import DailyPortfolioState

        # Get peak for drawdown calculation
        peak = repository.get_peak_portfolio_value()
        new_peak = max(peak, equity)
        drawdown_pct = (equity - new_peak) / new_peak if new_peak > 0 else 0.0

        repository.save_daily_state({
            "date": datetime.now(timezone.utc),
            "portfolio_value": equity,
            "cash": account["cash"],
            "equity": equity,
            "num_open_positions": len(client.get_positions()),
            "daily_pnl": 0.0,
            "daily_return_pct": 0.0,
            "peak_value": new_peak,
            "drawdown_pct": drawdown_pct,
            "regime": "UNKNOWN",
            "circuit_breaker_active": False,
            "day_trade_count": account["daytrade_count"],
        })

        if notification_queue:
            notification_queue.enqueue_daily_summary(
                portfolio_state={
                    "portfolio_value": equity,
                    "cash": account["cash"],
                    "drawdown_pct": drawdown_pct * 100,
                    "day_trade_count": account["daytrade_count"],
                },
                trades_today=[],
                metrics={},
            )
        logger.info("Post-market summary saved. Equity: $%.2f", equity)
    except Exception as e:
        logger.error("Post-market task failed: %s", str(e))

    # Generate HTML report
    try:
        from reports.dashboard import generate_dashboard
        generate_dashboard(config)
        logger.info("HTML dashboard generated.")
    except Exception as e:
        logger.warning("Dashboard generation failed: %s", str(e))


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGTERM/SIGINT handlers for graceful shutdown.

    Args:
        loop: The asyncio event loop.

    Example:
        >>> _setup_signal_handlers(asyncio.get_event_loop())
    """
    def _handle_shutdown(signum, frame):
        logger.info("Received signal %s — initiating graceful shutdown", signum)
        if _SHUTDOWN_EVENT:
            loop.call_soon_threadsafe(_SHUTDOWN_EVENT.set)
        if _STREAM_ENGINE:
            loop.call_soon_threadsafe(_STREAM_ENGINE.stop)
        if _NOTIFICATION_QUEUE:
            # shutdown() is a synchronous thread-safe call — drains and stops the worker.
            _NOTIFICATION_QUEUE.shutdown(timeout=5.0)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point.

    Parses CLI arguments and dispatches to the appropriate run mode.

    Example:
        >>> main()
    """
    # Setup logging first
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_dir = os.environ.get("LOG_DIR", "logs/")
    setup_logger(log_level=log_level, log_dir=log_dir)

    args = parse_args()
    config = load_config()

    # --reset-circuit-breaker: reset and exit immediately
    if args.reset_circuit_breaker:
        try:
            from src.database.engine import init_db
            init_db()
        except Exception:
            pass
        run_reset_circuit_breaker()
        sys.exit(0)

    # --report-only: generate dashboard and exit
    if args.report_only:
        try:
            from src.database.engine import init_db
            init_db()
        except Exception:
            pass
        run_report_only(config)
        sys.exit(0)

    # --test (or legacy --nighttest): run bar-by-bar historical replay and exit
    if args.test or args.nighttest:
        try:
            from src.database.engine import init_db
            init_db()
        except Exception:
            pass
        run_nighttest(args, config)
        sys.exit(0)

    # Default: live/paper trading
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _setup_signal_handlers(loop)
        loop.run_until_complete(run_live_trading(args, config))
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    except Exception as e:
        logger.critical("Fatal error in main: %s\n%s", str(e), traceback.format_exc())
        sys.exit(1)
    finally:
        try:
            loop.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
