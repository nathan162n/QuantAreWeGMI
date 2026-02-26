# AlgoTrader — Volatility-Adjusted Mean Reversion Trading System

## System Overview

AlgoTrader is a fully automated algorithmic trading system designed to run 24/7 for a minimum of one full month on Alpaca's paper trading platform, starting with a $500 simulated account. The system trades liquid S&P 500 equities using a volatility-adjusted mean reversion strategy with a VIX/SPY regime overlay. It is built for reliability above all else: automatic database failover from Supabase PostgreSQL to local SQLite, four independent circuit breaker triggers, strict Pattern Day Trader rule enforcement, and comprehensive logging ensure the system can operate unattended for extended periods.

Mean reversion was chosen as the primary strategy because it is one of the most empirically robust approaches for liquid, large-cap equities. The core premise is simple: stocks that deviate significantly from their recent average tend to revert toward that average. By combining z-score analysis (measuring how far a price has moved from its mean in standard deviations) with RSI confirmation (measuring momentum exhaustion), the system identifies statistically oversold conditions in high-quality, liquid stocks. The regime filter prevents the strategy from entering trades during trending or crashing markets — conditions where mean reversion structurally fails.

After one month of paper trading with a $500 account, realistic expectations include 5-15 completed trades, a small positive or negative P&L (the account size severely limits position sizing), and comprehensive data on strategy behavior across different market regimes. The primary value of the paper trading period is proving system reliability and gathering performance data, not generating significant returns.

## Strategy Explanation

The strategy identifies stocks that have become statistically oversold relative to their recent trading range. It uses a 20-day rolling z-score to measure how many standard deviations the current price sits below its 20-day moving average. When a stock's z-score drops below -2.0 (meaning it is more than two standard deviations below its recent average), and its 14-day RSI confirms oversold conditions by reading below 35, the system generates a BUY signal.

The z-score entry has a freefall protection threshold at -4.0. If a stock's z-score drops below -4.0, the system will not enter — the stock is likely in a fundamental breakdown rather than a mean-reverting dip. This prevents catching falling knives.

Exits are triggered when the z-score recovers above -0.5 (the stock has nearly returned to its mean), when RSI climbs above 65 (momentum has reversed to overbought), or after 15 trading days regardless of price action (time-based exit to prevent capital from being tied up indefinitely). All exits respect a 3-trading-day minimum holding period to protect against Pattern Day Trader violations.

The regime filter acts as a master switch on all new entries. It classifies the market into three regimes using SPY's position relative to its 200-day simple moving average and VIXY (a VIX proxy ETF) price levels. In a BULL regime (SPY above 200-day SMA, VIXY below 25), signals pass through at full position size. In a NEUTRAL regime (SPY above SMA but VIXY between 25 and 35), signals pass at 50% position size. In a BEAR regime (SPY below SMA or VIXY above 35), all new entries are blocked entirely. This prevents the strategy from taking mean reversion trades during sustained downtrends, where oversold stocks tend to become more oversold rather than reverting.

## Risk Model

Position sizing uses ATR (Average True Range) volatility targeting, the institutional standard for normalizing position sizes across stocks with different volatility profiles. The formula ensures that a volatile stock receives fewer shares than a calm stock, keeping the dollar risk per trade roughly constant.

Worked example at $500 equity:

The target portfolio volatility is 20% annualized, divided across 5 target positions. The daily volatility budget per position is calculated as: Dollar Risk Per Trade = $500 × (0.20 / sqrt(252)) × (1/5) = $500 × 0.01261 × 0.20 = $1.26. This means each trade should contribute approximately $1.26 of daily portfolio volatility.

For a stock trading at $50 with a 14-day ATR of $1.50, the ATR-based share calculation is: Raw Shares = $1.26 / ($1.50 × $50) = $1.26 / $75 = 0.017 shares. This is far less than one share, so the ATR formula alone would produce zero shares at this account size.

The maximum position value constraint is 15% of equity: $500 × 0.15 = $75. At $50 per share, this allows 1 share ($50 value). The system takes the minimum of the ATR-based calculation and the max position constraint, then floors to a whole number. In practice, for a $500 account, the 15% cap ($75 maximum position) will dominate the sizing for most stocks, producing 1-3 shares per trade depending on the stock price.

## PDT Rule Guide

The Pattern Day Trader rule applies to all margin accounts with equity below $25,000. A day trade is defined as buying and selling (or selling short and buying to cover) the same security on the same calendar day. Under the PDT rule, accounts are limited to 3 day trades per rolling 5-business-day window. Exceeding this limit on a live account triggers a 90-day account restriction.

The $500 paper account is well below the $25,000 threshold, making PDT enforcement critical. The system prevents PDT violations through two mechanisms. First, a minimum holding period of 3 trading days is enforced at the strategy level — the `should_exit()` function always returns False for positions held fewer than 3 days. Second, the PDTGuard class tracks the rolling day trade count (sourced from Alpaca's account API) and blocks any exit that would constitute a fourth day trade in the window.

The one exception is circuit breaker liquidation. When the circuit breaker triggers, all positions are liquidated immediately regardless of PDT status. The rationale is that protecting capital from a catastrophic loss takes priority over avoiding a PDT flag — a 90-day restriction on a paper account is meaningless, and on a live account, the capital preservation is worth the temporary restriction.

If a normal exit is blocked by the PDT guard, the position is held overnight and re-evaluated the next trading day. The system logs a WARNING with full context so the operator can monitor these situations.

## Quickstart

Step 1: Clone the repository and install dependencies. Run `git clone` to get the code, then `cd algotrader && pip install -r requirements.txt`. Python 3.11 or later is required. The system uses alpaca-py, pandas, numpy, sqlalchemy, apscheduler, rich, and jinja2.

Step 2: Create your Alpaca paper trading account at https://alpaca.markets. Navigate to the Paper Trading section and generate API keys. You need the API Key ID and the Secret Key.

Step 3: Set up your database. Create a free Supabase project at https://supabase.com. Go to Project Settings, then Database, then Connection Pooling. Copy the connection string with port 6543 (the PgBouncer pooler URL). If you skip this step, the system will automatically use a local SQLite database.

Step 4: Copy `.env.example` to `.env` and fill in your credentials. Set `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and optionally `SUPABASE_POOLER_URL`.

Step 5: Start paper trading with `python main.py --paper`. The system will initialize the database, verify the Alpaca connection, and begin the scheduled trading loop. Use `python main.py --dry-run` first to verify the system runs without actually placing orders.

## Database Setup

The system uses Supabase (managed PostgreSQL) as its primary database, with automatic fallback to local SQLite. To configure Supabase, create a free project at supabase.com. Once created, navigate to Project Settings, then Database. Under Connection Pooling, you will find two connection strings. Use the one labeled "Connection string" with port 6543 — this routes through PgBouncer, which is required for long-running processes like this trading system. The direct connection (port 5432) will time out after extended idle periods.

Copy the pooler URL and paste it as the `SUPABASE_POOLER_URL` value in your `.env` file. The format is: `postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres`.

To verify the connection locally, run `python -c "from src.database.engine import init_db; init_db()"`. The output will show either "Successfully connected to Supabase PostgreSQL via supabase_pooler" or "Connected to local SQLite database" if the connection failed.

If both Supabase URLs are unavailable or misconfigured, the system automatically falls back to SQLite stored at `data/algotrader.db`. This fallback is completely transparent — all features work identically on SQLite. The active backend is logged at startup and displayed in the dashboard.

## Configuration Guide

All tunable parameters live in `config.yml`. The defaults are specifically calibrated for a $500 paper trading account.

`max_positions: 5` — Maximum concurrent open positions. Set to 5 for the $500 account to produce tradeable position sizes. At $75 max per position (15% of $500), 5 positions uses at most $375 of the $500 account. Increasing this reduces per-position size and may produce zero-share calculations.

`max_position_pct: 0.15` — Maximum percentage of equity per position. At $500, this is $75. This constraint dominates the ATR-based sizing for most stocks and effectively caps exposure per name.

`zscore_entry_threshold: -2.0` — Z-score level that triggers a buy signal. A stock must be at least 2 standard deviations below its 20-day mean. Lower values (like -2.5) produce fewer but higher-conviction signals.

`zscore_entry_freefall: -4.0` — Z-score level below which entries are blocked. Prevents buying stocks in fundamental breakdown. The window between -2.0 and -4.0 captures genuine mean-reversion opportunities.

`min_holding_days: 3` — Minimum trading days before a position can be exited. Primary PDT protection mechanism for the $500 account.

`portfolio_vol_target: 0.20` — Target annualized portfolio volatility. Set higher than institutional standard (typically 0.10-0.15) because the small account needs larger relative position sizes to generate tradeable share counts.

`daily_loss_limit: 0.03` — Circuit breaker trigger at 3% daily loss. On a $500 account, this is approximately $15. Aggressive for institutional standards but appropriate for a small paper account where a 3% move could indicate a position sizing error.

`max_drawdown_limit: 0.15` — Circuit breaker trigger at 15% drawdown from peak. On a $500 account starting from $500 peak, this triggers at $425. Provides a hard floor on losses.

## Circuit Breaker Guide

The circuit breaker monitors four conditions every 5 minutes during market hours. If any condition is triggered, the system immediately cancels all open orders, liquidates all positions at market price (bypassing PDT rules), sets a database flag halting all future trading, and writes a detailed log entry to `ALERTS.log`.

Trigger 1 — Daily Loss: If the portfolio drops more than 3% from its value at today's market open. At $500, this means a drop to $485 or below triggers liquidation. This catches catastrophic intraday events.

Trigger 2 — Weekly Loss: If the portfolio drops more than 8% from its value 5 trading days ago. At $500, this means dropping below $460 triggers liquidation. This catches sustained multi-day deterioration.

Trigger 3 — Max Drawdown: If the portfolio drops more than 15% from its all-time peak value. Starting at $500, this triggers at $425. If the portfolio grew to $550 peak, it triggers at $467.50. This provides an absolute loss floor.

Trigger 4 — Consecutive Losses: If the system records 3 or more consecutive trading days with negative P&L. This catches systematic strategy failure even when individual daily losses are small.

Monitor the circuit breaker by watching `logs/ALERTS.log` and the dashboard output. The dashboard always shows circuit breaker status.

To reset the circuit breaker after investigating the trigger: `python main.py --reset-circuit-breaker`. This clears the database flag and allows trading to resume. The reset event is logged with a timestamp for audit purposes.

## Switching to Live Trading

Before switching to live trading, complete at least one full month of paper trading and review the performance data. Verify that the strategy is generating signals, the circuit breaker has not triggered, and the system has been running reliably without crashes.

To switch to live trading, make exactly two changes in your `.env` file: Set `ALPACA_BASE_URL=https://api.alpaca.markets` (removing "paper" from the URL) and set `TRADING_MODE=live`. Then run `python main.py --live` — the system will require you to type "YES" to confirm live trading mode.

Pre-flight checklist: Verify you have reviewed at least 21 trading days of paper performance. Confirm the Sharpe ratio is positive (or understand why it is not). Verify the circuit breaker has not been triggered during the paper period. Check that PDT violations have been zero. Confirm your live Alpaca account has the intended capital deposited. Run `python main.py --dry-run --live` first to verify the system connects to the live API without placing orders.

The system is strongly recommended to run on paper for significantly longer than one month before considering live capital. Mean reversion strategies can have extended periods of drawdown during trending markets, and a single month may not capture a full market cycle.

## Performance Report

Run `python reports/dashboard.py` to generate both a terminal report (using rich) and an HTML report saved to `reports/output/report_YYYY-MM-DD.html`.

The Sharpe Ratio measures risk-adjusted return. It is calculated as the mean excess return (above the risk-free rate) divided by the standard deviation of returns, annualized by multiplying by sqrt(252). A Sharpe above 1.0 is considered good, above 2.0 is excellent. After only 21 trading days, the Sharpe ratio will be noisy and should not be used for definitive strategy evaluation — 60+ days provides a more reliable estimate.

The Sortino Ratio is similar to Sharpe but only penalizes downside volatility, making it more appropriate for strategies that have asymmetric return distributions.

Max Drawdown shows the largest peak-to-trough decline in portfolio value as a percentage. For the $500 account with a 15% circuit breaker, this should never exceed 15%.

Win Rate is the percentage of closed trades with positive realized P&L. Mean reversion strategies typically target 55-65% win rates with modest average wins.

## Architecture Diagram

```
                    Alpaca API
                        |
           +------------+------------+
           |                         |
     REST (bars, account)     WebSocket (live bars)
           |                         |
           v                         v
    +------+-------+          +------+-------+
    | Market Data  |          | Stream Handler|
    | (async bulk) |          | (real-time)   |
    +--------------+          +--------------+
           |
           v
    +------+-------+
    |   Universe   |     Wikipedia S&P 500 -> 6-stage filter
    |   Screener   |     -> Price -> Liquidity -> Vol -> Earnings -> Spread
    +--------------+
           |
           v
    +------+-------+
    |  Regime      |     SPY vs 200-day SMA + VIXY level
    |  Filter      |     -> BULL / NEUTRAL / BEAR
    +--------------+
           |
           v
    +------+-------+
    | Mean Reversion|    Z-score + RSI -> BUY / HOLD / EXIT
    | Strategy      |    -> Freefall protection -> PDT check
    +---------------+
           |
           v
    +------+-------+
    | Risk Manager |     PDT Guard -> Position Sizer -> Stop Manager
    | (4 modules)  |     -> Circuit Breaker (4 triggers)
    +--------------+
           |
           v
    +------+-------+
    | Order Manager|     Bracket orders -> Fill tracking
    |              |     -> Emergency liquidation
    +--------------+
           |
           v
    +------+-------+
    | Database     |     Supabase PostgreSQL (primary)
    | (SQLAlchemy) |     SQLite (automatic fallback)
    +--------------+
           |
           v
    +------+-------+
    | Dashboard    |     Rich terminal + Jinja2 HTML report
    | (reports/)   |     Sharpe, Sortino, Drawdown, Win Rate
    +--------------+
```

## Known Limitations

The $500 account size severely limits trading frequency and position sizes. The ATR-based position sizer often produces zero shares for stocks above $75, meaning the tradeable universe is effectively filtered further by price. Most trades will be 1-2 shares, making commission and spread impact proportionally larger than in larger accounts.

Mean reversion fundamentally fails during sustained trends. The regime filter mitigates this by blocking new entries during BEAR markets, but NEUTRAL regime entries at 50% size can still produce losses during extended downtrends that do not yet breach the 200-day SMA threshold.

The PDT rule limits the account to 3 day trades per week, reducing the strategy's ability to manage risk through quick exits. The 3-day minimum holding period means the system must hold through short-term volatility even when the strategy would otherwise recommend an exit.

Paper trading does not accurately model real-world slippage, partial fills, or market impact. The system estimates slippage but does not simulate it — live trading with even 1-2 shares should see minimal market impact, but the fills may differ from the paper trading prices.

The VIXY ETF used as a VIX proxy has tracking error relative to the actual VIX index and experiences contango decay in calm markets. This means the regime filter thresholds may need recalibration over time as VIXY's absolute price level drifts.

The earnings blackout filter relies on external data sources that may not be available in paper trading mode. The current implementation defers earnings checking and may allow trades near earnings announcements.
