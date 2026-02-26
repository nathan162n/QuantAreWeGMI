"""Backtest engine for offline strategy simulation.

Runs the dual-layer strategy on real Alpaca historical daily bars to
evaluate performance without a live market. Signals are computed at bar
close and executed at the NEXT bar's open price (no look-ahead bias).

Layer 1 — Mean Reversion (adapted to daily bars):
    Entry: z-score(20d) < -2.0  AND  RSI(14) < 35
    Exit:  z-score > -0.5  OR  RSI > 65  OR  held >= MAX_HOLD_DAYS

Layer 2 — Momentum Breakout (SMA(20) used as VWAP proxy on daily bars):
    Entry: close > SMA(20) + ATR(14)*1.0  AND  volume > 1.5x avg  AND  RSI < 80
    Exit:  close < SMA(20)  OR  held >= MAX_HOLD_DAYS

Usage:
    python main.py --nighttest
    python main.py --nighttest --backtest-days 180
"""

import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.indicators import rsi, atr, rolling_zscore, sma
from src.broker.alpaca_client import get_client
from src.utils.logger import get_logger

logger = get_logger("backtest.engine")

# ── Strategy parameters ────────────────────────────────────────────────────────

WARMUP_BARS      = 25     # Bars needed before any signals are valid
MAX_POSITIONS    = 5      # Max concurrent open positions
MAX_POS_PCT      = 0.15   # Max 15% of equity per position
SLIPPAGE_PCT     = 0.001  # 0.1% each way (simulates bid-ask spread)
MIN_HOLD_DAYS    = 3      # PDT guard: minimum holding period
MAX_HOLD_DAYS    = 15     # Force-exit after this many days
CIRCUIT_BREAK_DD = 0.15   # Halt backtest if drawdown exceeds 15%
VOL_TARGET       = 0.15   # 15% annualized vol target for ATR sizing
ATR_MULT         = 1.5    # ATR multiplier for position sizing

# Layer 1 (mean reversion) thresholds
L1_Z_PERIOD    = 20
L1_RSI_PERIOD  = 14
L1_ENTRY_Z_LO  = -4.0   # Freefall filter: don't catch falling knives
L1_ENTRY_Z_HI  = -2.0   # Entry z-score threshold
L1_ENTRY_RSI   = 35
L1_EXIT_Z      = -0.5
L1_EXIT_RSI    = 65

# Layer 2 (momentum breakout) thresholds
L2_SMA_PERIOD    = 20    # SMA(20) as daily VWAP proxy
L2_ATR_PERIOD    = 14
L2_ATR_MULT      = 1.0   # Breakout = close > SMA + ATR * this
L2_VOL_PERIOD    = 20
L2_VOL_RATIO     = 1.5   # Volume must exceed N * rolling average
L2_RSI_PERIOD    = 14
L2_ENTRY_RSI_MAX = 80


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    layer: str
    entry_date: date
    entry_price: float
    shares: int
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price * (1 + SLIPPAGE_PCT)


@dataclass
class ClosedTrade:
    symbol: str
    layer: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: int
    exit_reason: str
    pnl: float = 0.0
    pnl_pct: float = 0.0

    @property
    def days_held(self) -> int:
        return (self.exit_date - self.entry_date).days


# ── Portfolio simulator ────────────────────────────────────────────────────────

class SimulatedPortfolio:
    """Simulates a $500 brokerage account with PDT compliance and circuit breaker."""

    def __init__(self, initial_equity: float = 500.0):
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.positions: Dict[str, Position] = {}
        self.trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[date, float]] = []
        self._peak = initial_equity
        self.max_drawdown = 0.0
        self.daily_returns: List[float] = []
        self._prev_equity = initial_equity
        self.circuit_broken = False
        self.circuit_break_date: Optional[date] = None

    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def can_open(self, symbol: str) -> bool:
        return (
            symbol not in self.positions
            and len(self.positions) < MAX_POSITIONS
            and not self.circuit_broken
        )

    def pdt_ok(self, symbol: str, today: date) -> bool:
        """Return True if PDT minimum hold period has been satisfied."""
        if symbol not in self.positions:
            return False
        return (today - self.positions[symbol].entry_date).days >= MIN_HOLD_DAYS

    def open_position(
        self, symbol: str, shares: int, price: float, trade_date: date, layer: str = "layer1"
    ) -> bool:
        cost = shares * price * (1 + SLIPPAGE_PCT)
        if cost > self.cash or shares <= 0:
            return False
        self.positions[symbol] = Position(
            symbol=symbol, layer=layer, entry_date=trade_date,
            entry_price=price, shares=shares, current_price=price,
        )
        self.cash -= cost
        return True

    def close_position(
        self, symbol: str, price: float, trade_date: date, reason: str = ""
    ) -> Optional[ClosedTrade]:
        if symbol not in self.positions:
            return None
        pos = self.positions.pop(symbol)
        proceeds = pos.shares * price * (1 - SLIPPAGE_PCT)
        self.cash += proceeds
        pnl = proceeds - pos.cost_basis
        pnl_pct = (price / pos.entry_price - 1) * 100
        trade = ClosedTrade(
            symbol=symbol, layer=pos.layer,
            entry_date=pos.entry_date, entry_price=pos.entry_price,
            exit_date=trade_date, exit_price=price,
            shares=pos.shares, exit_reason=reason, pnl=pnl, pnl_pct=pnl_pct,
        )
        self.trades.append(trade)
        return trade

    def mark_to_market(self, prices: Dict[str, float], today: date) -> None:
        for sym, pos in self.positions.items():
            if sym in prices:
                pos.current_price = prices[sym]
        eq = self.equity()
        self.equity_curve.append((today, eq))
        if eq > self._peak:
            self._peak = eq
        dd = (self._peak - eq) / self._peak if self._peak > 0 else 0.0
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        if dd >= CIRCUIT_BREAK_DD and not self.circuit_broken:
            self.circuit_broken = True
            self.circuit_break_date = today
        ret = (eq / self._prev_equity - 1) if self._prev_equity > 0 else 0.0
        self.daily_returns.append(ret)
        self._prev_equity = eq

    def stats(self) -> dict:
        final_eq = self.equity_curve[-1][1] if self.equity_curve else self.initial_equity
        n_days = len(self.equity_curve)
        total_ret = (final_eq / self.initial_equity - 1) * 100
        ann_ret = (
            ((final_eq / self.initial_equity) ** (252 / n_days) - 1) * 100
            if n_days > 0 else 0.0
        )
        daily = pd.Series(self.daily_returns)
        sharpe = (
            daily.mean() / daily.std() * math.sqrt(252)
            if daily.std() > 0 else 0.0
        )
        neg = daily[daily < 0]
        sortino = (
            daily.mean() / neg.std() * math.sqrt(252)
            if len(neg) > 0 and neg.std() > 0 else 0.0
        )
        calmar = ann_ret / (self.max_drawdown * 100) if self.max_drawdown > 0 else 0.0

        trades = self.trades
        base = {
            "initial_equity": self.initial_equity,
            "final_equity": final_eq,
            "total_return_pct": total_ret,
            "ann_return_pct": ann_ret,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown_pct": self.max_drawdown * 100,
            "equity_curve": self.equity_curve,
            "closed_trades": trades,
            "circuit_broken": self.circuit_broken,
            "circuit_break_date": self.circuit_break_date,
            "open_at_end": len(self.positions),
            "n_sim_days": n_days,
        }
        if not trades:
            return {**base, "total_trades": 0, "win_rate_pct": 0, "avg_win_pct": 0,
                    "avg_loss_pct": 0, "profit_factor": 0, "avg_hold_days": 0,
                    "layer1_trades": 0, "layer2_trades": 0, "best": None, "worst": None}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) * 100
        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
        gross_wins = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        avg_hold = sum(t.days_held for t in trades) / len(trades)
        best = max(trades, key=lambda t: t.pnl_pct)
        worst = min(trades, key=lambda t: t.pnl_pct)
        l1 = [t for t in trades if t.layer == "layer1"]
        l2 = [t for t in trades if t.layer == "layer2"]

        return {
            **base,
            "total_trades": len(trades),
            "win_rate_pct": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "avg_hold_days": avg_hold,
            "best": best,
            "worst": worst,
            "layer1_trades": len(l1),
            "layer2_trades": len(l2),
        }


# ── Backtest engine ────────────────────────────────────────────────────────────

class BacktestEngine:
    """Dual-layer backtester using Alpaca historical daily OHLCV bars.

    Adapts the live mean-reversion (Layer 1) and momentum breakout (Layer 2)
    strategies to daily bars. Signals computed at close of day t are executed
    at the open of day t+1 to prevent look-ahead bias.
    """

    def __init__(self, initial_equity: float = 500.0, n_days: int = 365):
        self.initial_equity = initial_equity
        self.n_days = n_days
        self.portfolio = SimulatedPortfolio(initial_equity)

    # ── Data fetching ──────────────────────────────────────────────────────────

    def _fetch_bars(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch daily OHLCV bars for all symbols in parallel via Alpaca."""
        # Use explicit start date — limit-only requests silently return empty
        # on some alpaca-py SDK versions. Buffer by 1.5x for weekends/holidays.
        start_date = datetime.now(timezone.utc) - timedelta(days=int(self.n_days * 1.5) + 90)
        client = get_client()
        data: Dict[str, pd.DataFrame] = {}

        _first_error: List[str] = []  # capture first error for diagnostics

        def fetch(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            try:
                bars = client.get_bars(sym, "1Day", start=start_date)
                if bars is not None and not bars.empty:
                    return sym, bars
            except Exception as e:
                err_str = str(e)
                if not _first_error:
                    _first_error.append(err_str)
                logger.debug("Fetch failed for %s: %s", sym, e)
            return sym, None

        print(f"\nFetching {len(symbols)} symbols ", end="", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(fetch, s): s for s in symbols}
            done = 0
            for fut in as_completed(futures):
                sym, bars = fut.result()
                if bars is not None:
                    data[sym] = bars
                done += 1
                if done % 5 == 0:
                    print(".", end="", flush=True)
        print(f" done  ({len(data)}/{len(symbols)} symbols)\n")
        if not data and _first_error:
            print(f"  First fetch error: {_first_error[0]}")
            if "401" in _first_error[0] or "unauthorized" in _first_error[0].lower():
                print("  -> Your Alpaca API keys are being rejected (401 Unauthorized).")
                print("     Real paper keys start with 'PK' and are found at:")
                print("     https://app.alpaca.markets/paper/dashboard/overview")
        return data

    # ── Indicator computation ──────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized indicator computation on the full DataFrame."""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        df["zscore"]  = rolling_zscore(close, L1_Z_PERIOD)
        df["rsi"]     = rsi(close, L1_RSI_PERIOD)
        df["atr"]     = atr(high, low, close, L2_ATR_PERIOD)
        df["sma20"]   = sma(close, L2_SMA_PERIOD)
        df["vol_avg"] = vol.rolling(L2_VOL_PERIOD).mean().values
        return df

    # ── Signal logic ───────────────────────────────────────────────────────────

    @staticmethod
    def _l1_entry(row: pd.Series) -> bool:
        z = row.get("zscore")
        r = row.get("rsi")
        if pd.isna(z) or pd.isna(r):
            return False
        return L1_ENTRY_Z_LO < z < L1_ENTRY_Z_HI and r < L1_ENTRY_RSI

    @staticmethod
    def _l1_exit(row: pd.Series) -> Tuple[bool, str]:
        z = row.get("zscore")
        r = row.get("rsi")
        if pd.isna(z) or pd.isna(r):
            return False, ""
        if z >= L1_EXIT_Z:
            return True, "zscore_recovery"
        if r >= L1_EXIT_RSI:
            return True, "rsi_overbought"
        return False, ""

    @staticmethod
    def _l2_entry(row: pd.Series) -> bool:
        c   = row.get("close", 0.0)
        s20 = row.get("sma20", 0.0)
        a   = row.get("atr", 0.0)
        v   = row.get("volume", 0.0)
        va  = row.get("vol_avg", 1.0)
        r   = row.get("rsi", 50.0)
        if any(pd.isna(x) for x in [c, s20, a, v, va, r]):
            return False
        if va <= 0 or a <= 0:
            return False
        return (
            c > s20 + a * L2_ATR_MULT
            and v > va * L2_VOL_RATIO
            and r < L2_ENTRY_RSI_MAX
        )

    @staticmethod
    def _l2_exit(row: pd.Series, _pos: Position) -> Tuple[bool, str]:
        c = row.get("close", 0.0)
        s = row.get("sma20", 0.0)
        if pd.isna(c) or pd.isna(s):
            return False, ""
        return (True, "price_below_sma") if c < s else (False, "")

    # ── Position sizing ────────────────────────────────────────────────────────

    def _size(self, atr_val: float, exec_price: float) -> int:
        """ATR-normalized volatility-targeted sizing, capped at MAX_POS_PCT."""
        eq = self.portfolio.equity()
        n_open = max(1, MAX_POSITIONS - len(self.portfolio.positions))
        dollar_risk = eq * (VOL_TARGET / math.sqrt(252)) * (1.0 / n_open)
        shares = int(dollar_risk / (atr_val * ATR_MULT))
        max_by_pct  = int((eq * MAX_POS_PCT) / exec_price)
        max_by_cash = int(self.portfolio.cash / (exec_price * (1 + SLIPPAGE_PCT)))
        return max(0, min(shares, max_by_pct, max_by_cash))

    # ── Main simulation loop ───────────────────────────────────────────────────

    def run(self, symbols: List[str]) -> dict:
        """Fetch data, run simulation, return stats dict."""

        # 1. Fetch daily bars
        raw = self._fetch_bars(symbols)
        if not raw:
            return {"error": "No data fetched — check your Alpaca API keys"}

        # 2. Compute all indicators upfront (vectorized, not in the loop)
        print("Computing indicators... ", end="", flush=True)
        data: Dict[str, pd.DataFrame] = {}
        for sym, df in raw.items():
            try:
                data[sym] = self._add_indicators(df)
            except Exception as e:
                logger.debug("Indicator error %s: %s", sym, e)
        print("done\n")

        # 3. Build O(1) date-indexed lookup: {symbol: {date: row}}
        lookup: Dict[str, Dict[date, pd.Series]] = {}
        for sym, df in data.items():
            lookup[sym] = {}
            for idx_val in df.index:
                d = idx_val.date() if hasattr(idx_val, "date") else idx_val
                lookup[sym][d] = df.loc[idx_val]

        # 4. Unified sorted date list, trimmed to n_days
        all_dates = sorted({d for dm in lookup.values() for d in dm})
        if len(all_dates) > self.n_days:
            all_dates = all_dates[-self.n_days:]

        sim_dates = all_dates[WARMUP_BARS:]
        if not sim_dates:
            return {"error": "Insufficient data after warmup period"}

        print(f"Simulation: {sim_dates[0]}  →  {sim_dates[-1]}  ({len(sim_dates)} trading days)")
        print(f"Universe:   {len(data)} symbols  |  Initial equity: ${self.initial_equity:.2f}\n")

        bar_width = 45

        # 5. Day-by-day simulation
        for step, today in enumerate(sim_dates[:-1]):
            tomorrow = sim_dates[step + 1]

            # Progress bar
            pct = (step + 1) / (len(sim_dates) - 1) * 100
            filled = int(bar_width * pct / 100)
            bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
            print(f"\r  [{bar}] {pct:5.1f}%  Equity: ${self.portfolio.equity():8.2f}", end="", flush=True)

            if self.portfolio.circuit_broken:
                print(f"\n\n  [!] Circuit breaker triggered on {self.portfolio.circuit_break_date}  (drawdown > {CIRCUIT_BREAK_DD*100:.0f}%)")
                break

            exits: List[Tuple[str, str]] = []    # (symbol, exit_reason)
            entries: List[Tuple[str, str]] = []  # (symbol, layer)

            for sym in data:
                row = lookup[sym].get(today)
                if row is None:
                    continue

                if sym in self.portfolio.positions:
                    pos = self.portfolio.positions[sym]
                    days_held = (today - pos.entry_date).days

                    if days_held >= MAX_HOLD_DAYS:
                        exits.append((sym, "max_hold_days"))
                        continue

                    if not self.portfolio.pdt_ok(sym, today):
                        continue  # PDT guard: must hold at least MIN_HOLD_DAYS

                    if pos.layer == "layer1":
                        ok, reason = self._l1_exit(row)
                    else:
                        ok, reason = self._l2_exit(row, pos)

                    if ok:
                        exits.append((sym, reason))

                elif self.portfolio.can_open(sym):
                    if self._l1_entry(row):
                        entries.append((sym, "layer1"))
                    elif self._l2_entry(row):
                        entries.append((sym, "layer2"))

            # Execute exits at tomorrow's open (no look-ahead bias)
            for sym, reason in exits:
                nrow = lookup[sym].get(tomorrow)
                if nrow is None:
                    continue
                exec_price = nrow.get("open") or nrow.get("close")
                if exec_price is not None and pd.notna(exec_price) and exec_price > 0:
                    self.portfolio.close_position(sym, float(exec_price), tomorrow, reason)

            # Execute entries at tomorrow's open (no look-ahead bias)
            for sym, layer in entries:
                if not self.portfolio.can_open(sym):
                    break
                nrow = lookup[sym].get(tomorrow)
                if nrow is None:
                    continue
                exec_price = nrow.get("open") or nrow.get("close")
                if exec_price is None or pd.isna(exec_price) or exec_price <= 0:
                    continue
                today_row = lookup[sym].get(today)
                if today_row is None:
                    continue
                atr_val = today_row.get("atr")
                if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
                    continue
                shares = self._size(float(atr_val), float(exec_price))
                if shares > 0:
                    self.portfolio.open_position(sym, shares, float(exec_price), tomorrow, layer)

            # Mark to market at today's close
            prices = {
                sym: float(lookup[sym][today]["close"])
                for sym in data
                if today in lookup[sym] and pd.notna(lookup[sym][today].get("close"))
            }
            self.portfolio.mark_to_market(prices, today)

        print()  # end progress bar line
        return self.portfolio.stats()


# ── Report printing & CSV export ───────────────────────────────────────────────

def print_report(stats: dict) -> None:
    """Print a formatted backtest results report to the terminal."""
    W = 66
    line = "=" * W

    def _sign(v: float) -> str:
        return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"

    def _color_sign(v: float) -> str:
        s = _sign(v)
        # ANSI green/red (works in Windows Terminal & most modern terminals)
        if v > 0:
            return f"\033[92m{s}\033[0m"
        elif v < 0:
            return f"\033[91m{s}\033[0m"
        return s

    print(f"\n{line}")
    print(f"  ALGOTRADER — BACKTEST RESULTS".center(W))
    if stats.get("equity_curve"):
        d0, d1 = stats["equity_curve"][0][0], stats["equity_curve"][-1][0]
        print(f"  {d0}  →  {d1}  |  {stats['n_sim_days']} trading days".center(W))
    print(line)

    i_eq = stats["initial_equity"]
    f_eq = stats["final_equity"]
    tr   = stats["total_return_pct"]
    ar   = stats["ann_return_pct"]
    sh   = stats["sharpe"]
    so   = stats["sortino"]
    ca   = stats["calmar"]
    dd   = stats["max_drawdown_pct"]

    print()
    print(f"  {'RETURNS':<32}  {'RISK'}")
    print(f"  {'-'*31}  {'-'*28}")
    print(f"  {'Starting Capital:':<24}  ${i_eq:>8.2f}    {'Sharpe Ratio:':<20} {sh:>6.2f}")
    print(f"  {'Ending Capital:':<24}  ${f_eq:>8.2f}    {'Sortino Ratio:':<20} {so:>6.2f}")
    print(f"  {'Total Return:':<24}  {_color_sign(tr):>15}    {'Calmar Ratio:':<20} {ca:>6.2f}")
    print(f"  {'Annualized Return:':<24}  {_color_sign(ar):>15}    {'Max Drawdown:':<20} {_color_sign(-dd):>15}")

    if stats.get("total_trades", 0) == 0:
        print(f"\n  No trades executed during this period.\n{line}\n")
        return

    nt   = stats["total_trades"]
    wr   = stats["win_rate_pct"]
    aw   = stats["avg_win_pct"]
    al   = stats["avg_loss_pct"]
    pf   = stats["profit_factor"]
    ah   = stats["avg_hold_days"]
    l1   = stats["layer1_trades"]
    l2   = stats["layer2_trades"]
    best = stats["best"]
    worst = stats["worst"]

    print()
    print(f"  TRADE STATISTICS")
    print(f"  {'-'*62}")
    print(f"  {'Total Closed Trades:':<26} {nt:<8}   {'Avg Holding Period:':<22} {ah:.1f} days")
    print(f"  {'Win Rate:':<26} {wr:.1f}%{'':<5}   {'Profit Factor:':<22} {pf:.2f}")
    print(f"  {'Avg Win:':<26} {_color_sign(aw):>15}   {'Layer 1 (Mean Rev):':<22} {l1} trades")
    print(f"  {'Avg Loss:':<26} {_color_sign(al):>15}   {'Layer 2 (Momentum):':<22} {l2} trades")
    if best:
        print(f"\n  Best Trade:   {_color_sign(best.pnl_pct):>10}  {best.symbol:<6} "
              f"({best.entry_date} → {best.exit_date})  [{best.exit_reason}]")
    if worst:
        print(f"  Worst Trade:  {_color_sign(worst.pnl_pct):>10}  {worst.symbol:<6} "
              f"({worst.entry_date} → {worst.exit_date})  [{worst.exit_reason}]")

    if stats.get("circuit_broken"):
        print(f"\n  [!] Circuit breaker triggered on {stats['circuit_break_date']}")

    if stats.get("open_at_end", 0) > 0:
        print(f"\n  {stats['open_at_end']} position(s) still open at end of backtest (unrealized P&L not counted)")

    # Recent trades table
    trades = stats.get("closed_trades", [])
    recent = trades[-10:]
    if recent:
        print(f"\n  RECENT TRADES  (last {len(recent)})")
        print(f"  {'-'*62}")
        print(f"  {'DATE IN':<12} {'DATE OUT':<12} {'SYM':<6} {'LAYER':<6} {'ENTRY':>8} {'EXIT':>8} {'PNL%':>7}  REASON")
        for t in reversed(recent):
            layer_tag = "L1-MR" if t.layer == "layer1" else "L2-MB"
            pnl_str = _color_sign(t.pnl_pct)
            print(f"  {str(t.entry_date):<12} {str(t.exit_date):<12} {t.symbol:<6} "
                  f"{layer_tag:<6} ${t.entry_price:>7.2f} ${t.exit_price:>7.2f} {pnl_str:>14}  {t.exit_reason}")

    print(f"\n{line}\n")


def save_results(stats: dict) -> str:
    """Save equity curve and trade log to CSV files in reports/. Returns equity CSV path."""
    os.makedirs("reports", exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")

    # Equity curve
    eq_path = f"reports/backtest_{today_str}_equity.csv"
    with open(eq_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "equity"])
        for d, eq in stats.get("equity_curve", []):
            writer.writerow([d, f"{eq:.4f}"])

    # Trade log
    trades_path = f"reports/backtest_{today_str}_trades.csv"
    trades = stats.get("closed_trades", [])
    if trades:
        with open(trades_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "layer", "entry_date", "entry_price",
                              "exit_date", "exit_price", "shares",
                              "pnl_usd", "pnl_pct", "days_held", "exit_reason"])
            for t in trades:
                writer.writerow([
                    t.symbol, t.layer, t.entry_date, f"{t.entry_price:.4f}",
                    t.exit_date, f"{t.exit_price:.4f}", t.shares,
                    f"{t.pnl:.4f}", f"{t.pnl_pct:.4f}", t.days_held, t.exit_reason,
                ])

    print(f"  Equity curve saved: {eq_path}")
    if trades:
        print(f"  Trade log saved:    {trades_path}")
    return eq_path
