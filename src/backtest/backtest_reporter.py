"""Backtest reporter — formats and displays simulation results."""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger("backtest.reporter")


class BacktestReporter:
    """Formats backtest results into the standard display format.

    Output format matches the specification in prompt.md:
    - Returns section with capital, return, annualized return
    - Risk section with Sharpe, Sortino, Calmar, max drawdown
    - Trade statistics with win rate, profit factor, avg hold
    - Per-layer breakdown
    - Top/worst performing symbols
    - Recent trades table
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir

    def generate_report(
        self,
        trade_history: List[dict],
        equity_curve: List[dict],
        start_date,
        end_date,
        initial_equity: float = 500.0,
        pdt_blocks: int = 0,
        eod_closes: int = 0,
    ) -> str:
        """Generate the full backtest report string.

        Args:
            trade_history: List of trade dicts from BacktestBroker.
            equity_curve: List of {date, equity} dicts.
            start_date: Backtest start date.
            end_date: Backtest end date.
            initial_equity: Starting capital.
            pdt_blocks: Number of PDT-blocked trades.
            eod_closes: Number of EOD forced closes.

        Returns:
            str: Formatted report string ready to print.
        """
        metrics = self._compute_metrics(trade_history, equity_curve, initial_equity)
        layer_stats = self._per_layer_breakdown(trade_history)
        symbol_stats = self._symbol_stats(trade_history)

        final_equity = equity_curve[-1]["equity"] if equity_curve else initial_equity
        total_pnl = final_equity - initial_equity

        lines = []
        lines.append("=" * 70)
        lines.append("  BACKTEST RESULTS")
        lines.append("=" * 70)
        lines.append(f"  Period:         {start_date}  to  {end_date}")
        lines.append(f"  Initial Equity: ${initial_equity:,.2f}")
        lines.append(f"  Final Equity:   ${final_equity:,.2f}")
        lines.append("")

        # Returns section
        lines.append("-" * 70)
        lines.append("  RETURNS")
        lines.append("-" * 70)
        total_ret = metrics["total_return_pct"]
        ann_ret = metrics["annualized_return_pct"]
        lines.append(f"  Total Return:        {total_ret:+.2f}%")
        lines.append(f"  Annualized Return:   {ann_ret:+.2f}%")
        lines.append(f"  Net P&L:             ${total_pnl:+.2f}")
        lines.append("")

        # Risk section
        lines.append("-" * 70)
        lines.append("  RISK METRICS")
        lines.append("-" * 70)
        lines.append(f"  Sharpe Ratio:        {metrics['sharpe']:.4f}  (annualized, 252 days)")
        lines.append(f"  Sortino Ratio:       {metrics['sortino']:.4f}")
        lines.append(f"  Calmar Ratio:        {metrics['calmar']:.4f}")
        lines.append(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        lines.append(f"  Best Day:            ${metrics['best_day']:+.2f}")
        lines.append(f"  Worst Day:           ${metrics['worst_day']:+.2f}")
        lines.append(f"  Avg Daily P&L:       ${metrics['avg_daily_pnl']:+.2f}")
        lines.append("")

        # Trade statistics
        lines.append("-" * 70)
        lines.append("  TRADE STATISTICS")
        lines.append("-" * 70)
        lines.append(f"  Total Trades:        {metrics['total_trades']}")
        lines.append(f"  Winning Trades:      {metrics['winning_trades']}")
        lines.append(f"  Losing Trades:       {metrics['losing_trades']}")
        win_rate = metrics["win_rate"]
        lines.append(f"  Win Rate:            {win_rate:.1f}%")
        pf = metrics["profit_factor"]
        pf_str = f"{pf:.4f}" if pf != float("inf") else "inf (no losing trades)"
        lines.append(f"  Profit Factor:       {pf_str}")
        lines.append(f"  Avg Hold Time:       {metrics['avg_hold_minutes']:.1f} minutes")
        lines.append(f"  Avg Win:             ${metrics['avg_win']:+.2f}")
        lines.append(f"  Avg Loss:            ${metrics['avg_loss']:+.2f}")
        lines.append(f"  PDT Blocks:          {pdt_blocks}")
        lines.append(f"  EOD Forced Closes:   {eod_closes}")
        lines.append("")

        # Per-layer breakdown
        lines.append("-" * 70)
        lines.append("  PER-LAYER BREAKDOWN")
        lines.append("-" * 70)
        if layer_stats:
            header = f"  {'Layer':<18} {'Trades':>7} {'Win%':>7} {'AvgHold':>10} {'AvgPnL':>9} {'TotalPnL':>10}"
            lines.append(header)
            lines.append("  " + "-" * 62)
            for layer_name, ls in sorted(layer_stats.items()):
                row = (
                    f"  {layer_name:<18}"
                    f" {ls['trades']:>7}"
                    f" {ls['win_rate']:>6.1f}%"
                    f" {ls['avg_hold_minutes']:>9.1f}m"
                    f" ${ls['avg_pnl']:>+8.2f}"
                    f" ${ls['total_pnl']:>+9.2f}"
                )
                lines.append(row)
        else:
            lines.append("  No layer data available.")
        lines.append("")

        # Symbol stats — top 5 and worst 5
        lines.append("-" * 70)
        lines.append("  SYMBOL PERFORMANCE")
        lines.append("-" * 70)
        if symbol_stats:
            header = f"  {'Symbol':<8} {'Trades':>7} {'Win%':>7} {'TotalPnL':>10}"
            lines.append(header)
            lines.append("  " + "-" * 36)
            top5 = symbol_stats[:5]
            worst5 = symbol_stats[-5:] if len(symbol_stats) > 5 else []
            lines.append("  -- Top performers --")
            for ss in top5:
                lines.append(
                    f"  {ss['symbol']:<8}"
                    f" {ss['trades']:>7}"
                    f" {ss['win_rate']:>6.1f}%"
                    f" ${ss['total_pnl']:>+9.2f}"
                )
            if worst5:
                lines.append("  -- Worst performers --")
                for ss in worst5:
                    lines.append(
                        f"  {ss['symbol']:<8}"
                        f" {ss['trades']:>7}"
                        f" {ss['win_rate']:>6.1f}%"
                        f" ${ss['total_pnl']:>+9.2f}"
                    )
        else:
            lines.append("  No symbol data available.")
        lines.append("")

        # Recent trades table (last 20)
        lines.append("-" * 70)
        lines.append("  RECENT TRADES (last 20)")
        lines.append("-" * 70)
        recent = trade_history[-20:] if len(trade_history) >= 20 else trade_history
        if recent:
            hdr = (
                f"  {'Symbol':<6} {'Layer':<16} {'Entry':>10} {'Exit':>10}"
                f" {'Qty':>4} {'PnL':>8} {'Hold':>7} {'Reason':<10}"
            )
            lines.append(hdr)
            lines.append("  " + "-" * 75)
            for t in recent:
                entry_t = t.get("entry_time", "")
                exit_t = t.get("exit_time", "")
                # Format timestamps to HH:MM if datetime
                if isinstance(entry_t, datetime):
                    entry_t = entry_t.strftime("%H:%M")
                if isinstance(exit_t, datetime):
                    exit_t = exit_t.strftime("%H:%M")
                row = (
                    f"  {t.get('symbol','?'):<6}"
                    f" {t.get('layer_name','?'):<16}"
                    f" {str(entry_t):>10}"
                    f" {str(exit_t):>10}"
                    f" {t.get('qty',0):>4}"
                    f" ${t.get('pnl',0):>+7.2f}"
                    f" {t.get('hold_minutes',0):>6.0f}m"
                    f" {t.get('exit_reason','?'):<10}"
                )
                lines.append(row)
        else:
            lines.append("  No trades recorded.")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save_csv_files(
        self,
        trade_history: List[dict],
        equity_curve: List[dict],
        date_suffix: str,
    ) -> tuple:
        """Save trade log and equity curve as CSV files.

        Args:
            trade_history: List of trade dicts.
            equity_curve: List of {date, equity} dicts.
            date_suffix: Date string for filename.

        Returns:
            tuple: (trades_path, equity_path) file paths.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        trades_path = os.path.join(self.output_dir, f"trades_{date_suffix}.csv")
        equity_path = os.path.join(self.output_dir, f"equity_{date_suffix}.csv")

        # Write trades CSV
        if trade_history:
            fieldnames = [
                "symbol", "layer_name", "entry_time", "exit_time",
                "entry_price", "exit_price", "qty", "pnl",
                "slippage", "exit_reason", "hold_minutes",
            ]
            # Ensure all keys exist in each row
            with open(trades_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for trade in trade_history:
                    row = {k: trade.get(k, "") for k in fieldnames}
                    writer.writerow(row)
        else:
            # Write header-only file
            with open(trades_path, "w", newline="", encoding="utf-8") as f:
                f.write("symbol,layer_name,entry_time,exit_time,entry_price,exit_price,qty,pnl,slippage,exit_reason,hold_minutes\n")

        # Write equity CSV
        with open(equity_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "equity"])
            writer.writeheader()
            for row in equity_curve:
                writer.writerow({"date": row.get("date", ""), "equity": row.get("equity", 0)})

        logger.info("Saved trades CSV: %s", trades_path)
        logger.info("Saved equity CSV: %s", equity_path)

        return trades_path, equity_path

    def _compute_metrics(
        self,
        trade_history: List[dict],
        equity_curve: List[dict],
        initial_equity: float,
    ) -> dict:
        """Compute all performance metrics from trade history.

        Returns dict with:
            total_return_pct, annualized_return_pct, sharpe, sortino, calmar,
            max_drawdown_pct, win_rate, profit_factor, avg_hold_minutes,
            avg_win, avg_loss, best_day, worst_day, avg_daily_pnl,
            total_trades, winning_trades, losing_trades
        """
        import math

        try:
            import numpy as np
            _has_numpy = True
        except ImportError:
            _has_numpy = False

        total_trades = len(trade_history)
        winning_trades = sum(1 for t in trade_history if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in trade_history if t.get("pnl", 0) < 0)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        pnls = [t.get("pnl", 0.0) for t in trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

        hold_times = [t.get("hold_minutes", 0.0) for t in trade_history]
        avg_hold_minutes = sum(hold_times) / len(hold_times) if hold_times else 0.0

        # Final and initial equity
        final_equity = equity_curve[-1]["equity"] if equity_curve else initial_equity
        total_return_pct = ((final_equity - initial_equity) / initial_equity * 100) if initial_equity > 0 else 0.0

        # Number of trading days in the curve
        n_days = len(equity_curve)
        if n_days > 1:
            annualized_return_pct = total_return_pct * (252.0 / n_days)
        else:
            annualized_return_pct = total_return_pct

        # Max drawdown from equity curve
        max_drawdown_pct = 0.0
        if equity_curve:
            equities = [e["equity"] for e in equity_curve]
            peak = equities[0]
            for eq in equities:
                if eq > peak:
                    peak = eq
                if peak > 0:
                    dd = (peak - eq) / peak * 100
                    if dd > max_drawdown_pct:
                        max_drawdown_pct = dd

        # Daily P&L series from equity curve
        daily_pnl_series = []
        if len(equity_curve) >= 2:
            for i in range(1, len(equity_curve)):
                daily_pnl_series.append(
                    equity_curve[i]["equity"] - equity_curve[i - 1]["equity"]
                )

        best_day = max(daily_pnl_series) if daily_pnl_series else 0.0
        worst_day = min(daily_pnl_series) if daily_pnl_series else 0.0
        avg_daily_pnl = sum(daily_pnl_series) / len(daily_pnl_series) if daily_pnl_series else 0.0

        # Sharpe and Sortino — use daily returns as pct of equity at start of each day
        sharpe = 0.0
        sortino = 0.0
        if daily_pnl_series and len(equity_curve) >= 2 and _has_numpy:
            daily_equity_start = [equity_curve[i]["equity"] for i in range(len(equity_curve) - 1)]
            daily_returns = []
            for i, pnl in enumerate(daily_pnl_series):
                base = daily_equity_start[i] if daily_equity_start[i] > 0 else initial_equity
                daily_returns.append(pnl / base)
            arr = np.array(daily_returns, dtype=float)
            mean_ret = float(np.mean(arr))
            std_ret = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 1e-10 else 0.0
            neg = arr[arr < 0]
            downside_std = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
            sortino = (mean_ret / downside_std * math.sqrt(252)) if downside_std > 1e-10 else 0.0
        elif daily_pnl_series and len(equity_curve) >= 2:
            # Fallback without numpy
            daily_equity_start = [equity_curve[i]["equity"] for i in range(len(equity_curve) - 1)]
            daily_returns = []
            for i, pnl in enumerate(daily_pnl_series):
                base = daily_equity_start[i] if daily_equity_start[i] > 0 else initial_equity
                daily_returns.append(pnl / base)
            n = len(daily_returns)
            if n > 1:
                mean_ret = sum(daily_returns) / n
                variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (n - 1)
                std_ret = variance ** 0.5
                sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 1e-10 else 0.0
                neg_returns = [r for r in daily_returns if r < 0]
                if len(neg_returns) > 1:
                    neg_mean = sum(neg_returns) / len(neg_returns)
                    neg_var = sum((r - neg_mean) ** 2 for r in neg_returns) / (len(neg_returns) - 1)
                    downside_std = neg_var ** 0.5
                    sortino = (mean_ret / downside_std * math.sqrt(252)) if downside_std > 1e-10 else 0.0

        calmar = (annualized_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

        return {
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_hold_minutes": avg_hold_minutes,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "best_day": best_day,
            "worst_day": worst_day,
            "avg_daily_pnl": avg_daily_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
        }

    def _per_layer_breakdown(self, trade_history: List[dict]) -> Dict[str, dict]:
        """Compute per-layer statistics.

        Returns dict keyed by layer_name with:
            trades, win_rate, avg_hold_minutes, avg_pnl, total_pnl
        """
        layers: Dict[str, dict] = {}

        for trade in trade_history:
            layer = trade.get("layer_name", "UNKNOWN") or "UNKNOWN"
            if layer not in layers:
                layers[layer] = {
                    "trades": 0,
                    "wins": 0,
                    "pnls": [],
                    "hold_times": [],
                }
            layers[layer]["trades"] += 1
            pnl = trade.get("pnl", 0.0)
            layers[layer]["pnls"].append(pnl)
            if pnl > 0:
                layers[layer]["wins"] += 1
            layers[layer]["hold_times"].append(trade.get("hold_minutes", 0.0))

        result = {}
        for layer_name, data in layers.items():
            t = data["trades"]
            pnls = data["pnls"]
            holds = data["hold_times"]
            result[layer_name] = {
                "trades": t,
                "win_rate": (data["wins"] / t * 100) if t > 0 else 0.0,
                "avg_hold_minutes": sum(holds) / len(holds) if holds else 0.0,
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
                "total_pnl": sum(pnls),
            }

        return result

    def _symbol_stats(self, trade_history: List[dict]) -> List[dict]:
        """Compute per-symbol statistics sorted by total P&L desc.

        Returns list of dicts with: symbol, trades, win_rate, total_pnl
        """
        symbols: Dict[str, dict] = {}

        for trade in trade_history:
            sym = trade.get("symbol", "UNKNOWN") or "UNKNOWN"
            if sym not in symbols:
                symbols[sym] = {"trades": 0, "wins": 0, "pnls": []}
            symbols[sym]["trades"] += 1
            pnl = trade.get("pnl", 0.0)
            symbols[sym]["pnls"].append(pnl)
            if pnl > 0:
                symbols[sym]["wins"] += 1

        result = []
        for sym, data in symbols.items():
            t = data["trades"]
            result.append({
                "symbol": sym,
                "trades": t,
                "win_rate": (data["wins"] / t * 100) if t > 0 else 0.0,
                "total_pnl": sum(data["pnls"]),
            })

        result.sort(key=lambda x: x["total_pnl"], reverse=True)
        return result
