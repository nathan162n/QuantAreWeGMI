"""Terminal and HTML performance dashboard.

Run standalone: python reports/dashboard.py
Generates a rich terminal report and saves an HTML report to
reports/output/report_YYYY-MM-DD.html.
"""

import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.database import repository
from src.database.engine import ACTIVE_DB_BACKEND, init_db
from src.utils.logger import get_logger

logger = get_logger("reports.dashboard")


def _compute_metrics(daily_states: list, closed_trades: list, config: dict) -> dict:
    """Compute all performance metrics from historical data.

    Args:
        daily_states: List of DailyPortfolioState records, most recent first.
        closed_trades: List of closed Trade records.
        config: Reporting configuration with risk_free_rate and starting_capital.

    Returns:
        dict: Performance metrics including sharpe, sortino, max_drawdown,
            win_rate, total_return_pct, avg_hold_days.

    Example:
        >>> metrics = _compute_metrics(states, trades, {"risk_free_rate": 0.05,
        ...     "starting_capital": 500.0})
    """
    starting_capital = config.get("starting_capital", 500.0)
    risk_free_rate = config.get("risk_free_rate", 0.05)

    metrics = {
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_trades": len(closed_trades),
        "avg_hold_days": 0.0,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
    }

    if not daily_states:
        return metrics

    sorted_states = sorted(daily_states, key=lambda s: s.date)
    portfolio_values = [float(s.portfolio_value) for s in sorted_states]
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i - 1] > 0:
            ret = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            daily_returns.append(ret)

    current_equity = portfolio_values[-1] if portfolio_values else starting_capital
    metrics["total_pnl"] = round(current_equity - starting_capital, 6)
    metrics["total_return_pct"] = round(
        (current_equity - starting_capital) / starting_capital, 6
    )

    if daily_returns:
        daily_arr = np.array(daily_returns)
        excess = daily_arr - (risk_free_rate / 252.0)

        if len(excess) > 1 and np.std(excess) > 0:
            metrics["sharpe"] = round(
                float(np.mean(excess) / np.std(excess) * math.sqrt(252)), 4
            )

        downside = excess[excess < 0]
        if len(downside) > 1 and np.std(downside) > 0:
            metrics["sortino"] = round(
                float(np.mean(excess) / np.std(downside) * math.sqrt(252)), 4
            )

        cumulative = np.array(portfolio_values)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / np.where(rolling_max > 0, rolling_max, 1)
        metrics["max_drawdown"] = round(float(np.min(drawdowns)), 6)

    if closed_trades:
        winners = [t for t in closed_trades if float(t.realized_pnl or 0) > 0]
        metrics["win_rate"] = round(len(winners) / len(closed_trades), 4)

        hold_days_list = []
        for t in closed_trades:
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                hold_days_list.append(delta.days)
        if hold_days_list:
            metrics["avg_hold_days"] = round(sum(hold_days_list) / len(hold_days_list), 1)

    return metrics


def generate_terminal_report() -> dict:
    """Generate and display a rich terminal report.

    Returns:
        dict: Report data used for both terminal and HTML output.

    Example:
        >>> data = generate_terminal_report()
        >>> data["portfolio_value"]
        500.0
    """
    import yaml
    config_path = os.path.join(project_root, "config.yml")
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    reporting_config = full_config.get("reporting", {})
    starting_capital = reporting_config.get("starting_capital", 500.0)

    daily_states = repository.get_daily_states(n_days=365)
    closed_trades = repository.get_closed_trades(limit=20)
    open_trades = repository.get_open_trades()
    cb_active = repository.is_circuit_breaker_active()
    pdt_count = repository.get_day_trades_this_week()

    sorted_states = sorted(daily_states, key=lambda s: s.date) if daily_states else []

    if sorted_states:
        latest = sorted_states[-1]
        portfolio_value = float(latest.portfolio_value)
        cash = float(latest.cash)
        equity = float(latest.equity)
        regime = latest.regime
    else:
        portfolio_value = starting_capital
        cash = starting_capital
        equity = starting_capital
        regime = "UNKNOWN"

    all_closed = repository.get_closed_trades(limit=9999)
    metrics = _compute_metrics(daily_states, all_closed, reporting_config)

    days_running = len(sorted_states)

    report_data = {
        "report_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "db_backend": ACTIVE_DB_BACKEND,
        "strategy_name": full_config.get("strategy", {}).get("name", "mean_reversion_v1"),
        "circuit_breaker_active": cb_active,
        "portfolio_value": portfolio_value,
        "cash": cash,
        "equity": equity,
        "starting_capital": starting_capital,
        "total_pnl": metrics["total_pnl"],
        "total_return_pct": metrics["total_return_pct"],
        "days_running": days_running,
        "regime": regime,
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "total_trades": metrics["total_trades"],
        "avg_hold_days": metrics["avg_hold_days"],
        "pdt_count": pdt_count,
        "spy_price": 0.0,
        "spy_sma": 0.0,
        "vix_level": 0.0,
        "open_positions": [],
        "closed_trades": [],
    }

    for trade in open_trades:
        report_data["open_positions"].append({
            "symbol": trade.symbol,
            "entry_price": float(trade.entry_price),
            "current_price": float(trade.highest_price_since_entry or trade.entry_price),
            "unrealized_pnl": round(
                float(trade.highest_price_since_entry or trade.entry_price)
                - float(trade.entry_price),
                2,
            ) * trade.qty,
            "days_held": (datetime.now(timezone.utc) - trade.entry_time).days if trade.entry_time else 0,
            "stop_price": float(trade.trailing_stop_price or trade.stop_price or 0),
        })

    for trade in closed_trades:
        hold = (trade.exit_time - trade.entry_time).days if trade.exit_time and trade.entry_time else 0
        report_data["closed_trades"].append({
            "symbol": trade.symbol,
            "entry_price": float(trade.entry_price),
            "exit_price": float(trade.exit_price or 0),
            "realized_pnl": float(trade.realized_pnl or 0),
            "exit_reason": trade.exit_reason or "N/A",
            "hold_days": hold,
            "was_day_trade": trade.was_day_trade,
        })

    console = Console()

    console.print()
    if cb_active:
        console.print(Panel("[bold red]CIRCUIT BREAKER ACTIVE - TRADING HALTED[/]", style="red"))

    console.print(Panel(
        f"[bold cyan]AlgoTrader Dashboard[/] | DB: {ACTIVE_DB_BACKEND} | "
        f"Strategy: {report_data['strategy_name']} | "
        f"Generated: {report_data['generated_at']}",
        style="cyan",
    ))

    portfolio_table = Table(title="Portfolio Summary", show_header=False, padding=(0, 2))
    portfolio_table.add_column("Metric", style="bold")
    portfolio_table.add_column("Value")
    pnl_style = "green" if metrics["total_pnl"] >= 0 else "red"
    portfolio_table.add_row("Portfolio Value", f"${portfolio_value:.2f}")
    portfolio_table.add_row("Cash", f"${cash:.2f}")
    portfolio_table.add_row("Starting Capital", f"${starting_capital:.2f}")
    portfolio_table.add_row(
        "Total P&L",
        f"[{pnl_style}]${metrics['total_pnl']:.2f} ({metrics['total_return_pct'] * 100:.2f}%)[/]",
    )
    portfolio_table.add_row("Days Running", str(days_running))
    portfolio_table.add_row("Open Positions", str(len(open_trades)))
    console.print(portfolio_table)

    perf_table = Table(title="Performance Metrics", show_header=False, padding=(0, 2))
    perf_table.add_column("Metric", style="bold")
    perf_table.add_column("Value")
    perf_table.add_row("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    perf_table.add_row("Sortino Ratio", f"{metrics['sortino']:.2f}")
    perf_table.add_row("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    perf_table.add_row("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
    perf_table.add_row("Total Trades", str(metrics["total_trades"]))
    perf_table.add_row("Avg Hold Days", f"{metrics['avg_hold_days']:.1f}")
    perf_table.add_row("PDT This Week", f"{pdt_count}/3")
    console.print(perf_table)

    regime_style = {"BULL": "green", "NEUTRAL": "yellow", "BEAR": "red"}.get(regime, "white")
    console.print(f"\nRegime: [{regime_style}]{regime}[/]")

    if open_trades:
        pos_table = Table(title="Open Positions")
        pos_table.add_column("Symbol", style="bold")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Current", justify="right")
        pos_table.add_column("Unrealized P&L", justify="right")
        pos_table.add_column("Days Held", justify="right")
        pos_table.add_column("Stop", justify="right")
        for p in report_data["open_positions"]:
            upnl_style = "green" if p["unrealized_pnl"] >= 0 else "red"
            pos_table.add_row(
                p["symbol"],
                f"${p['entry_price']:.2f}",
                f"${p['current_price']:.2f}",
                f"[{upnl_style}]${p['unrealized_pnl']:.2f}[/]",
                str(p["days_held"]),
                f"${p['stop_price']:.2f}",
            )
        console.print(pos_table)

    if closed_trades:
        trade_table = Table(title="Last 20 Closed Trades")
        trade_table.add_column("Symbol", style="bold")
        trade_table.add_column("Entry", justify="right")
        trade_table.add_column("Exit", justify="right")
        trade_table.add_column("P&L", justify="right")
        trade_table.add_column("Reason")
        trade_table.add_column("Days", justify="right")
        trade_table.add_column("DT?")
        for t in report_data["closed_trades"]:
            pnl_s = "green" if t["realized_pnl"] >= 0 else "red"
            trade_table.add_row(
                t["symbol"],
                f"${t['entry_price']:.2f}",
                f"${t['exit_price']:.2f}",
                f"[{pnl_s}]${t['realized_pnl']:.2f}[/]",
                t["exit_reason"],
                str(t["hold_days"]),
                "Yes" if t["was_day_trade"] else "No",
            )
        console.print(trade_table)
    else:
        console.print("\n[dim]No closed trades yet.[/]")

    console.print()
    return report_data


def generate_html_report(report_data: dict) -> str:
    """Generate an HTML report from report data using Jinja2.

    Args:
        report_data: Dictionary of report data from generate_terminal_report().

    Returns:
        str: File path of the generated HTML report.

    Example:
        >>> path = generate_html_report(data)
        >>> path
        'reports/output/report_2024-01-15.html'
    """
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html.j2")

    html_content = template.render(**report_data)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"report_{report_data['report_date']}.html"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("HTML report saved to %s", output_path)
    return output_path


def main() -> None:
    """Main entry point for standalone dashboard execution.

    Initializes the database, generates terminal report,
    and saves HTML report.

    Example:
        >>> main()  # Displays terminal report and saves HTML
    """
    init_db()
    report_data = generate_terminal_report()
    html_path = generate_html_report(report_data)
    print(f"\nHTML report saved: {html_path}")


if __name__ == "__main__":
    main()
