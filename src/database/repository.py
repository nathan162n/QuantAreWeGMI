"""Database repository providing all read/write operations.

Centralizes all database access so no other module imports SQLAlchemy
directly. Every function uses context-managed sessions that always
close on exit, even on exception.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import desc, func

from src.database.engine import get_session
from src.database.models import (
    CircuitBreakerState,
    DailyPortfolioState,
    NotificationLog,
    SystemLog,
    Trade,
    Universe,
)
from src.utils.logger import get_logger

logger = get_logger("database.repository")


def save_trade(trade_data: dict) -> Trade:
    """Persist a new trade record to the database.

    Args:
        trade_data: Dictionary of trade attributes matching Trade model columns.
            Required keys: symbol, strategy_name, side, qty, entry_price, entry_time.

    Returns:
        Trade: The persisted trade ORM object with its assigned ID.

    Example:
        >>> trade = save_trade({"symbol": "AAPL", "strategy_name": "mean_reversion_v1",
        ...     "side": "buy", "qty": 3, "entry_price": 150.0,
        ...     "entry_time": datetime.now(timezone.utc)})
        >>> trade.id  # auto-assigned database ID
    """
    with get_session() as session:
        trade = Trade(**trade_data)
        session.add(trade)
        session.flush()
        trade_id = trade.id
        logger.info("Saved trade: %s", trade)
    return _get_trade_by_id(trade_id)


def _get_trade_by_id(trade_id: int) -> Trade:
    """Fetch a trade by its primary key.

    Args:
        trade_id: The trade's database ID.

    Returns:
        Trade: The trade ORM object.

    Example:
        >>> trade = _get_trade_by_id(1)
    """
    with get_session() as session:
        trade = session.query(Trade).filter_by(id=trade_id).first()
        if trade:
            session.expunge(trade)
        return trade


def update_trade(trade_id: int, update_data: dict) -> Trade:
    """Update an existing trade record.

    Args:
        trade_id: Database ID of the trade to update.
        update_data: Dictionary of column names to new values.

    Returns:
        Trade: The updated trade ORM object.

    Example:
        >>> trade = update_trade(1, {"exit_price": 155.0, "exit_time": datetime.now(timezone.utc),
        ...     "realized_pnl": 15.0, "status": "CLOSED", "exit_reason": "SIGNAL"})
    """
    # Bug 3 guard: block corrupt $0.00 exit prices from being persisted.
    if "exit_price" in update_data and (
        update_data["exit_price"] is None or update_data["exit_price"] <= 0
    ):
        logger.error(
            "Attempted to write exit_price=%.4f for trade %d — blocked",
            update_data.get("exit_price", 0) or 0,
            trade_id,
        )
        update_data = dict(update_data)
        update_data.pop("exit_price", None)

    with get_session() as session:
        trade = session.query(Trade).filter_by(id=trade_id).first()
        if trade is None:
            logger.error("Trade ID %d not found for update", trade_id)
            return None
        for key, value in update_data.items():
            setattr(trade, key, value)
        session.flush()
        logger.info("Updated trade %d: %s", trade_id, list(update_data.keys()))
    return _get_trade_by_id(trade_id)


def get_open_trades() -> List[Trade]:
    """Fetch all trades with status OPEN.

    Returns:
        List[Trade]: List of open trade ORM objects.

    Example:
        >>> open_trades = get_open_trades()
        >>> len(open_trades)
        3
    """
    with get_session() as session:
        trades = session.query(Trade).filter_by(status="OPEN").all()
        for t in trades:
            session.expunge(t)
        return trades


def get_trades_by_date_range(start: datetime, end: datetime) -> List[Trade]:
    """Fetch all trades entered within a date range.

    Args:
        start: Start of the date range (inclusive, UTC).
        end: End of the date range (inclusive, UTC).

    Returns:
        List[Trade]: List of trade ORM objects within the range.

    Example:
        >>> from datetime import datetime, timezone
        >>> trades = get_trades_by_date_range(
        ...     datetime(2024, 1, 1, tzinfo=timezone.utc),
        ...     datetime(2024, 1, 31, tzinfo=timezone.utc))
    """
    with get_session() as session:
        trades = (
            session.query(Trade)
            .filter(Trade.entry_time >= start, Trade.entry_time <= end)
            .order_by(Trade.entry_time)
            .all()
        )
        for t in trades:
            session.expunge(t)
        return trades


def get_closed_trades(limit: int = 20) -> List[Trade]:
    """Fetch the most recent closed trades.

    Args:
        limit: Maximum number of trades to return. Default 20.

    Returns:
        List[Trade]: List of closed trade ORM objects, most recent first.

    Example:
        >>> closed = get_closed_trades(limit=10)
    """
    with get_session() as session:
        trades = (
            session.query(Trade)
            .filter(Trade.status.in_(["CLOSED", "STOPPED"]))
            .order_by(desc(Trade.exit_time))
            .limit(limit)
            .all()
        )
        for t in trades:
            session.expunge(t)
        return trades


def has_open_position(symbol: str) -> bool:
    """Check if there is an open trade for a given symbol.

    Args:
        symbol: Ticker symbol to check.

    Returns:
        bool: True if an open trade exists for the symbol.

    Example:
        >>> has_open_position("AAPL")
        True
    """
    with get_session() as session:
        count = (
            session.query(func.count(Trade.id))
            .filter_by(symbol=symbol, status="OPEN")
            .scalar()
        )
        return count > 0


def save_daily_state(state_data: dict) -> DailyPortfolioState:
    """Persist an end-of-day portfolio snapshot.

    If a record already exists for the given date, it is updated
    instead of creating a duplicate.

    Args:
        state_data: Dictionary matching DailyPortfolioState columns.
            Required keys: date, portfolio_value, cash, equity, peak_value.

    Returns:
        DailyPortfolioState: The persisted state ORM object.

    Example:
        >>> state = save_daily_state({"date": datetime(2024,1,1, tzinfo=timezone.utc),
        ...     "portfolio_value": 505.0, "cash": 425.0, "equity": 505.0,
        ...     "peak_value": 505.0, "daily_pnl": 5.0, "daily_return_pct": 0.01})
    """
    with get_session() as session:
        existing = (
            session.query(DailyPortfolioState)
            .filter_by(date=state_data["date"])
            .first()
        )
        if existing:
            for key, value in state_data.items():
                setattr(existing, key, value)
            session.flush()
            state_id = existing.id
            logger.info("Updated daily state for %s", state_data["date"])
        else:
            state = DailyPortfolioState(**state_data)
            session.add(state)
            session.flush()
            state_id = state.id
            logger.info("Saved daily state for %s", state_data["date"])

    with get_session() as session:
        result = session.query(DailyPortfolioState).filter_by(id=state_id).first()
        if result:
            session.expunge(result)
        return result


def get_daily_states(n_days: int = 30) -> List[DailyPortfolioState]:
    """Fetch the most recent daily portfolio states.

    Args:
        n_days: Number of recent days to fetch. Default 30.

    Returns:
        List[DailyPortfolioState]: List of states ordered by date descending.

    Example:
        >>> states = get_daily_states(n_days=7)
        >>> states[0].portfolio_value  # most recent
    """
    with get_session() as session:
        states = (
            session.query(DailyPortfolioState)
            .order_by(desc(DailyPortfolioState.date))
            .limit(n_days)
            .all()
        )
        for s in states:
            session.expunge(s)
        return states


def get_peak_portfolio_value() -> float:
    """Get the highest portfolio value ever recorded.

    Returns:
        float: Peak portfolio value, or 500.0 if no records exist.

    Example:
        >>> peak = get_peak_portfolio_value()
        >>> peak
        525.50
    """
    with get_session() as session:
        result = session.query(
            func.max(DailyPortfolioState.peak_value)
        ).scalar()
        return float(result) if result is not None else 500.0


def is_circuit_breaker_active() -> bool:
    """Check if any circuit breaker event is currently active.

    Returns:
        bool: True if the circuit breaker is engaged.

    Example:
        >>> is_circuit_breaker_active()
        False
    """
    with get_session() as session:
        active = (
            session.query(CircuitBreakerState)
            .filter_by(is_active=True)
            .first()
        )
        return active is not None


def set_circuit_breaker(
    active: bool, reason: str, reset_by: str = "manual"
) -> None:
    """Activate or deactivate the circuit breaker.

    When activating, creates a new CircuitBreakerState record.
    When deactivating, updates all active records with reset timestamp.

    Args:
        active: True to activate, False to deactivate.
        reason: Human-readable reason for the state change.
        reset_by: How the reset was performed ('manual' or 'auto').

    Example:
        >>> set_circuit_breaker(True, "Daily loss limit breached: -3.2%")
        >>> set_circuit_breaker(False, "Manual reset by operator", reset_by="manual")
    """
    with get_session() as session:
        if active:
            cb = CircuitBreakerState(
                trigger_reason=reason,
                is_active=True,
            )
            session.add(cb)
            logger.critical("Circuit breaker ACTIVATED: %s", reason)
        else:
            active_records = (
                session.query(CircuitBreakerState)
                .filter_by(is_active=True)
                .all()
            )
            now = datetime.now(timezone.utc)
            for record in active_records:
                record.is_active = False
                record.reset_at = now
                record.reset_by = reset_by
            logger.info(
                "Circuit breaker RESET by %s: %s (updated %d records)",
                reset_by,
                reason,
                len(active_records),
            )


def save_universe(
    date: datetime, symbols: List[str], rejected: Dict[str, str]
) -> None:
    """Save the daily universe screening results.

    Args:
        date: Date of the screening (UTC).
        symbols: List of symbols that passed all filters.
        rejected: Dictionary mapping rejected symbols to their exclusion reason.

    Example:
        >>> save_universe(datetime.now(timezone.utc),
        ...     ["AAPL", "MSFT", "GOOGL"],
        ...     {"PENNY": "Price below $15", "ILLIQ": "ADV below $50M"})
    """
    with get_session() as session:
        for symbol in symbols:
            entry = Universe(
                date=date,
                symbol=symbol,
                passed_filters=True,
                exclusion_reason=None,
            )
            session.add(entry)
        for symbol, reason in rejected.items():
            entry = Universe(
                date=date,
                symbol=symbol,
                passed_filters=False,
                exclusion_reason=reason,
            )
            session.add(entry)
        logger.info(
            "Saved universe: %d passed, %d rejected for %s",
            len(symbols),
            len(rejected),
            date.strftime("%Y-%m-%d"),
        )


def get_latest_universe() -> List[str]:
    """Get the most recently screened universe of tradeable symbols.

    Returns:
        List[str]: List of ticker symbols from the latest screening that passed filters.

    Example:
        >>> symbols = get_latest_universe()
        >>> "AAPL" in symbols
        True
    """
    with get_session() as session:
        latest_date = session.query(func.max(Universe.date)).scalar()
        if latest_date is None:
            logger.warning("No universe data found in database")
            return []
        symbols = (
            session.query(Universe.symbol)
            .filter_by(date=latest_date, passed_filters=True)
            .all()
        )
        return [s[0] for s in symbols]


def get_consecutive_loss_days() -> int:
    """Count consecutive days of negative P&L ending today.

    Returns:
        int: Number of consecutive loss days. 0 if last day was profitable.

    Example:
        >>> get_consecutive_loss_days()
        2
    """
    with get_session() as session:
        states = (
            session.query(DailyPortfolioState)
            .order_by(desc(DailyPortfolioState.date))
            .limit(30)
            .all()
        )
        count = 0
        for state in states:
            if float(state.daily_pnl) < 0:
                count += 1
            else:
                break
        return count


def get_portfolio_value_n_days_ago(n: int) -> float:
    """Get the portfolio value from n trading days ago.

    Args:
        n: Number of trading days to look back.

    Returns:
        float: Portfolio value from n days ago, or 500.0 if not enough history.

    Example:
        >>> get_portfolio_value_n_days_ago(7)
        498.50
    """
    with get_session() as session:
        states = (
            session.query(DailyPortfolioState)
            .order_by(desc(DailyPortfolioState.date))
            .limit(n + 1)
            .all()
        )
        if len(states) > n:
            return float(states[n].portfolio_value)
        if states:
            return float(states[-1].portfolio_value)
        return 500.0


def save_system_log(
    level: str, module: str, message: str, extra_data: Optional[dict] = None
) -> None:
    """Persist a structured log entry to the database.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        module: Source module name.
        message: Log message text.
        extra_data: Optional dictionary of additional context data.

    Example:
        >>> save_system_log("CRITICAL", "circuit_breaker",
        ...     "Daily loss limit breached", {"loss_pct": -0.032})
    """
    try:
        with get_session() as session:
            log_entry = SystemLog(
                level=level,
                module=module,
                message=message,
                extra_data=extra_data,
            )
            session.add(log_entry)
    except Exception as e:
        logger.error("Failed to save system log to DB: %s", str(e))


def get_today_trades() -> List[Trade]:
    """Get all trades entered today (UTC).

    Returns:
        List[Trade]: List of trades entered today.

    Example:
        >>> today_trades = get_today_trades()
    """
    with get_session() as session:
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        trades = (
            session.query(Trade)
            .filter(Trade.entry_time >= today_start)
            .all()
        )
        for t in trades:
            session.expunge(t)
        return trades


def get_day_trades_this_week() -> int:
    """Count day trades in the rolling 5-business-day window.

    Returns:
        int: Number of day trades in the current rolling window.

    Example:
        >>> get_day_trades_this_week()
        2
    """
    with get_session() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        count = (
            session.query(func.count(Trade.id))
            .filter(
                Trade.was_day_trade == True,
                Trade.exit_time >= cutoff,
            )
            .scalar()
        )
        return count or 0


def save_notification_log(
    channel: str,
    notification_type: str,
    status: str,
    symbol: Optional[str] = None,
    payload_summary: Optional[str] = None,
    retry_count: int = 0,
) -> None:
    """Persist a notification log entry to the database.

    Args:
        channel: Notification channel (trades, signals, alerts, daily).
        notification_type: Type of notification event.
        status: Delivery status (sent, failed, skipped).
        symbol: Optional ticker symbol associated with the notification.
        payload_summary: Optional truncated payload text (max 500 chars).
        retry_count: Number of retry attempts before final status.

    Example:
        >>> save_notification_log("trades", "trade_entry", "sent", "AAPL", "BUY 2 AAPL @ $150", 0)
    """
    try:
        with get_session() as session:
            entry = NotificationLog(
                channel=channel,
                notification_type=notification_type,
                status=status,
                symbol=symbol,
                payload_summary=(payload_summary or "")[:500],
                retry_count=retry_count,
            )
            session.add(entry)
    except Exception as e:
        logger.error("Failed to save notification log: %s", str(e))


def get_notification_stats_today() -> dict:
    """Get notification statistics for today.

    Returns:
        dict: Keys are channel names, values are dicts with sent/failed counts.

    Example:
        >>> stats = get_notification_stats_today()
        >>> stats["trades"]["sent"]
        5
    """
    with get_session() as session:
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        results = (
            session.query(
                NotificationLog.channel,
                NotificationLog.status,
                func.count(NotificationLog.id),
            )
            .filter(NotificationLog.timestamp >= today_start)
            .group_by(NotificationLog.channel, NotificationLog.status)
            .all()
        )
        stats: Dict[str, Dict[str, int]] = {}
        for channel, status, count in results:
            if channel not in stats:
                stats[channel] = {"sent": 0, "failed": 0, "skipped": 0}
            stats[channel][status] = count
        return stats


def get_trades_closed_today() -> List[Trade]:
    """Get all trades closed today (UTC).

    Returns:
        List[Trade]: List of trades closed today.

    Example:
        >>> closed = get_trades_closed_today()
    """
    with get_session() as session:
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        trades = (
            session.query(Trade)
            .filter(
                Trade.exit_time >= today_start,
                Trade.status.in_(["CLOSED", "STOPPED"]),
            )
            .all()
        )
        for t in trades:
            session.expunge(t)
        return trades


def get_session_win_rate() -> float:
    """Get the win rate of all trades closed today.

    Returns:
        float: Win rate as a decimal (0.0 to 1.0). Returns 0.0 if no trades.

    Example:
        >>> get_session_win_rate()
        0.667
    """
    trades = get_trades_closed_today()
    if not trades:
        return 0.0
    winners = [t for t in trades if float(t.realized_pnl or 0) > 0]
    return round(len(winners) / len(trades), 4)


def get_session_pnl_today() -> float:
    """Get the total P&L of all trades closed today.

    Returns:
        float: Total realized P&L for today's closed trades.

    Example:
        >>> get_session_pnl_today()
        3.50
    """
    trades = get_trades_closed_today()
    return round(sum(float(t.realized_pnl or 0) for t in trades), 6)
