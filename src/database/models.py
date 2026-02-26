"""SQLAlchemy ORM models for the algorithmic trading system.

All models are compatible with both PostgreSQL (Supabase) and SQLite.
Monetary values use Numeric(18, 6). All timestamps are stored in UTC.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models.

    Provides the declarative base that SQLAlchemy uses to map
    Python classes to database tables.
    """
    pass


def _utcnow() -> datetime:
    """Return current UTC datetime.

    Returns:
        datetime: Timezone-aware UTC datetime.

    Example:
        >>> ts = _utcnow()
        >>> ts.tzinfo  # UTC
    """
    return datetime.now(timezone.utc)


class Universe(Base):
    """Daily universe of tradeable symbols after filter pipeline.

    Each row represents one symbol evaluated on a given date,
    recording whether it passed all filters and the reason for
    any exclusion.

    Attributes:
        id: Auto-incrementing primary key.
        date: Date the screening was performed (UTC).
        symbol: Ticker symbol evaluated.
        passed_filters: True if the symbol passed all filter stages.
        exclusion_reason: Human-readable reason for exclusion, or None if passed.
        created_at: Timestamp when the record was created (UTC).
    """
    __tablename__ = "universe"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    passed_filters: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    exclusion_reason: Mapped[str] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow
    )

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Formatted string with date, symbol, and filter status.

        Example:
            >>> u = Universe(date=datetime(2024,1,1), symbol="AAPL", passed_filters=True)
            >>> repr(u)
            "<Universe(date=2024-01-01, symbol=AAPL, passed=True)>"
        """
        date_str = self.date.strftime("%Y-%m-%d") if self.date else "None"
        return f"<Universe(date={date_str}, symbol={self.symbol}, passed={self.passed_filters})>"


class Trade(Base):
    """Individual trade record tracking full lifecycle from entry to exit.

    Stores entry/exit prices, stop levels, P&L, and metadata for
    every trade executed by the system.

    Attributes:
        id: Auto-incrementing primary key.
        symbol: Ticker symbol traded.
        strategy_name: Name of the strategy that generated the signal.
        side: Trade direction ('buy' or 'sell').
        qty: Number of shares traded.
        entry_price: Price at which the position was entered.
        exit_price: Price at which the position was exited, or None if open.
        entry_time: UTC timestamp of entry.
        exit_time: UTC timestamp of exit, or None if open.
        stop_price: Current hard stop price.
        take_profit_price: Target profit price.
        highest_price_since_entry: Highest price observed since entry (for trailing stop).
        trailing_stop_price: Current trailing stop price.
        realized_pnl: Realized profit/loss after exit.
        commissions: Estimated commission cost.
        slippage_estimate: Estimated slippage cost.
        exit_reason: Why the position was closed.
        signal_metadata: JSON blob of strategy signal details (z-score, RSI, etc).
        status: Current trade status (OPEN, CLOSED, STOPPED).
        was_day_trade: Whether this trade was a same-day round trip.
        alpaca_order_id: Alpaca order ID for reconciliation.
    """
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    exit_price: Mapped[float] = mapped_column(Numeric(18, 6), nullable=True)
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    exit_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    stop_price: Mapped[float] = mapped_column(Numeric(18, 6), nullable=True)
    take_profit_price: Mapped[float] = mapped_column(Numeric(18, 6), nullable=True)
    highest_price_since_entry: Mapped[float] = mapped_column(Numeric(18, 6), nullable=True)
    trailing_stop_price: Mapped[float] = mapped_column(Numeric(18, 6), nullable=True)
    realized_pnl: Mapped[float] = mapped_column(Numeric(18, 6), nullable=True, default=0)
    commissions: Mapped[float] = mapped_column(Numeric(18, 6), nullable=True, default=0)
    slippage_estimate: Mapped[float] = mapped_column(Numeric(18, 6), nullable=True, default=0)
    exit_reason: Mapped[str] = mapped_column(String(20), nullable=True)
    signal_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)
    prediction_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="OPEN", index=True)
    was_day_trade: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alpaca_order_id: Mapped[str] = mapped_column(String(50), nullable=True)
    layer: Mapped[str] = mapped_column(String(30), nullable=True)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Formatted string with symbol, side, qty, status, and P&L.

        Example:
            >>> t = Trade(symbol="AAPL", side="buy", qty=5, status="OPEN", entry_price=150.0)
            >>> repr(t)
            "<Trade(AAPL buy 5 shares, status=OPEN, pnl=0)>"
        """
        pnl = round(float(self.realized_pnl or 0), 2)
        return f"<Trade({self.symbol} {self.side} {self.qty} shares, status={self.status}, pnl={pnl})>"


class DailyPortfolioState(Base):
    """End-of-day portfolio snapshot for performance tracking.

    Captured at market close each trading day. Used by circuit breaker,
    performance reporting, and drawdown calculations.

    Attributes:
        id: Auto-incrementing primary key.
        date: Trading date (unique constraint, UTC).
        portfolio_value: Total portfolio value including positions.
        cash: Available cash balance.
        equity: Account equity.
        num_open_positions: Number of open positions at close.
        daily_pnl: Day's profit/loss in dollars.
        daily_return_pct: Day's return as a percentage.
        peak_value: Highest portfolio value seen to date.
        drawdown_pct: Current drawdown from peak as a percentage.
        regime: Market regime at close (BULL, NEUTRAL, BEAR).
        circuit_breaker_active: Whether the circuit breaker is currently engaged.
        day_trade_count: Number of day trades used in the rolling window.
    """
    __tablename__ = "daily_portfolio_state"
    __table_args__ = (UniqueConstraint("date", name="uq_daily_state_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    portfolio_value: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    equity: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    num_open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    daily_pnl: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False, default=0)
    daily_return_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_value: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    regime: Mapped[str] = mapped_column(String(10), nullable=False, default="UNKNOWN")
    circuit_breaker_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    day_trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    signals_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    orders_submitted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    discord_notifications_sent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    discord_notifications_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Formatted string with date, portfolio value, P&L, and regime.

        Example:
            >>> d = DailyPortfolioState(date=datetime(2024,1,1), portfolio_value=505.0, daily_pnl=5.0, regime="BULL")
            >>> repr(d)
            "<DailyState(2024-01-01, value=$505.00, pnl=$5.00, regime=BULL)>"
        """
        date_str = self.date.strftime("%Y-%m-%d") if self.date else "None"
        return (
            f"<DailyState({date_str}, value=${float(self.portfolio_value):.2f}, "
            f"pnl=${float(self.daily_pnl):.2f}, regime={self.regime})>"
        )


class SystemLog(Base):
    """Persistent system log entries for auditing and debugging.

    Supplements file-based logging with structured database records
    that can be queried for monitoring and incident investigation.

    Attributes:
        id: Auto-incrementing primary key.
        timestamp: UTC timestamp of the log entry.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        module: Source module name.
        message: Log message text.
        extra_data: Optional JSON blob with additional context.
    """
    __tablename__ = "system_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow, index=True
    )
    level: Mapped[str] = mapped_column(String(10), nullable=False)
    module: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    extra_data: Mapped[dict] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Formatted string with timestamp, level, and truncated message.

        Example:
            >>> s = SystemLog(level="INFO", module="broker", message="Connected")
            >>> repr(s)
            "<SystemLog(INFO broker: Connected)>"
        """
        msg_short = (self.message[:50] + "...") if len(self.message) > 50 else self.message
        return f"<SystemLog({self.level} {self.module}: {msg_short})>"


class CircuitBreakerState(Base):
    """Circuit breaker activation and reset history.

    Tracks every circuit breaker trigger and reset event for
    audit trail and operational monitoring.

    Attributes:
        id: Auto-incrementing primary key.
        triggered_at: UTC timestamp when the circuit breaker was triggered.
        trigger_reason: Human-readable description of the trigger condition.
        reset_at: UTC timestamp when the circuit breaker was reset, or None if active.
        reset_by: How the reset was performed ('manual' or 'auto').
        is_active: Whether this circuit breaker event is currently active.
    """
    __tablename__ = "circuit_breaker_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow
    )
    trigger_reason: Mapped[str] = mapped_column(Text, nullable=False)
    reset_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    reset_by: Mapped[str] = mapped_column(String(10), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Formatted string with active status and trigger reason.

        Example:
            >>> cb = CircuitBreakerState(is_active=True, trigger_reason="Daily loss limit")
            >>> repr(cb)
            "<CircuitBreaker(active=True, reason=Daily loss limit)>"
        """
        reason_short = (
            (self.trigger_reason[:40] + "...")
            if len(self.trigger_reason) > 40
            else self.trigger_reason
        )
        return f"<CircuitBreaker(active={self.is_active}, reason={reason_short})>"


class NotificationLog(Base):
    """Log of Discord notifications sent, failed, or skipped.

    Tracks every notification attempt for audit trail,
    debugging, and rate-limiting analytics.

    Attributes:
        id: Auto-incrementing primary key.
        timestamp: UTC timestamp of the notification event.
        channel: Target channel (trades, signals, alerts, daily).
        notification_type: Type of notification (trade_entry, trade_exit, signal, etc).
        status: Delivery status (sent, failed, skipped).
        symbol: Associated ticker symbol, if applicable.
        payload_summary: Truncated summary of the notification payload.
        retry_count: Number of retries attempted before final status.
    """
    __tablename__ = "notification_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_utcnow, index=True
    )
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    notification_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    symbol: Mapped[str] = mapped_column(String(10), nullable=True)
    payload_summary: Mapped[str] = mapped_column(String(500), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Formatted string with channel, type, and status.

        Example:
            >>> n = NotificationLog(channel="trades", notification_type="trade_entry", status="sent")
            >>> repr(n)
            "<NotificationLog(trades/trade_entry: sent)>"
        """
        return f"<NotificationLog({self.channel}/{self.notification_type}: {self.status})>"
