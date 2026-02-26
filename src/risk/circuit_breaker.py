"""Portfolio-level circuit breaker for 24/7 month-long run safety.

Monitors four trigger conditions every intraday cycle and performs
emergency liquidation when any condition is met. Can only be reset
via manual CLI command.
"""

import os
from datetime import datetime, timezone
from typing import Optional

import yaml

from src.database import repository
from src.utils.logger import get_logger

logger = get_logger("risk.circuit_breaker")

TRADING_HALTED: bool = False


def _load_config() -> dict:
    """Load risk configuration from config.yml.

    Returns:
        dict: Risk configuration parameters.

    Example:
        >>> config = _load_config()
        >>> config["daily_loss_limit"]
        0.03
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("risk", {})


class CircuitBreaker:
    """Portfolio-level emergency shutdown system.

    Trigger conditions (checked in order every intraday cycle):
    1. Daily loss: portfolio down > 3% from day's open (~$15 on $500)
    2. Weekly loss: portfolio down > 8% from 7 days ago (~$40 on $500)
    3. Max drawdown: portfolio down > 15% from peak (~$75 below peak)
    4. Consecutive losses: 3+ days of negative P&L in a row

    On trigger: cancel all orders, liquidate all positions (bypassing PDT),
    set circuit breaker flag, halt trading.

    Reset: Only via `python main.py --reset-circuit-breaker`.

    Attributes:
        config: Risk configuration parameters.
        daily_loss_limit: Maximum allowed daily loss (decimal). Default 0.03.
        weekly_loss_limit: Maximum allowed weekly loss (decimal). Default 0.08.
        max_drawdown_limit: Maximum allowed drawdown from peak (decimal). Default 0.15.
        consecutive_loss_days_limit: Max consecutive loss days. Default 3.
    """

    def __init__(self, config=None, account_monitor=None):
        """Initialize the circuit breaker with configuration.

        Args:
            config: Optional full config dict. If None, loads from file.
            account_monitor: Optional AccountMonitor for live equity data.
        """
        self.account_monitor = account_monitor
        risk_config = config.get("risk", {}) if config else _load_config()
        self.config = risk_config
        self.daily_loss_limit = self.config.get("daily_loss_limit", 0.015)
        self.weekly_loss_limit = self.config.get("weekly_loss_limit", 0.05)
        self.max_drawdown_limit = self.config.get("max_drawdown_limit", 0.05)
        self.consecutive_loss_days_limit = self.config.get("consecutive_loss_days", 4)

    def is_active(self) -> bool:
        """Check if the circuit breaker is currently engaged.

        Returns:
            bool: True if circuit breaker is active and trading is halted.

        Example:
            >>> cb = CircuitBreaker()
            >>> cb.is_active()
            False
        """
        return repository.is_circuit_breaker_active()

    def check_all_conditions(
        self,
        portfolio_now: float,
        portfolio_at_open: float,
    ) -> Optional[str]:
        """Check all four circuit breaker conditions in order.

        Args:
            portfolio_now: Current portfolio value.
            portfolio_at_open: Portfolio value at today's market open.

        Returns:
            Optional[str]: Trigger reason string if any condition is met,
                None if all conditions are safe.

        Example:
            >>> cb = CircuitBreaker()
            >>> reason = cb.check_all_conditions(484.50, 500.0)
            >>> reason
            'Daily loss limit breached: -3.10% ($484.50 from $500.00)'
        """
        reason = self._check_daily_loss(portfolio_now, portfolio_at_open)
        if reason:
            return reason

        reason = self._check_weekly_loss(portfolio_now)
        if reason:
            return reason

        reason = self._check_max_drawdown(portfolio_now)
        if reason:
            return reason

        reason = self._check_consecutive_losses()
        if reason:
            return reason

        return None

    def _check_daily_loss(
        self, portfolio_now: float, portfolio_at_open: float
    ) -> Optional[str]:
        """Check if daily loss exceeds the limit.

        Trigger: (portfolio_now - portfolio_at_open) / portfolio_at_open < -DAILY_LOSS_LIMIT

        Args:
            portfolio_now: Current portfolio value.
            portfolio_at_open: Portfolio value at today's open.

        Returns:
            Optional[str]: Trigger reason or None.

        Example:
            >>> cb = CircuitBreaker()
            >>> cb._check_daily_loss(484.50, 500.0)
            'Daily loss limit breached: -3.10% ($484.50 from $500.00)'
        """
        if portfolio_at_open <= 0:
            return None

        daily_change = (portfolio_now - portfolio_at_open) / portfolio_at_open

        if daily_change < -self.daily_loss_limit:
            reason = (
                f"Daily loss limit breached: {daily_change * 100:.2f}% "
                f"(${portfolio_now:.2f} from ${portfolio_at_open:.2f})"
            )
            logger.critical(reason)
            return reason
        return None

    def _check_weekly_loss(self, portfolio_now: float) -> Optional[str]:
        """Check if weekly loss exceeds the limit.

        Trigger: portfolio_now < portfolio_7d_ago * (1 - WEEKLY_LOSS_LIMIT)

        Args:
            portfolio_now: Current portfolio value.

        Returns:
            Optional[str]: Trigger reason or None.

        Example:
            >>> cb._check_weekly_loss(460.0)  # If 7d ago was $500
            'Weekly loss limit breached: ...'
        """
        portfolio_7d = repository.get_portfolio_value_n_days_ago(5)
        threshold = portfolio_7d * (1.0 - self.weekly_loss_limit)

        if portfolio_now < threshold:
            loss_pct = (portfolio_now - portfolio_7d) / portfolio_7d * 100
            reason = (
                f"Weekly loss limit breached: {loss_pct:.2f}% "
                f"(${portfolio_now:.2f} vs ${portfolio_7d:.2f} 7d ago, "
                f"limit={self.weekly_loss_limit * 100:.0f}%)"
            )
            logger.critical(reason)
            return reason
        return None

    def _check_max_drawdown(self, portfolio_now: float) -> Optional[str]:
        """Check if drawdown from peak exceeds the limit.

        Trigger: portfolio_now < peak_value * (1 - MAX_DRAWDOWN_LIMIT)

        Args:
            portfolio_now: Current portfolio value.

        Returns:
            Optional[str]: Trigger reason or None.

        Example:
            >>> cb._check_max_drawdown(425.0)  # If peak was $500
            'Max drawdown limit breached: ...'
        """
        peak = repository.get_peak_portfolio_value()
        threshold = peak * (1.0 - self.max_drawdown_limit)

        if portfolio_now < threshold:
            drawdown_pct = (portfolio_now - peak) / peak * 100
            reason = (
                f"Max drawdown limit breached: {drawdown_pct:.2f}% "
                f"(${portfolio_now:.2f} vs peak ${peak:.2f}, "
                f"limit={self.max_drawdown_limit * 100:.0f}%)"
            )
            logger.critical(reason)
            return reason
        return None

    def _check_consecutive_losses(self) -> Optional[str]:
        """Check if consecutive loss days exceed the limit.

        Trigger: CONSECUTIVE_LOSS_DAYS days of negative P&L in a row.

        Returns:
            Optional[str]: Trigger reason or None.

        Example:
            >>> cb._check_consecutive_losses()  # If 3+ loss days
            'Consecutive loss days limit: 3 days...'
        """
        consecutive = repository.get_consecutive_loss_days()

        if consecutive >= self.consecutive_loss_days_limit:
            reason = (
                f"Consecutive loss days limit: {consecutive} days of negative P&L "
                f"(limit={self.consecutive_loss_days_limit})"
            )
            logger.critical(reason)
            return reason
        return None

    def trigger(self, reason: str) -> None:
        """Activate the circuit breaker and perform emergency liquidation.

        Steps:
        1. Log CRITICAL with reason and dollar amounts
        2. Cancel all open orders
        3. Liquidate all positions (bypasses PDT guard)
        4. Set circuit breaker flag in database
        5. Set TRADING_HALTED = True

        Args:
            reason: Human-readable trigger reason.

        Example:
            >>> cb.trigger("Daily loss limit breached: -3.1%")
        """
        global TRADING_HALTED

        logger.critical("CIRCUIT BREAKER TRIGGERED: %s", reason)

        try:
            from src.broker.order_manager import OrderManager
            om = OrderManager(dry_run=False)
            om.emergency_liquidate()
        except Exception as e:
            logger.critical("Emergency liquidation failed: %s", str(e))

        repository.set_circuit_breaker(active=True, reason=reason)
        repository.save_system_log(
            level="CRITICAL",
            module="circuit_breaker",
            message=f"Circuit breaker triggered: {reason}",
            extra_data={"reason": reason},
        )

        TRADING_HALTED = True
        logger.critical(
            "Trading HALTED. Reset with: python main.py --reset-circuit-breaker"
        )

    def check(self, portfolio_now: float = 0.0, portfolio_at_open: float = 0.0) -> bool:
        """Check all conditions and trigger if needed. Called on every bar.

        Uses AccountMonitor live data if available, otherwise falls back
        to the provided arguments or database.

        Args:
            portfolio_now: Current portfolio value (fallback).
            portfolio_at_open: Portfolio value at today's market open (fallback).

        Returns:
            bool: True if circuit breaker was just triggered, False otherwise.
        """
        if TRADING_HALTED or self.is_active():
            return True

        # Prefer live account monitor data
        if hasattr(self, 'account_monitor') and self.account_monitor and self.account_monitor.last_updated:
            portfolio_now = self.account_monitor.equity
            portfolio_at_open = self.account_monitor.equity_at_open

        reason = self.check_all_conditions(portfolio_now, portfolio_at_open)
        if reason:
            self.trigger(reason)
            return True
        return False

    def is_triggered(self) -> bool:
        """Return True if trading is currently halted.

        Returns:
            bool: True if TRADING_HALTED or DB circuit breaker active.

        Example:
            >>> cb.is_triggered()
            False
        """
        return TRADING_HALTED or self.is_active()

    @staticmethod
    def reset(reason: str = "Manual reset") -> None:
        """Reset the circuit breaker and re-enable trading.

        Can only be called via CLI: python main.py --reset-circuit-breaker

        Args:
            reason: Reason for the reset. Default "Manual reset".

        Example:
            >>> CircuitBreaker.reset("Manual reset by operator")
        """
        global TRADING_HALTED

        repository.set_circuit_breaker(
            active=False, reason=reason, reset_by="manual"
        )
        TRADING_HALTED = False

        logger.info(
            "Circuit breaker RESET: %s at %s",
            reason,
            datetime.now(timezone.utc).isoformat(),
        )
        repository.save_system_log(
            level="INFO",
            module="circuit_breaker",
            message=f"Circuit breaker reset: {reason}",
        )
