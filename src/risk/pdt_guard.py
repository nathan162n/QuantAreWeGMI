"""Pattern Day Trader rule enforcement — smart rolling window tracking.

The PDT rule limits accounts under $25,000 to 3 day trades per rolling
5 business-day window. A day trade is opening AND closing the same equity
on the same calendar day.

Key difference from previous version:
  - Previous: blanket 3-day minimum hold (too restrictive)
  - New: track exact rolling count, block ONLY same-day exits when at limit
  - Entries are NEVER restricted — only the closing leg counts as a day trade

Rolling window definition:
    A day trade counts against the limit for 5 business days from the
    trade date. "5 business days" skips Saturday and Sunday only (not
    federal holidays, for simplicity).

Example:
    >>> guard = PDTGuard(max_day_trades=3)
    >>> guard.record_day_trade("AAPL")
    >>> guard.get_rolling_count()
    1
    >>> can_exit = guard.can_exit("MSFT", entry_time_yesterday)
    >>> can_exit
    True  # overnight exit always allowed
"""

from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("risk.pdt_guard")

_PDT_INSTANCE: Optional["PDTGuard"] = None

# Keep legacy lowercase alias in sync with the uppercase one so that code
# that imported the old singleton name still works.
_pdt_instance: Optional["PDTGuard"] = None


class PDTGuard:
    """Rolling 5-business-day day trade counter with smart blocking.

    Tracks exact day trades and blocks same-day exits when at the 3-trade
    limit. Never blocks new entries. Never blocks overnight exits.

    State:
        _day_trades: deque of (symbol, trade_date) tuples for same-day closed
            positions, bounded to the last 50 entries.

    Attributes:
        max_day_trades: Maximum day trades per rolling 5-business-day window.
            Default 3.
        simulation_mode: If True, track counts but never block trades (for
            backtesting). Default False.

    Example:
        >>> guard = PDTGuard(max_day_trades=3, simulation_mode=False)
        >>> guard.record_day_trade("AAPL")
        >>> guard.record_day_trade("MSFT")
        >>> guard.get_rolling_count()
        2
        >>> guard.can_exit("NVDA", entry_time=datetime.now(timezone.utc))
        True  # count is 2 < 3
    """

    def __init__(self, max_day_trades: int = 3, simulation_mode: bool = False):
        """Initialize the PDT guard.

        Args:
            max_day_trades: Maximum day trades allowed in rolling 5-business-day
                window. Default 3.
            simulation_mode: If True, counts are tracked but exits are never
                blocked (for backtesting). Default False.

        Example:
            >>> guard = PDTGuard(max_day_trades=3)
            >>> guard = PDTGuard(max_day_trades=3, simulation_mode=True)
        """
        self.max_day_trades = max_day_trades
        self.simulation_mode = simulation_mode
        # Store (symbol: str, trade_date: date) tuples; cap at 50 entries.
        self._day_trades: deque = deque(maxlen=50)

    def is_day_trade(self, symbol: str, entry_time: datetime) -> bool:
        """Check if exiting now would be a day trade.

        A day trade occurs when a position is opened and closed on the same
        calendar day (in UTC). If entry_time is None or naive, the check is
        performed using today's UTC date.

        Args:
            symbol: Ticker symbol.
            entry_time: UTC datetime when the position was entered. May be
                timezone-aware or naive (assumed UTC if naive).

        Returns:
            bool: True if closing now would complete a same-day round trip.

        Example:
            >>> from datetime import datetime, timezone
            >>> guard = PDTGuard()
            >>> entry = datetime.now(timezone.utc)
            >>> guard.is_day_trade("AAPL", entry)
            True
        """
        if entry_time is None:
            return False

        today_utc = datetime.now(timezone.utc).date()

        # Normalize entry_time to UTC date.
        if entry_time.tzinfo is not None:
            entry_date = entry_time.astimezone(timezone.utc).date()
        else:
            # Assume naive datetime is UTC.
            entry_date = entry_time.date()

        return entry_date == today_utc

    def can_exit(
        self,
        symbol: str,
        entry_time: datetime,
        force: bool = False,
    ) -> bool:
        """Determine if an exit is allowed under PDT rules.

        Decision logic:
          1. If force=True: allow exit regardless (circuit breaker bypass).
          2. If closing is NOT a day trade (overnight position): always allow.
          3. If day trade and rolling_count < max_day_trades: allow.
          4. If simulation_mode: log would-block but allow anyway.
          5. Otherwise: log WARNING and block.

        Args:
            symbol: Ticker symbol.
            entry_time: UTC datetime when position was entered.
            force: If True, bypass PDT check (circuit breaker). Default False.

        Returns:
            bool: True if exit is permitted.

        Example:
            >>> guard = PDTGuard()
            >>> for _ in range(3):
            ...     guard.record_day_trade("X")
            >>> guard.can_exit("AAPL", datetime.now(timezone.utc))
            False  # at limit, same-day exit blocked
            >>> guard.can_exit("AAPL", datetime.now(timezone.utc), force=True)
            True   # circuit breaker bypass
        """
        # Force bypass — used by circuit breaker emergency liquidation.
        if force:
            logger.info(
                "PDT BYPASS (force=True) for %s — proceeding with exit", symbol
            )
            return True

        # Overnight exit: always allowed (not a day trade).
        if not self.is_day_trade(symbol, entry_time):
            return True

        # Same-day exit: check rolling count.
        rolling_count = self.get_rolling_count()

        if rolling_count < self.max_day_trades:
            logger.info(
                "PDT allowed: %s same-day exit is day trade #%d/%d",
                symbol,
                rolling_count + 1,
                self.max_day_trades,
            )
            return True

        # At or over the limit.
        if self.simulation_mode:
            logger.info(
                "PDT simulation: would block same-day exit for %s "
                "(rolling=%d, limit=%d) — allowing in simulation mode",
                symbol,
                rolling_count,
                self.max_day_trades,
            )
            return True

        logger.warning(
            "PDT BLOCK: cannot close %s same-day — rolling count=%d equals "
            "limit=%d. Exit DENIED.",
            symbol,
            rolling_count,
            self.max_day_trades,
        )
        return False

    def record_day_trade(self, symbol: str, trade_date: date = None) -> None:
        """Record a completed day trade.

        Appends the symbol and date to the rolling deque. If trade_date is
        not provided, today's UTC date is used.

        Args:
            symbol: Symbol that was day traded.
            trade_date: Date of the day trade. Defaults to today (UTC).

        Example:
            >>> guard = PDTGuard()
            >>> guard.record_day_trade("AAPL")
            >>> guard.get_rolling_count()
            1
        """
        if trade_date is None:
            trade_date = datetime.now(timezone.utc).date()

        self._day_trades.append((symbol, trade_date))
        logger.info(
            "Day trade recorded: %s on %s (rolling_count=%d)",
            symbol,
            trade_date.isoformat(),
            self.get_rolling_count(),
        )

    def get_rolling_count(self) -> int:
        """Get current rolling 5-business-day day trade count.

        Counts only those trades whose trade_date falls on or after the
        cutoff returned by _get_5_business_days_ago().

        Returns:
            int: Count of day trades within the rolling window.

        Example:
            >>> guard = PDTGuard()
            >>> guard.record_day_trade("AAPL")
            >>> guard.get_rolling_count()
            1
        """
        cutoff = self._get_5_business_days_ago()
        count = sum(1 for _, td in self._day_trades if td >= cutoff)
        return count

    def get_reset_date(self) -> Optional[date]:
        """Get the date when the oldest day trade expires from the window.

        Finds the oldest trade within the current 5-business-day window and
        computes the next day after its expiry (the first day it no longer
        counts against the PDT limit).

        Returns:
            Optional[date]: The calendar date on which the rolling count will
                drop by 1, or None if no day trades are currently in the window.

        Example:
            >>> guard = PDTGuard()
            >>> guard.record_day_trade("AAPL")
            >>> reset = guard.get_reset_date()
            >>> reset is not None
            True
        """
        cutoff = self._get_5_business_days_ago()
        in_window = [td for _, td in self._day_trades if td >= cutoff]
        if not in_window:
            return None

        oldest_trade_date = min(in_window)

        # Walk forward from oldest_trade_date adding 5 business days to find
        # the expiry date (the day the trade falls off the window).
        expiry = oldest_trade_date
        bdays_added = 0
        while bdays_added < 5:
            expiry += timedelta(days=1)
            # weekday(): Mon=0 ... Fri=4, Sat=5, Sun=6
            if expiry.weekday() < 5:
                bdays_added += 1

        return expiry

    def _get_5_business_days_ago(self) -> date:
        """Compute the date 5 business days ago from today.

        Walks backward from today, decrementing one day at a time and
        skipping Saturday (weekday 5) and Sunday (weekday 6) until five
        business days have been counted.

        Returns:
            date: The date that is exactly 5 business days before today.

        Example:
            >>> guard = PDTGuard()
            >>> cutoff = guard._get_5_business_days_ago()
            >>> isinstance(cutoff, date)
            True
        """
        today = datetime.now(timezone.utc).date()
        current = today
        bdays_back = 0
        while bdays_back < 5:
            current -= timedelta(days=1)
            # weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
            if current.weekday() < 5:
                bdays_back += 1
        return current

    # ------------------------------------------------------------------
    # Legacy compatibility methods — kept so existing call sites that pass
    # current_count as a positional argument continue to work.
    # ------------------------------------------------------------------

    def get_current_count(self) -> int:
        """Return the rolling day trade count (legacy alias).

        Returns:
            int: Current rolling 5-business-day day trade count.

        Example:
            >>> guard.get_current_count()
            0
        """
        return self.get_rolling_count()


def get_pdt_guard() -> "PDTGuard":
    """Get or create the singleton PDTGuard instance.

    Returns:
        PDTGuard: The global singleton PDTGuard.

    Example:
        >>> guard = get_pdt_guard()
        >>> guard.get_rolling_count()
        0
    """
    global _PDT_INSTANCE, _pdt_instance
    if _PDT_INSTANCE is None:
        _PDT_INSTANCE = PDTGuard()
        _pdt_instance = _PDT_INSTANCE
    return _PDT_INSTANCE


def reset_pdt_guard() -> None:
    """Reset the singleton PDTGuard (used in tests).

    Example:
        >>> reset_pdt_guard()
    """
    global _PDT_INSTANCE, _pdt_instance
    _PDT_INSTANCE = None
    _pdt_instance = None
