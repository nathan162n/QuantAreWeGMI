"""Unit tests for PDTGuard — rolling 5-business-day Pattern Day Trader enforcement.

Tests cover:
  - is_day_trade() same-day detection
  - can_exit() blocking logic and force bypass
  - record_day_trade() and get_rolling_count() rolling window arithmetic
  - simulation_mode pass-through
  - _get_5_business_days_ago() business day calculation
  - get_reset_date() expiry tracking
  - singleton management

All tests import from the actual src/risk/pdt_guard.py module which uses
an in-memory deque (no database dependency).
"""

import pytest
from datetime import datetime, date, timezone, timedelta

from src.risk.pdt_guard import PDTGuard, get_pdt_guard, reset_pdt_guard


# ---------------------------------------------------------------------------
# is_day_trade() tests
# ---------------------------------------------------------------------------

class TestIsDayTrade:

    def test_is_day_trade_same_day_returns_true(self):
        """Entry time = today (UTC) -> is_day_trade = True."""
        guard = PDTGuard()
        today_entry = datetime.now(timezone.utc).replace(hour=14, minute=30)
        assert guard.is_day_trade("AAPL", today_entry) is True

    def test_is_day_trade_yesterday_returns_false(self):
        """Entry time = yesterday -> is_day_trade = False."""
        guard = PDTGuard()
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        assert guard.is_day_trade("AAPL", yesterday) is False

    def test_is_day_trade_two_days_ago_returns_false(self):
        """Entry time = 2 days ago -> is_day_trade = False."""
        guard = PDTGuard()
        two_days_ago = datetime.now(timezone.utc) - timedelta(days=2)
        assert guard.is_day_trade("MSFT", two_days_ago) is False

    def test_is_day_trade_none_entry_time_returns_false(self):
        """Entry time = None -> is_day_trade = False (safe fallback)."""
        guard = PDTGuard()
        assert guard.is_day_trade("AAPL", None) is False

    def test_is_day_trade_today_morning_and_evening(self):
        """Both morning and evening of today are same-day."""
        guard = PDTGuard()
        today_morning = datetime.now(timezone.utc).replace(hour=9, minute=30)
        today_evening = datetime.now(timezone.utc).replace(hour=22, minute=0)
        assert guard.is_day_trade("TSLA", today_morning) is True
        assert guard.is_day_trade("TSLA", today_evening) is True

    def test_is_day_trade_symbol_does_not_affect_result(self):
        """Symbol parameter does not change the day-trade determination."""
        guard = PDTGuard()
        today_entry = datetime.now(timezone.utc).replace(hour=14, minute=0)
        for symbol in ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA", "SPY"]:
            assert guard.is_day_trade(symbol, today_entry) is True

    def test_is_day_trade_naive_datetime_treated_as_utc(self):
        """Naive datetime (no tzinfo) is treated as UTC for date comparison."""
        guard = PDTGuard()
        today_naive = datetime.utcnow().replace(hour=14, minute=30)
        assert guard.is_day_trade("AAPL", today_naive) is True


# ---------------------------------------------------------------------------
# can_exit() tests
# ---------------------------------------------------------------------------

class TestCanExit:

    def test_can_exit_returns_true_when_rolling_count_below_limit(self):
        """Rolling count = 2 < max = 3, same-day entry -> can_exit = True."""
        guard = PDTGuard(max_day_trades=3)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        assert guard.get_rolling_count() == 2

        today_entry = datetime.now(timezone.utc).replace(hour=13, minute=30)
        assert guard.can_exit("TSLA", today_entry) is True

    def test_can_exit_blocked_at_limit(self):
        """Rolling count = 3 = max_day_trades, same-day entry -> can_exit = False."""
        guard = PDTGuard(max_day_trades=3)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")
        assert guard.get_rolling_count() == 3

        today_entry = datetime.now(timezone.utc).replace(hour=13, minute=30)
        assert guard.can_exit("TSLA", today_entry) is False

    def test_can_exit_force_bypasses_limit(self):
        """At limit, force=True -> can_exit = True (circuit breaker bypass)."""
        guard = PDTGuard(max_day_trades=3)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")

        today_entry = datetime.now(timezone.utc).replace(hour=13, minute=30)
        assert guard.can_exit("TSLA", today_entry, force=True) is True

    def test_can_exit_overnight_always_allowed_at_limit(self):
        """At limit, yesterday entry -> can_exit = True (not a day trade)."""
        guard = PDTGuard(max_day_trades=3)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")

        yesterday_entry = datetime.now(timezone.utc) - timedelta(days=1)
        assert guard.can_exit("TSLA", yesterday_entry) is True

    def test_can_exit_true_at_zero_count(self):
        """Zero day trades recorded -> can_exit = True for any same-day entry."""
        guard = PDTGuard(max_day_trades=3)
        today_entry = datetime.now(timezone.utc).replace(hour=13, minute=30)
        assert guard.can_exit("AAPL", today_entry) is True

    def test_can_exit_simulation_mode_never_blocks(self):
        """In simulation_mode=True, can_exit always returns True even at limit."""
        guard = PDTGuard(max_day_trades=3, simulation_mode=True)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")
        assert guard.get_rolling_count() == 3

        today_entry = datetime.now(timezone.utc).replace(hour=13, minute=30)
        result = guard.can_exit("TSLA", today_entry)
        assert result is True, (
            "simulation_mode=True should never block exits even at limit"
        )

    def test_can_exit_overnight_older_entries_always_allowed(self):
        """Positions entered 3 days ago are always exit-able regardless of count."""
        guard = PDTGuard(max_day_trades=3)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")

        old_entry = datetime.now(timezone.utc) - timedelta(days=3)
        assert guard.can_exit("SPY", old_entry) is True

    def test_can_exit_blocked_for_multiple_symbols(self):
        """When at limit, different symbols are all blocked for same-day exits."""
        guard = PDTGuard(max_day_trades=3)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")

        today_entry = datetime.now(timezone.utc).replace(hour=14, minute=0)
        for symbol in ["TSLA", "AMZN", "GOOGL", "META"]:
            assert guard.can_exit(symbol, today_entry) is False, (
                f"Symbol {symbol} should be blocked at limit"
            )


# ---------------------------------------------------------------------------
# record_day_trade() and get_rolling_count() tests
# ---------------------------------------------------------------------------

class TestRecordDayTradeAndRollingCount:

    def test_rolling_count_starts_at_zero(self):
        """Fresh guard has rolling count of 0."""
        guard = PDTGuard()
        assert guard.get_rolling_count() == 0

    def test_rolling_count_increments_per_trade(self):
        """Each record_day_trade() increments the rolling count by 1."""
        guard = PDTGuard()
        guard.record_day_trade("A")
        assert guard.get_rolling_count() == 1

        guard.record_day_trade("B")
        assert guard.get_rolling_count() == 2

        guard.record_day_trade("C")
        assert guard.get_rolling_count() == 3

    def test_rolling_count_accurate_after_3_records(self):
        """After 3 records, get_rolling_count() == 3."""
        guard = PDTGuard()
        guard.record_day_trade("A")
        guard.record_day_trade("B")
        guard.record_day_trade("C")
        assert guard.get_rolling_count() == 3

    def test_record_day_trade_uses_today_by_default(self):
        """record_day_trade() without trade_date defaults to today (UTC)."""
        guard = PDTGuard()
        guard.record_day_trade("AAPL")
        today_utc = datetime.now(timezone.utc).date()
        stored_dates = [td for _, td in guard._day_trades]
        assert today_utc in stored_dates, "Default trade_date should be today UTC"

    def test_record_day_trade_with_explicit_date(self):
        """record_day_trade() accepts an explicit trade_date."""
        guard = PDTGuard()
        custom_date = datetime.now(timezone.utc).date()
        guard.record_day_trade("AAPL", trade_date=custom_date)
        assert guard.get_rolling_count() == 1
        stored = [td for _, td in guard._day_trades]
        assert custom_date in stored

    def test_rolling_count_excludes_trades_outside_5_business_days(self):
        """Trades recorded more than 5 business days ago are excluded from count.

        Uses 14 calendar days ago which is guaranteed to span at least two
        weekends and therefore exceed the 5-business-day rolling window.
        """
        guard = PDTGuard()
        # 14 calendar days covers 2 full weeks -> always > 5 business days
        old_date = datetime.now(timezone.utc).date() - timedelta(days=14)
        guard.record_day_trade("AAPL", trade_date=old_date)
        # The old trade should not be in the rolling 5-business-day window.
        count = guard.get_rolling_count()
        assert count == 0, (
            f"Trade from 14 days ago should be outside the rolling window, got count={count}"
        )

    def test_rolling_count_includes_trades_within_5_business_days(self):
        """Trades recorded within the last 5 business days are included."""
        guard = PDTGuard()
        # Record a trade from yesterday (within 5 business days).
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        guard.record_day_trade("AAPL", trade_date=yesterday)
        assert guard.get_rolling_count() == 1

    def test_rolling_count_mixed_old_and_recent(self):
        """Only trades within the rolling window are counted."""
        guard = PDTGuard()
        # Two trades from last week (outside window) + one from today (inside)
        old_date = datetime.now(timezone.utc).date() - timedelta(days=8)
        guard.record_day_trade("AAPL", trade_date=old_date)
        guard.record_day_trade("MSFT", trade_date=old_date)
        guard.record_day_trade("TSLA")  # today
        assert guard.get_rolling_count() == 1, (
            "Only today's trade should be in the rolling 5-business-day window"
        )

    def test_same_symbol_counted_twice(self):
        """Same symbol can be day-traded multiple times and each counts."""
        guard = PDTGuard()
        guard.record_day_trade("AAPL")
        guard.record_day_trade("AAPL")
        assert guard.get_rolling_count() == 2


# ---------------------------------------------------------------------------
# PDT state interaction: record then can_exit
# ---------------------------------------------------------------------------

class TestRecordAndCanExit:

    def test_record_3_then_same_day_exit_blocked(self):
        """Record 3 day trades, then same-day exit is blocked."""
        guard = PDTGuard(max_day_trades=3)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")

        today_entry = datetime.now(timezone.utc).replace(hour=13, minute=30)
        assert guard.can_exit("TSLA", today_entry) is False

    def test_record_2_then_same_day_exit_allowed(self):
        """Record 2 day trades, then same-day exit is still allowed."""
        guard = PDTGuard(max_day_trades=3)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")

        today_entry = datetime.now(timezone.utc).replace(hour=13, minute=30)
        assert guard.can_exit("NVDA", today_entry) is True

    def test_force_exit_always_true_regardless_of_count(self):
        """Force=True bypasses the guard at any count."""
        guard = PDTGuard(max_day_trades=3)
        # Record many trades
        for sym in ["A", "B", "C", "D", "E"]:
            guard.record_day_trade(sym)
        assert guard.get_rolling_count() == 5  # well above limit

        today_entry = datetime.now(timezone.utc)
        assert guard.can_exit("X", today_entry, force=True) is True


# ---------------------------------------------------------------------------
# simulation_mode tests
# ---------------------------------------------------------------------------

class TestSimulationMode:

    def test_simulation_mode_tracks_count_but_never_blocks(self):
        """simulation_mode=True: count is tracked correctly but no blocking."""
        guard = PDTGuard(max_day_trades=3, simulation_mode=True)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")
        assert guard.get_rolling_count() == 3

        today_entry = datetime.now(timezone.utc)
        # Never blocked in simulation mode
        assert guard.can_exit("TSLA", today_entry) is True

    def test_simulation_mode_false_does_block(self):
        """simulation_mode=False (default): exits blocked at limit."""
        guard = PDTGuard(max_day_trades=3, simulation_mode=False)
        guard.record_day_trade("AAPL")
        guard.record_day_trade("MSFT")
        guard.record_day_trade("NVDA")

        today_entry = datetime.now(timezone.utc)
        assert guard.can_exit("TSLA", today_entry) is False

    def test_simulation_mode_default_is_false(self):
        """PDTGuard default simulation_mode is False."""
        guard = PDTGuard()
        assert guard.simulation_mode is False


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestConfiguration:

    def test_default_max_day_trades_is_3(self):
        """Default max_day_trades is 3."""
        guard = PDTGuard()
        assert guard.max_day_trades == 3

    def test_custom_max_day_trades_4_blocks_at_4(self):
        """PDTGuard(max_day_trades=4): count=3 allowed, count=4 blocked."""
        guard = PDTGuard(max_day_trades=4)
        for sym in ["A", "B", "C"]:
            guard.record_day_trade(sym)
        assert guard.get_rolling_count() == 3

        today_entry = datetime.now(timezone.utc)
        assert guard.can_exit("D", today_entry) is True

        guard.record_day_trade("D")
        assert guard.get_rolling_count() == 4
        assert guard.can_exit("E", today_entry) is False

    def test_max_day_trades_0_always_blocks_same_day(self):
        """PDTGuard(max_day_trades=0) blocks all same-day exits immediately."""
        guard = PDTGuard(max_day_trades=0)
        today_entry = datetime.now(timezone.utc)
        assert guard.can_exit("AAPL", today_entry) is False


# ---------------------------------------------------------------------------
# _get_5_business_days_ago() tests
# ---------------------------------------------------------------------------

class TestBusinessDayCalculation:

    def test_5_business_days_ago_is_before_today(self):
        """The cutoff date is strictly before today."""
        guard = PDTGuard()
        cutoff = guard._get_5_business_days_ago()
        today = datetime.now(timezone.utc).date()
        assert cutoff < today, f"Cutoff {cutoff} should be before today {today}"

    def test_5_business_days_ago_is_a_weekday(self):
        """The cutoff date falls on a Monday-Friday (weekday 0-4)."""
        guard = PDTGuard()
        cutoff = guard._get_5_business_days_ago()
        assert cutoff.weekday() < 5, (
            f"Cutoff {cutoff} should be a weekday, got weekday={cutoff.weekday()}"
        )

    def test_5_business_days_ago_at_most_9_calendar_days(self):
        """5 business days ago is at most 9 calendar days ago (skipping 2 weekends)."""
        guard = PDTGuard()
        cutoff = guard._get_5_business_days_ago()
        today = datetime.now(timezone.utc).date()
        calendar_delta = (today - cutoff).days
        assert calendar_delta <= 9, (
            f"5 business days should be at most 9 calendar days, got {calendar_delta}"
        )

    def test_5_business_days_ago_at_least_5_calendar_days(self):
        """5 business days ago is at least 5 calendar days ago."""
        guard = PDTGuard()
        cutoff = guard._get_5_business_days_ago()
        today = datetime.now(timezone.utc).date()
        calendar_delta = (today - cutoff).days
        assert calendar_delta >= 5, (
            f"5 business days should be at least 5 calendar days, got {calendar_delta}"
        )


# ---------------------------------------------------------------------------
# get_reset_date() tests
# ---------------------------------------------------------------------------

class TestGetResetDate:

    def test_reset_date_is_none_when_no_trades(self):
        """No day trades -> get_reset_date() returns None."""
        guard = PDTGuard()
        assert guard.get_reset_date() is None

    def test_reset_date_is_after_oldest_trade(self):
        """Reset date is strictly after the oldest trade date in the window."""
        guard = PDTGuard()
        guard.record_day_trade("AAPL")
        reset = guard.get_reset_date()
        today = datetime.now(timezone.utc).date()
        assert reset is not None
        assert reset > today, f"Reset date {reset} should be after today {today}"

    def test_reset_date_is_a_weekday(self):
        """Reset date is always a weekday (Mon-Fri)."""
        guard = PDTGuard()
        guard.record_day_trade("AAPL")
        reset = guard.get_reset_date()
        assert reset is not None
        assert reset.weekday() < 5, (
            f"Reset date {reset} should be a weekday, got weekday={reset.weekday()}"
        )


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------

class TestSingleton:

    def test_get_pdt_guard_returns_pdt_guard_instance(self):
        """get_pdt_guard() returns a PDTGuard instance."""
        reset_pdt_guard()
        guard = get_pdt_guard()
        assert isinstance(guard, PDTGuard)
        reset_pdt_guard()

    def test_get_pdt_guard_returns_same_instance_twice(self):
        """get_pdt_guard() is a singleton."""
        reset_pdt_guard()
        guard1 = get_pdt_guard()
        guard2 = get_pdt_guard()
        assert guard1 is guard2
        reset_pdt_guard()

    def test_reset_pdt_guard_creates_new_instance(self):
        """After reset_pdt_guard(), get_pdt_guard() returns a fresh instance."""
        reset_pdt_guard()
        guard1 = get_pdt_guard()
        guard1.record_day_trade("AAPL")
        assert guard1.get_rolling_count() == 1

        reset_pdt_guard()
        guard2 = get_pdt_guard()
        assert guard2 is not guard1
        assert guard2.get_rolling_count() == 0, "Fresh guard should have 0 trades"
        reset_pdt_guard()
