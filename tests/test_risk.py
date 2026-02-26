"""Unit tests for risk management: position sizing, stops, and circuit breaker.

All tests include numerical assertions verifying exact dollar amounts
and share counts for the $500 account configuration.
"""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss_manager import StopLossManager
from src.strategy.base_strategy import SignalResult


# ---------------------------------------------------------------------------
# Helper: build a SignalResult for PositionSizer tests
# ---------------------------------------------------------------------------

def _make_signal(
    symbol="AAPL",
    signal="BUY",
    confidence=0.8,
    signal_price=50.0,
    layer_name="L1_VWAP_MR",
    stop_price=49.0,
    metadata=None,
):
    """Return a SignalResult with the given parameters."""
    return SignalResult(
        symbol=symbol,
        signal=signal,
        confidence=confidence,
        signal_price=signal_price,
        layer_name=layer_name,
        stop_price=stop_price,
        metadata=metadata or {},
    )


def _make_sizer(
    dollar_risk=8.0,
    max_position_pct=0.25,
    max_positions=3,
):
    """Return a PositionSizer with explicit parameters."""
    return PositionSizer(
        dollar_risk_per_trade=dollar_risk,
        max_position_pct=max_position_pct,
        max_positions=max_positions,
    )


class TestPositionSizer:
    """Tests for fixed dollar-risk position sizing calibrated for $500."""

    def test_stop_distance_reduces_shares_at_high_risk(self):
        """Larger stop distance (higher per-share risk) produces fewer shares.

        price=$10, stop_tight=$9.50 -> dist=$0.50 -> raw=$8/$0.50=16 shares
        price=$10, stop_wide=$8.00  -> dist=$2.00 -> raw=$8/$2.00=4 shares
        Both capped by max_by_value = floor($500*0.25/$10) = 12 shares.
        tight: min(16, 12) = 12; wide: min(4, 12) = 4.
        """
        sizer = PositionSizer(dollar_risk_per_trade=8.0,
                              max_position_pct=0.25, max_positions=3)

        sig_tight = _make_signal(signal_price=10.0, stop_price=9.50)
        sig_wide = _make_signal(signal_price=10.0, stop_price=8.00)

        shares_tight = sizer.compute_shares(sig_tight, equity=500.0,
                                            buying_power=500.0,
                                            current_open_positions=0)
        shares_wide = sizer.compute_shares(sig_wide, equity=500.0,
                                           buying_power=500.0,
                                           current_open_positions=0)

        assert shares_wide < shares_tight, (
            f"Wider stop should produce fewer shares: "
            f"tight={shares_tight}, wide={shares_wide}"
        )
        assert shares_wide == 4, f"Expected 4 shares with $2 stop, got {shares_wide}"
        assert shares_tight == 12, f"Expected 12 shares (capped), got {shares_tight}"

    def test_position_value_never_exceeds_25pct_equity(self):
        """With equity=$500 and max_position_pct=0.25, shares*price <= $125.

        For every test price, position value must not exceed $125.
        """
        sizer = PositionSizer(dollar_risk_per_trade=8.0,
                              max_position_pct=0.25, max_positions=3)

        for price in [5.0, 10.0, 25.0, 50.0, 100.0]:
            stop_price = price * 0.95  # 5% stop
            sig = _make_signal(signal_price=price, stop_price=stop_price)
            shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                          current_open_positions=0)
            position_value = shares * price
            assert position_value <= 125.01, (
                f"Position value ${position_value:.2f} exceeds $125 "
                f"at price ${price}"
            )

    def test_position_size_zero_when_buying_power_insufficient(self):
        """buying_power < price -> returns 0 (can't afford even 1 share).

        price=$10, stop=$9 -> 8 shares needed, but buying_power=$5 < $10.
        After buying_power cap: floor($5/$10) = 0 -> returns 0.
        """
        sizer = PositionSizer(dollar_risk_per_trade=8.0,
                              max_position_pct=0.25, max_positions=3)
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=5.0,
                                      current_open_positions=0)
        assert shares == 0, (
            f"Expected 0 shares when buying_power=$5 < price=$10, got {shares}"
        )

    def test_position_size_zero_when_at_max_positions(self):
        """current_open_positions >= max_positions -> returns 0.

        The position count guard fires before any computation.
        """
        sizer = PositionSizer(dollar_risk_per_trade=8.0,
                              max_position_pct=0.25, max_positions=3)
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=3)
        assert shares == 0, (
            f"Expected 0 shares at max_positions=3 (3 open), got {shares}"
        )

    def test_shares_floored_to_whole_number(self):
        """compute_shares always returns a non-negative integer.

        With price=$10, stop=$9: raw=8, max_by_val=12 -> exactly 8 (integer).
        With regime_scalar=0.7: floor(8*0.7)=floor(5.6)=5 (integer, not 5.6).
        """
        sizer = PositionSizer(dollar_risk_per_trade=8.0,
                              max_position_pct=0.25, max_positions=3)
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        shares_full = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                           current_open_positions=0,
                                           regime_scalar=1.0)
        shares_partial = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                              current_open_positions=0,
                                              regime_scalar=0.7)

        assert isinstance(shares_full, int), "Must return int"
        assert shares_full == 8
        assert isinstance(shares_partial, int), "Must return int after regime scalar"
        assert shares_partial == 5  # floor(8 * 0.7) = floor(5.6) = 5
        assert shares_partial >= 0


class TestStopLossManager:
    """Tests for hard and trailing stop-loss logic."""

    def test_trailing_stop_never_decreases(self):
        """5-day price series: up 3 days, down 2 days.
        trailing_stop[day5] >= trailing_stop[day3].

        Example:
            >>> # Trailing stop should only ratchet up, never down
        """
        slm = StopLossManager()
        entry_price = 100.0
        atr_val = 2.5
        initial_stop = slm.compute_initial_stop(entry_price, atr_val)

        prices = [100.0, 103.0, 106.0, 104.0, 102.0]
        highest = entry_price
        trailing_stops = []

        for price in prices:
            highest = max(highest, price)
            trail = slm.compute_trailing_stop(entry_price, highest, atr_val, initial_stop)
            trailing_stops.append(trail)

        assert trailing_stops[4] >= trailing_stops[2], (
            f"Trailing stop decreased: day5={trailing_stops[4]} < "
            f"day3={trailing_stops[2]}"
        )

        for i in range(1, len(trailing_stops)):
            assert trailing_stops[i] >= trailing_stops[i - 1], (
                f"Trailing stop decreased at day {i + 1}"
            )

    def test_initial_stop_always_below_entry(self):
        """For any positive entry_price and ATR, stop < entry_price.

        Example:
            >>> slm.compute_initial_stop(100.0, 2.5)
            95.0  # Always below 100.0
        """
        slm = StopLossManager()
        for entry in [15.0, 50.0, 100.0, 500.0]:
            for atr_val in [0.1, 1.0, 5.0, 10.0]:
                stop = slm.compute_initial_stop(entry, atr_val)
                assert stop < entry, (
                    f"Stop ${stop} not below entry ${entry} with ATR={atr_val}"
                )

    def test_check_stops_trail_breach(self):
        """Current price below trailing stop should trigger TRAIL exit.

        Example:
            >>> # price=94 < trail=95 -> exit
        """
        slm = StopLossManager()
        positions = [{
            "trade_id": 1,
            "symbol": "AAPL",
            "current_price": 94.0,
            "trailing_stop_price": 95.0,
            "stop_price": 90.0,
            "days_held": 5,
        }]
        exits = slm.check_stops(positions)
        assert len(exits) == 1
        assert exits[0]["exit_reason"] == "TRAIL"

    def test_check_stops_time_exit(self):
        """Position held >= MAX_HOLDING_DAYS should trigger TIME exit.

        Example:
            >>> # days_held=15 >= max=15 -> TIME exit
        """
        slm = StopLossManager()
        positions = [{
            "trade_id": 1,
            "symbol": "AAPL",
            "current_price": 105.0,
            "trailing_stop_price": 95.0,
            "stop_price": 90.0,
            "days_held": 15,
        }]
        exits = slm.check_stops(positions)
        assert len(exits) == 1
        assert exits[0]["exit_reason"] == "TIME"


class TestCircuitBreaker:
    """Tests for portfolio-level circuit breaker triggers."""

    @patch("src.risk.circuit_breaker._load_config")
    @patch("src.risk.circuit_breaker.repository")
    def test_circuit_breaker_triggers_at_3_1_pct_loss(
        self, mock_repo, mock_config
    ):
        """Open=$500, current=$484.50 (3.1% loss) should trigger.

        3.1% > 3.0% daily limit -> trigger = True

        Example:
            >>> cb.check_all_conditions(484.50, 500.0)
            'Daily loss limit breached: -3.10%...'
        """
        mock_config.return_value = {
            "daily_loss_limit": 0.03,
            "weekly_loss_limit": 0.08,
            "max_drawdown_limit": 0.15,
            "consecutive_loss_days": 3,
        }
        mock_repo.get_portfolio_value_n_days_ago.return_value = 500.0
        mock_repo.get_peak_portfolio_value.return_value = 500.0
        mock_repo.get_consecutive_loss_days.return_value = 0

        cb = CircuitBreaker()
        cb.config = mock_config.return_value
        cb.daily_loss_limit = 0.03
        cb.weekly_loss_limit = 0.08
        cb.max_drawdown_limit = 0.15
        cb.consecutive_loss_days_limit = 3

        reason = cb.check_all_conditions(484.50, 500.0)
        assert reason is not None
        assert "Daily loss" in reason

    @patch("src.risk.circuit_breaker._load_config")
    @patch("src.risk.circuit_breaker.repository")
    def test_circuit_breaker_no_trigger_at_2_8_pct_loss(
        self, mock_repo, mock_config
    ):
        """Open=$500, current=$486.00 (2.8% loss) should NOT trigger.

        2.8% < 3.0% daily limit -> trigger = False

        Example:
            >>> cb.check_all_conditions(486.00, 500.0)
            None
        """
        mock_config.return_value = {
            "daily_loss_limit": 0.03,
            "weekly_loss_limit": 0.08,
            "max_drawdown_limit": 0.15,
            "consecutive_loss_days": 3,
        }
        mock_repo.get_portfolio_value_n_days_ago.return_value = 500.0
        mock_repo.get_peak_portfolio_value.return_value = 500.0
        mock_repo.get_consecutive_loss_days.return_value = 0

        cb = CircuitBreaker()
        cb.config = mock_config.return_value
        cb.daily_loss_limit = 0.03
        cb.weekly_loss_limit = 0.08
        cb.max_drawdown_limit = 0.15
        cb.consecutive_loss_days_limit = 3

        reason = cb.check_all_conditions(486.00, 500.0)
        assert reason is None

    @patch("src.risk.circuit_breaker._load_config")
    @patch("src.risk.circuit_breaker.repository")
    def test_circuit_breaker_triggers_on_3_consecutive_loss_days(
        self, mock_repo, mock_config
    ):
        """3 days of negative P&L should trigger.

        Example:
            >>> cb.check_all_conditions(498.0, 500.0)
            'Consecutive loss days limit: 3 days...'
        """
        mock_config.return_value = {
            "daily_loss_limit": 0.03,
            "weekly_loss_limit": 0.08,
            "max_drawdown_limit": 0.15,
            "consecutive_loss_days": 3,
        }
        mock_repo.get_portfolio_value_n_days_ago.return_value = 500.0
        mock_repo.get_peak_portfolio_value.return_value = 500.0
        mock_repo.get_consecutive_loss_days.return_value = 3

        cb = CircuitBreaker()
        cb.config = mock_config.return_value
        cb.daily_loss_limit = 0.03
        cb.weekly_loss_limit = 0.08
        cb.max_drawdown_limit = 0.15
        cb.consecutive_loss_days_limit = 3

        reason = cb.check_all_conditions(498.0, 500.0)
        assert reason is not None
        assert "Consecutive" in reason

    @patch("src.risk.circuit_breaker._load_config")
    @patch("src.risk.circuit_breaker.repository")
    def test_circuit_breaker_no_trigger_on_2_consecutive_loss_days(
        self, mock_repo, mock_config
    ):
        """2 days of negative P&L should NOT trigger (limit is 3).

        Example:
            >>> cb.check_all_conditions(498.0, 500.0)
            None  # Only 2 consecutive loss days
        """
        mock_config.return_value = {
            "daily_loss_limit": 0.03,
            "weekly_loss_limit": 0.08,
            "max_drawdown_limit": 0.15,
            "consecutive_loss_days": 3,
        }
        mock_repo.get_portfolio_value_n_days_ago.return_value = 500.0
        mock_repo.get_peak_portfolio_value.return_value = 500.0
        mock_repo.get_consecutive_loss_days.return_value = 2

        cb = CircuitBreaker()
        cb.config = mock_config.return_value
        cb.daily_loss_limit = 0.03
        cb.weekly_loss_limit = 0.08
        cb.max_drawdown_limit = 0.15
        cb.consecutive_loss_days_limit = 3

        reason = cb.check_all_conditions(498.0, 500.0)
        assert reason is None


# ---------------------------------------------------------------------------
# New PositionSizer tests using the real interface:
#   compute_shares(signal: SignalResult, equity, buying_power,
#                  current_open_positions, regime_scalar=1.0)
# ---------------------------------------------------------------------------

class TestPositionSizerExtended:
    """Additional position sizing tests with concrete numerical assertions.

    The actual PositionSizer formula:
        stop_distance = abs(signal_price - stop_price)
                        OR metadata["stop_distance"] if present
        raw_shares    = floor(dollar_risk / stop_distance)
        max_by_value  = floor(equity * max_position_pct / signal_price)
        shares        = min(raw_shares, max_by_value)
        shares        = floor(shares * regime_scalar)

    Default parameters: dollar_risk=8.0, max_position_pct=0.25, max_positions=3
    """

    def test_basic_formula_raw_shares_from_dollar_risk(self):
        """Verify raw_shares = floor(dollar_risk / stop_distance).

        price=$10, stop=$9 -> stop_distance=$1.0
        raw_shares = floor($8 / $1.0) = 8
        max_by_value = floor($500 * 0.25 / $10) = floor(12.5) = 12
        shares = min(8, 12) = 8
        """
        sizer = _make_sizer(dollar_risk=8.0, max_position_pct=0.25, max_positions=3)
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=0)
        assert shares == 8, (
            f"Expected 8 shares ($8 risk / $1 stop = 8), got {shares}"
        )

    def test_position_value_cap_applies_when_raw_exceeds_25pct(self):
        """When raw_shares would exceed 25% of equity, cap applies.

        price=$5, stop=$4 -> stop_distance=$1.0
        raw_shares = floor($8 / $1) = 8
        max_by_value = floor($500 * 0.25 / $5) = floor(25) = 25
        shares = min(8, 25) = 8 -> not capped (raw < max_by_value)

        Use price=$5, stop=$4.95 -> stop_distance=$0.05
        raw_shares = floor($8 / $0.05) = 160
        max_by_value = floor($500 * 0.25 / $5) = 25
        shares = min(160, 25) = 25 -> cap applies
        """
        sizer = _make_sizer(dollar_risk=8.0, max_position_pct=0.25, max_positions=3)
        sig = _make_signal(signal_price=5.0, stop_price=4.95)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=0)
        # raw_shares = floor(8 / 0.05) = 160, capped at 25
        assert shares == 25, (
            f"Expected 25 shares (capped at 25% of $500 / $5), got {shares}"
        )
        assert shares * 5.0 <= 500.0 * 0.25 + 0.01, (
            f"Position value ${shares * 5.0:.2f} exceeds 25% cap ($125)"
        )

    def test_position_value_never_exceeds_25pct_equity(self):
        """shares * price <= equity * 0.25 for all valid price/stop combinations."""
        sizer = _make_sizer(dollar_risk=8.0, max_position_pct=0.25, max_positions=3)
        equity = 500.0
        max_position_dollars = equity * 0.25  # = $125.0

        for price in [5.0, 10.0, 20.0, 50.0, 100.0]:
            for stop_dist in [0.05, 0.10, 0.25, 0.50, 1.0]:
                stop_price = price - stop_dist
                sig = _make_signal(signal_price=price, stop_price=stop_price)
                shares = sizer.compute_shares(sig, equity=equity, buying_power=equity,
                                              current_open_positions=0)
                position_value = shares * price
                assert position_value <= max_position_dollars + 0.01, (
                    f"Position ${position_value:.2f} exceeds max ${max_position_dollars:.2f} "
                    f"(price=${price}, stop_dist={stop_dist})"
                )

    def test_max_positions_blocks_new_entry(self):
        """Already at max_positions (3) -> compute_shares returns 0."""
        sizer = _make_sizer(max_positions=3)
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=3)
        assert shares == 0, (
            f"Expected 0 shares when already at max_positions=3, got {shares}"
        )

    def test_below_max_positions_allows_entry(self):
        """With current_open_positions=2 < max_positions=3, entry is allowed."""
        sizer = _make_sizer(max_positions=3)
        # price=$10, stop=$9 -> stop_dist=$1, raw=8, max_by_val=12 -> 8 shares
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=2)
        assert shares > 0, "Should allow entry with 2 of 3 positions filled"
        assert shares == 8

    def test_regime_scalar_halves_shares(self):
        """regime_scalar=0.5 produces floor(full_shares * 0.5) shares.

        price=$10, stop=$9 -> stop_dist=$1
        raw_shares = floor($8 / $1) = 8
        max_by_value = floor($500 * 0.25 / $10) = 12
        full_shares = min(8, 12) = 8
        half_shares = floor(8 * 0.5) = 4
        """
        sizer = _make_sizer(dollar_risk=8.0, max_position_pct=0.25, max_positions=3)
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        full_shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                           current_open_positions=0, regime_scalar=1.0)
        half_shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                           current_open_positions=0, regime_scalar=0.5)

        assert full_shares == 8, f"Expected 8 full shares, got {full_shares}"
        assert half_shares == 4, (
            f"Expected 4 half shares (floor(8*0.5)), got {half_shares}"
        )
        assert half_shares == math.floor(full_shares * 0.5)

    def test_zero_stop_distance_returns_zero(self):
        """stop_price == signal_price -> stop_distance=0 -> returns 0."""
        sizer = _make_sizer()
        sig = _make_signal(signal_price=50.0, stop_price=50.0)  # zero distance

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=0)
        assert shares == 0, "Zero stop distance must return 0"

    def test_zero_signal_price_returns_zero(self):
        """signal_price=0 -> compute_shares returns 0 (no division by zero)."""
        sizer = _make_sizer()
        sig = _make_signal(signal_price=0.0, stop_price=0.0)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=0)
        assert shares == 0

    def test_zero_equity_returns_zero(self):
        """equity=0 -> max_position_value=0 -> max_by_value=0 -> returns 0."""
        sizer = _make_sizer()
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        shares = sizer.compute_shares(sig, equity=0.0, buying_power=500.0,
                                      current_open_positions=0)
        assert shares == 0, "Zero equity must return 0"

    def test_insufficient_buying_power_returns_zero(self):
        """buying_power too small for even 1 share -> returns 0.

        price=$200, stop=$190 -> stop_dist=$10
        raw_shares = floor($8 / $10) = 0 -> returns 0 before buying_power check.

        Use price=$10, stop=$9 (8 shares needed), buying_power=$5:
        -> capped to floor($5 / $10) = 0 -> returns 0
        """
        sizer = _make_sizer()
        sig = _make_signal(signal_price=10.0, stop_price=9.0)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=5.0,
                                      current_open_positions=0)
        assert shares == 0, (
            f"Expected 0 shares when buying_power=$5 < price=$10, got {shares}"
        )

    def test_result_is_always_integer(self):
        """compute_shares always returns an int (no fractional shares)."""
        sizer = _make_sizer()

        for price in [5.0, 10.0, 25.0, 50.0]:
            for stop_dist in [0.10, 0.25, 0.50, 1.0]:
                sig = _make_signal(signal_price=price, stop_price=price - stop_dist)
                shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                              current_open_positions=0)
                assert isinstance(shares, int), (
                    f"compute_shares must return int, got {type(shares)} "
                    f"for price={price}, stop_dist={stop_dist}"
                )
                assert shares >= 0

    def test_larger_stop_distance_yields_fewer_shares(self):
        """Larger stop_distance (higher risk per share) produces fewer shares.

        price=$10, stop=$9 (dist=$1.0) -> raw=$8/$1=8 shares
        price=$10, stop=$8 (dist=$2.0) -> raw=$8/$2=4 shares
        """
        sizer = _make_sizer(dollar_risk=8.0, max_position_pct=0.25, max_positions=3)

        sig_tight = _make_signal(signal_price=10.0, stop_price=9.0)   # dist=$1.0
        sig_wide = _make_signal(signal_price=10.0, stop_price=8.0)    # dist=$2.0

        shares_tight = sizer.compute_shares(sig_tight, equity=500.0, buying_power=500.0,
                                            current_open_positions=0)
        shares_wide = sizer.compute_shares(sig_wide, equity=500.0, buying_power=500.0,
                                           current_open_positions=0)

        assert shares_tight == 8, f"Expected 8 shares with $1 stop, got {shares_tight}"
        assert shares_wide == 4, f"Expected 4 shares with $2 stop, got {shares_wide}"
        assert shares_wide < shares_tight, (
            f"Wider stop should produce fewer shares: "
            f"tight={shares_tight}, wide={shares_wide}"
        )

    def test_metadata_stop_distance_overrides_stop_price(self):
        """metadata['stop_distance'] takes priority over stop_price field.

        With stop_distance=2.0 in metadata (ignoring stop_price field):
        raw_shares = floor($8 / $2.0) = 4
        max_by_value = floor($500 * 0.25 / $10) = 12
        shares = min(4, 12) = 4
        """
        sizer = _make_sizer(dollar_risk=8.0, max_position_pct=0.25, max_positions=3)
        # stop_price would give dist=$1 (8 shares), but metadata overrides to dist=$2 (4 shares)
        sig = _make_signal(
            signal_price=10.0,
            stop_price=9.0,
            metadata={"stop_distance": 2.0},
        )

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=0)
        assert shares == 4, (
            f"Expected 4 shares (metadata stop_distance=$2 -> $8/$2=4), got {shares}"
        )

    def test_position_cap_enforced_on_cheap_stock(self):
        """Cheap stock with tiny stop: position cap at 25% of equity applies.

        price=$2, stop=$1.99 -> stop_dist=$0.01
        raw_shares = floor($8 / $0.01) = 800
        max_by_value = floor($500 * 0.25 / $2) = floor(62.5) = 62
        shares = min(800, 62) = 62 -> 62 * $2 = $124 <= $125
        """
        sizer = _make_sizer(dollar_risk=8.0, max_position_pct=0.25, max_positions=3)
        sig = _make_signal(signal_price=2.0, stop_price=1.99)

        shares = sizer.compute_shares(sig, equity=500.0, buying_power=500.0,
                                      current_open_positions=0)
        assert shares == 62, (
            f"Expected 62 shares (capped at 25% of $500 / $2), got {shares}"
        )
        assert shares * 2.0 <= 125.01, (
            f"Position value ${shares * 2.0:.2f} exceeds 25% cap ($125)"
        )
