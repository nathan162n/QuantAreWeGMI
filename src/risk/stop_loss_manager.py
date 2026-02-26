"""Hard and trailing stop-loss management for open positions.

Tracks stop levels on all open positions, updates trailing stops
based on highest price since entry, and identifies positions that
have breached their stops or exceeded the maximum holding period.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import yaml

from src.database import repository
from src.utils.logger import get_logger

logger = get_logger("risk.stop_loss_manager")


def _load_config() -> dict:
    """Load risk and strategy configuration from config.yml.

    Returns:
        dict: Combined risk and strategy configuration.

    Example:
        >>> config = _load_config()
        >>> config["atr_stop_multiplier"]
        2.0
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return {**config.get("risk", {}), **config.get("strategy", {})}


def _load_full_config() -> dict:
    """Load the full config.yml as a dict. Returns {} on error."""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yml"
        )
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class StopLossManager:
    """Manages hard and trailing stop-losses for all open positions.

    Hard stop: entry_price - (ATR_STOP_MULTIPLIER * ATR)
    Trailing stop: max(initial_stop, highest_since_entry - ATR_TRAIL_MULTIPLIER * ATR)
    The trailing stop never moves down.

    Attributes:
        config: Risk configuration parameters.
        atr_stop_mult: ATR multiplier for initial hard stop. Default 2.0.
        atr_trail_mult: ATR multiplier for trailing stop. Default 1.5.
        max_holding_days: Maximum days to hold a position. Default 15.
    """

    def __init__(self):
        """Initialize the stop loss manager with configuration.

        Example:
            >>> slm = StopLossManager()
        """
        self.config = _load_config()
        self.atr_stop_mult = self.config.get("atr_stop_multiplier", 2.0)
        self.atr_trail_mult = self.config.get("atr_trail_multiplier", 1.5)
        self.max_holding_days = self.config.get("max_holding_days", 15)

    def compute_initial_stop(self, entry_price: float, atr: float) -> float:
        """Compute the initial hard stop price.

        Formula: stop = entry_price - (ATR_STOP_MULTIPLIER * atr)

        Args:
            entry_price: Position entry price.
            atr: Current ATR(14) value for the symbol.

        Returns:
            float: Initial stop price, always below entry price.

        Example:
            >>> slm = StopLossManager()
            >>> slm.compute_initial_stop(100.0, 2.5)
            95.0  # 100 - (2.0 * 2.5)
        """
        stop = entry_price - (self.atr_stop_mult * atr)
        return round(stop, 6)

    def compute_stop_with_floor(
        self,
        symbol: str,
        entry_price: float,
        atr_value: float,
        atr_multiplier: float,
        min_stop_pct: float,
    ) -> float:
        """Compute stop price with an ATR-based calculation and a percentage floor.

        Phase 2 fix: prevents 3bp stop-outs by ensuring the stop distance is
        at least min_stop_pct of entry price even when ATR is very small.

        Formula:
            atr_stop_distance  = atr_multiplier * atr_value
            pct_floor_distance = entry_price * min_stop_pct
            stop_distance      = max(atr_stop_distance, pct_floor_distance)
            stop_price         = entry_price - stop_distance

        The maximum of the two distances is used, which gives the lower (wider)
        stop price — protecting against runaway slippage while still respecting
        the ATR signal when ATR is meaningful.

        Args:
            symbol: Ticker symbol for logging.
            entry_price: Position entry price.
            atr_value: Current ATR value for the symbol.
            atr_multiplier: Multiplier applied to ATR.
            min_stop_pct: Minimum stop distance as a fraction of entry price
                (e.g. 0.003 = 0.3%).

        Returns:
            float: Stop price, rounded to 6 decimal places.

        Example:
            >>> slm.compute_stop_with_floor("AAPL", 100.0, 0.05, 1.0, 0.003)
            99.7  # max(0.05, 0.30) = 0.30 below entry
        """
        atr_stop_distance = atr_multiplier * atr_value
        pct_floor_distance = entry_price * min_stop_pct
        stop_distance = max(atr_stop_distance, pct_floor_distance)
        stop_price = entry_price - stop_distance

        logger.info(
            "Stop calc %s: atr_stop=%.4f pct_floor=%.4f final=%.4f (%.3f%%)",
            symbol,
            entry_price - atr_stop_distance,
            entry_price - pct_floor_distance,
            stop_price,
            stop_distance / entry_price * 100,
        )
        return round(stop_price, 6)

    def update_trailing_stop(
        self,
        position: dict,
        bar: dict,
        config: dict,
    ) -> Optional[float]:
        """Update trailing stop for an open position using configured trail parameters.

        Phase 3: The trailing stop activates once the position has moved
        trailing_stop_activation_r × stop_distance in profit. Once active,
        it trails at trailing_stop_trail_pct × stop_distance below the
        session high. The stop never moves down.

        Args:
            position: Position dict with entry_price, stop_price,
                highest_price_since_entry, trailing_stop_price keys.
            bar: Current bar dict with a 'close' key.
            config: Full config dict (keys: risk.trailing_stop_activation_r,
                risk.trailing_stop_trail_pct).

        Returns:
            Optional[float]: New trailing stop price, or None if position data
                is invalid. The stop is guaranteed to be >= the initial stop.

        Example:
            >>> slm.update_trailing_stop(
            ...     {"entry_price": 100.0, "stop_price": 99.7,
            ...      "highest_price_since_entry": 101.0,
            ...      "trailing_stop_price": 99.7},
            ...     {"close": 101.5},
            ...     config)
            99.82  # trails 0.4 * 0.3 below highest 101.5
        """
        entry_price = float(position.get("entry_price", 0.0))
        stop_price = float(position.get("stop_price", 0.0))
        current_price = float(bar.get("close", 0.0))
        highest = float(position.get("highest_price_since_entry", entry_price))

        if entry_price <= 0 or stop_price <= 0 or current_price <= 0:
            return None

        stop_distance = entry_price - stop_price
        if stop_distance <= 0:
            return stop_price

        risk_cfg = config.get("risk", {})
        activation_r = float(risk_cfg.get("trailing_stop_activation_r", 1.0))
        trail_pct = float(risk_cfg.get("trailing_stop_trail_pct", 0.4))

        # Update highest observed price
        new_highest = max(highest, current_price)

        # Trailing stop only activates after activation_r × stop_distance profit
        profit = new_highest - entry_price
        if profit < activation_r * stop_distance:
            # Not yet activated — return the initial stop unchanged
            return round(stop_price, 6)

        # Trail at trail_pct × stop_distance below the highest seen
        trail_distance = trail_pct * stop_distance
        new_trail = new_highest - trail_distance

        # Never move the stop down
        current_trail = float(position.get("trailing_stop_price", stop_price))
        final_stop = max(current_trail, new_trail, stop_price)

        return round(final_stop, 6)

    def compute_trailing_stop(
        self,
        entry_price: float,
        highest_since_entry: float,
        atr: float,
        initial_stop: float,
    ) -> float:
        """Compute the trailing stop price that never moves down.

        Formula: max(initial_stop, highest_since_entry - ATR_TRAIL_MULTIPLIER * atr)

        Args:
            entry_price: Position entry price.
            highest_since_entry: Highest price observed since entry.
            atr: Current ATR(14) value.
            initial_stop: The initial hard stop level.

        Returns:
            float: Trailing stop price, guaranteed >= initial_stop.

        Example:
            >>> slm = StopLossManager()
            >>> slm.compute_trailing_stop(100.0, 108.0, 2.5, 95.0)
            104.25  # max(95.0, 108.0 - 1.5*2.5)
        """
        trail_stop = highest_since_entry - (self.atr_trail_mult * atr)
        final_stop = max(initial_stop, trail_stop)
        return round(final_stop, 6)

    def check_stops(
        self,
        positions: List[dict],
    ) -> List[dict]:
        """Check all positions for stop breaches and time-based exits.

        For each position, checks if:
        1. Current price <= trailing stop (TRAIL exit)
        2. Current price <= hard stop (STOP exit)
        3. Days held >= MAX_HOLDING_DAYS (TIME exit)

        Args:
            positions: List of position dicts with keys:
                trade_id, symbol, entry_price, current_price, stop_price,
                trailing_stop_price, highest_price_since_entry, days_held, atr.

        Returns:
            List[dict]: Positions that should be exited, each with added
                'exit_reason' key ('STOP', 'TRAIL', or 'TIME').

        Example:
            >>> exits = slm.check_stops([{
            ...     "trade_id": 1, "symbol": "AAPL",
            ...     "current_price": 94.0, "trailing_stop_price": 95.0,
            ...     "stop_price": 95.0, "days_held": 5}])
            >>> exits[0]["exit_reason"]
            'TRAIL'
        """
        exits = []

        for pos in positions:
            symbol = pos["symbol"]
            current_price = pos.get("current_price", 0)
            trailing_stop = pos.get("trailing_stop_price", 0)
            hard_stop = pos.get("stop_price", 0)
            days_held = pos.get("days_held", 0)
            trade_id = pos.get("trade_id")

            if trailing_stop > 0 and current_price <= trailing_stop:
                logger.info(
                    "TRAIL stop breach: %s price=$%.2f <= trail=$%.2f",
                    symbol,
                    current_price,
                    trailing_stop,
                )
                pos["exit_reason"] = "TRAIL"
                exits.append(pos)
                continue

            if hard_stop > 0 and current_price <= hard_stop:
                logger.info(
                    "HARD stop breach: %s price=$%.2f <= stop=$%.2f",
                    symbol,
                    current_price,
                    hard_stop,
                )
                pos["exit_reason"] = "STOP"
                exits.append(pos)
                continue

            if days_held >= self.max_holding_days:
                logger.info(
                    "TIME exit: %s held %d days (max=%d)",
                    symbol,
                    days_held,
                    self.max_holding_days,
                )
                pos["exit_reason"] = "TIME"
                exits.append(pos)
                continue

        if exits:
            logger.info(
                "Stop check: %d/%d positions breached",
                len(exits),
                len(positions),
            )
        return exits

    def update_trailing_stops(
        self,
        positions: List[dict],
    ) -> List[dict]:
        """Update trailing stops for all positions based on current prices.

        Updates the highest_price_since_entry and recalculates trailing
        stops. Persists updates to the Trade table.

        Args:
            positions: List of position dicts with keys:
                trade_id, symbol, current_price, entry_price,
                stop_price, highest_price_since_entry, atr.

        Returns:
            List[dict]: Updated positions with new trailing stop values.

        Example:
            >>> updated = slm.update_trailing_stops([{
            ...     "trade_id": 1, "symbol": "AAPL", "current_price": 108.0,
            ...     "entry_price": 100.0, "stop_price": 95.0,
            ...     "highest_price_since_entry": 105.0, "atr": 2.5}])
            >>> updated[0]["highest_price_since_entry"]
            108.0
        """
        updated = []

        for pos in positions:
            trade_id = pos.get("trade_id")
            symbol = pos["symbol"]
            current_price = pos.get("current_price", 0)
            highest = pos.get("highest_price_since_entry", current_price)
            initial_stop = pos.get("stop_price", 0)
            atr = pos.get("atr", 0)

            new_highest = max(highest, current_price)
            pos["highest_price_since_entry"] = new_highest

            if atr > 0:
                new_trail = self.compute_trailing_stop(
                    pos.get("entry_price", current_price),
                    new_highest,
                    atr,
                    initial_stop,
                )
                old_trail = pos.get("trailing_stop_price", 0)
                pos["trailing_stop_price"] = max(new_trail, old_trail)
            else:
                pos["trailing_stop_price"] = pos.get("trailing_stop_price", initial_stop)

            if trade_id:
                repository.update_trade(trade_id, {
                    "highest_price_since_entry": pos["highest_price_since_entry"],
                    "trailing_stop_price": pos["trailing_stop_price"],
                })

            updated.append(pos)

        return updated
