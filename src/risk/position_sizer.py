"""Position sizer using fixed dollar risk per trade.

Computes share count based on $8 fixed risk per trade and per-layer
ATR-based stop distances. Caps position value at 25% of equity ($125).

Sizing model:
    DOLLAR_RISK = $8.00 (1.6% of $500)
    MAX_POSITION_VALUE = equity * 0.25 (= $125 at $500)

    stop_distance = ATR * layer_multiplier (layer-specific)
    shares = floor(DOLLAR_RISK / stop_distance)
    shares = min(shares, floor(MAX_POSITION_VALUE / current_price))

Per-layer stop multipliers:
    L1_VWAP_MR:   1.0 * ATR(14, 1-min)
    L2_ORB:       opening_range_size * 0.5 (stored in signal.metadata["stop_distance"])
    L3_RSI_SCALP: 0.5 * ATR(14, 5-min)
    L4_VOL_SURGE: 1.5 * ATR(14, 1-min)

For L2, stop_distance is pre-computed and stored in signal.metadata["stop_distance"].
For other layers, stop_price is in signal.stop_price.
Stop distance = abs(signal_price - stop_price).

Example:
    >>> sizer = PositionSizer(dollar_risk_per_trade=8.0)
    >>> signal = SignalResult(
    ...     symbol="AAPL", signal="BUY", confidence=0.8,
    ...     signal_price=150.0, layer_name="L1_VWAP_MR",
    ...     stop_price=149.0)
    >>> shares = sizer.compute_shares(signal, equity=500.0,
    ...     buying_power=500.0, current_open_positions=0)
    >>> shares
    8  # $8.00 / $1.00 stop = 8, capped at $125/150=0→0 ... see example notes
"""

import math
from typing import Optional

from src.strategy.base_strategy import SignalResult
from src.utils.logger import get_logger

logger = get_logger("risk.position_sizer")


class PositionSizer:
    """Computes share count using fixed dollar risk model.

    Sizing model:
        DOLLAR_RISK = $8.00 (1.6% of $500)
        MAX_POSITION_VALUE = equity * max_position_pct (default 0.25 * equity)

        stop_distance = abs(signal.signal_price - signal.stop_price)
                        OR signal.metadata["stop_distance"] if available
        shares = floor(DOLLAR_RISK / stop_distance)
        shares = min(shares, floor(MAX_POSITION_VALUE / current_price))

    If computed shares < 1: returns 0 and logs.

    Per-layer stop multipliers (applied inside the strategy, not here):
        L1_VWAP_MR:   1.0 * ATR(14, 1-min)
        L2_ORB:       opening_range_size * 0.5 (pre-computed in metadata)
        L3_RSI_SCALP: 0.5 * ATR(14, 5-min)
        L4_VOL_SURGE: 1.5 * ATR(14, 1-min)

    Attributes:
        dollar_risk_per_trade: Fixed dollar risk per trade. Default 8.00.
        max_position_pct: Maximum position as fraction of equity. Default 0.25.
        max_positions: Maximum simultaneous positions. Default 3.

    Example:
        >>> sizer = PositionSizer(dollar_risk_per_trade=8.0)
        >>> from src.strategy.base_strategy import SignalResult
        >>> signal = SignalResult(
        ...     symbol="AAPL", signal="BUY", confidence=0.75,
        ...     signal_price=150.0, layer_name="L1_VWAP_MR", stop_price=149.0)
        >>> shares = sizer.compute_shares(signal, equity=500.0,
        ...     buying_power=500.0, current_open_positions=0)
    """

    def __init__(
        self,
        dollar_risk_per_trade: float = 150.00,
        max_position_pct: float = 0.10,
        max_positions: int = 15,
        max_position_value: float = 10000.00,
        **kwargs,
    ):
        """Initialize the position sizer.

        Args:
            dollar_risk_per_trade: Fixed dollar amount to risk per trade.
            max_position_pct: Maximum position size as a fraction of equity.
            max_positions: Maximum simultaneous open positions.
            max_position_value: Hard cap on position value in dollars.
        """
        self.dollar_risk = dollar_risk_per_trade
        self.dollar_risk_per_trade = dollar_risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.max_position_value = max_position_value

        logger.info(
            f"PositionSizer loaded — "
            f"dollar_risk=${self.dollar_risk:.2f}, "
            f"max_positions={self.max_positions}, "
            f"max_position_value=${self.max_position_value:,.0f}"
        )
        if self.dollar_risk < 50:
            logger.warning(f"dollar_risk_per_trade is very low (${self.dollar_risk}) — check config.yml risk section")
        if self.max_positions < 5:
            logger.warning(f"max_positions is very low ({self.max_positions}) — check config.yml risk section")

    def compute_shares(
        self,
        signal: SignalResult,
        equity: float,
        buying_power: float,
        current_open_positions: int,
        regime_scalar: float = 1.0,
    ) -> int:
        """Compute number of shares to buy for a signal.

        Applies the fixed dollar risk model:
          1. Reject if current_open_positions >= max_positions.
          2. Compute max_position_value = equity * max_position_pct.
          3. Determine stop_distance from metadata["stop_distance"] or
             abs(signal_price - stop_price).
          4. Reject if stop_distance <= 0.
          5. raw_shares = floor(dollar_risk / stop_distance).
          6. max_by_value = floor(max_position_value / signal_price).
          7. shares = min(raw_shares, max_by_value).
          8. Apply regime scalar: shares = floor(shares * regime_scalar).
          9. Reject if shares < 1.
          10. Cap at buying power.
          11. Reject if still < 1.

        Args:
            signal: SignalResult with signal_price, stop_price, layer_name,
                and optional metadata["stop_distance"].
            equity: Current account equity in dollars.
            buying_power: Available buying power in dollars.
            current_open_positions: Number of currently open positions.
            regime_scalar: Size multiplier from regime filter. 1.0 for BULL,
                0.5 for BEAR. Default 1.0.

        Returns:
            int: Number of shares to buy. 0 if trade should be skipped.

        Example:
            >>> from src.strategy.base_strategy import SignalResult
            >>> sizer = PositionSizer()
            >>> sig = SignalResult("AAPL", "BUY", 0.8, 150.0, "L1_VWAP_MR",
            ...                   stop_price=148.0)
            >>> shares = sizer.compute_shares(sig, equity=500.0,
            ...     buying_power=500.0, current_open_positions=1,
            ...     regime_scalar=1.0)
            >>> shares  # $8 / $2 stop = 4 shares; capped at 500*0.25/150=0→0
            4
        """
        # Step 1: Position count guard.
        if current_open_positions >= self.max_positions:
            logger.info(
                "PositionSizer: at max positions (%d/%d) for %s/%s — returning 0",
                current_open_positions,
                self.max_positions,
                signal.symbol,
                signal.layer_name,
            )
            return 0

        # Step 2: Maximum allowed position value (capped by hard limit).
        max_position_value = min(equity * self.max_position_pct, self.max_position_value)

        # Step 3: Determine stop distance.
        stop_distance: float = 0.0
        if signal.metadata and "stop_distance" in signal.metadata:
            stop_distance = float(signal.metadata["stop_distance"])
        elif signal.stop_price and signal.signal_price:
            stop_distance = abs(float(signal.signal_price) - float(signal.stop_price))

        # Step 4: Reject zero stop distance.
        if stop_distance <= 0:
            logger.warning(
                "PositionSizer: zero/negative stop_distance for %s/%s "
                "(signal_price=%.4f, stop_price=%.4f) — returning 0",
                signal.symbol,
                signal.layer_name,
                float(signal.signal_price) if signal.signal_price else 0.0,
                float(signal.stop_price) if signal.stop_price else 0.0,
            )
            return 0

        # Step 5: Raw shares from dollar risk.
        raw_shares = int(self.dollar_risk_per_trade / stop_distance)

        # Step 6: Maximum shares by position value cap.
        if signal.signal_price and float(signal.signal_price) > 0:
            max_by_value = int(max_position_value / float(signal.signal_price))
        else:
            logger.warning(
                "PositionSizer: zero signal_price for %s/%s — returning 0",
                signal.symbol,
                signal.layer_name,
            )
            return 0

        # Step 7: Take the smaller of the two.
        shares = min(raw_shares, max_by_value)

        # Step 8: Apply regime scalar (reduces size in BEAR regime).
        shares = int(shares * regime_scalar)

        # Step 9: Reject if still below 1 share.
        if shares < 1:
            logger.info(
                "PositionSizer: position too small for %s/%s "
                "(price=%.4f, stop_distance=%.4f, raw=%d, max_by_value=%d, "
                "regime_scalar=%.2f) — returning 0",
                signal.symbol,
                signal.layer_name,
                float(signal.signal_price),
                stop_distance,
                raw_shares,
                max_by_value,
                regime_scalar,
            )
            return 0

        # Step 10: Cap at available buying power.
        position_cost = shares * float(signal.signal_price)
        if position_cost > buying_power:
            shares = int(buying_power / float(signal.signal_price))
            logger.info(
                "PositionSizer: buying power cap applied for %s/%s — "
                "shares capped from original to %d (buying_power=%.2f)",
                signal.symbol,
                signal.layer_name,
                shares,
                buying_power,
            )

        # Step 11: Reject if buying power cap brought shares below 1.
        if shares < 1:
            logger.info(
                "PositionSizer: insufficient buying power for %s/%s "
                "(price=%.4f, buying_power=%.2f) — returning 0",
                signal.symbol,
                signal.layer_name,
                float(signal.signal_price),
                buying_power,
            )
            return 0

        entry_price = float(signal.signal_price)
        final_shares = shares
        logger.debug(
            f"Sizing {signal.symbol}: price=${entry_price:.2f}, "
            f"stop_dist=${stop_distance:.4f}/share, "
            f"raw={raw_shares}, capped={final_shares}, "
            f"value=${final_shares * entry_price:.2f}"
        )
        logger.info(
            "PositionSizer: %s/%s -> %d shares @ $%.4f "
            "(stop_dist=%.4f, risk=$%.2f, position=$%.2f, regime=%.2f)",
            signal.symbol,
            signal.layer_name,
            final_shares,
            entry_price,
            stop_distance,
            stop_distance * final_shares,
            final_shares * entry_price,
            regime_scalar,
        )
        return final_shares
