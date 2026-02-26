"""Rule-based 5-bar forward forecast engine.

Produces a directional score (-100 to +100) for each incoming signal,
along with a price range estimate and key driver description. Attached
to all BUY signals and displayed in Discord notifications.

This is NOT machine learning — it is a weighted scoring system grounded
in well-known technical factors. The disclaimer is always included.
"""

from dataclasses import dataclass
from typing import Optional

from src.strategy.base_strategy import PredictionResult
from src.utils.logger import get_logger

logger = get_logger("strategy.prediction_engine")

_DISCLAIMER = (
    "Statistical estimate based on historical pattern similarity. "
    "Not a guaranteed prediction. Past patterns do not guarantee future price movements."
)


@dataclass
class PredictionInput:
    """Input data for the prediction engine.

    All inputs can be derived from the BarStore and current bar data.

    Attributes:
        symbol: Ticker symbol being scored.
        current_price: Current bar close price.
        distance_from_vwap: (price - vwap) / vwap as decimal.
        rsi_current: Current RSI value of the triggering layer.
        rsi_slope_5bar: Linear slope of RSI over last 5 bars.
        volume_ratio: current_volume / 20-bar avg volume.
        atr_current: ATR of the triggering timeframe.
        price_slope_5bar: Linear slope of close prices over 5 bars.
        regime: Market regime string — "BULL", "NEUTRAL", "BEAR", or "EXTREME".
        layer_name: Which strategy layer generated the signal.

    Example:
        >>> inp = PredictionInput(
        ...     symbol="AAPL", current_price=150.0,
        ...     distance_from_vwap=-0.008, rsi_current=38.0,
        ...     rsi_slope_5bar=0.5, volume_ratio=1.2, atr_current=0.45,
        ...     price_slope_5bar=-0.10, regime="BULL", layer_name="L1_VWAP_MR")
    """

    symbol: str
    current_price: float
    distance_from_vwap: float
    rsi_current: float
    rsi_slope_5bar: float
    volume_ratio: float
    atr_current: float
    price_slope_5bar: float
    regime: str
    layer_name: str


class PredictionEngine:
    """Score-based 5-bar forward forecast for trading signals.

    Scoring components (sum clipped to [-100, +100]):
        +20  price below VWAP (mean reversion opportunity)
        -20  price above VWAP + 1 ATR (extended, potential fade)
        +15  RSI slope positive over 5 bars (recovering momentum)
        -15  RSI slope negative (weakening momentum)
        +15  volume_ratio > 1.5 (conviction behind move)
        -10  volume_ratio < 0.8 (low conviction)
        +10  price slope positive (momentum aligned with reversal)
        -10  price slope negative (price still declining)
        +10  Layer 3 RSI oversold (strongest mean reversion case)
        -20  regime = BEAR
        +5   regime = BULL

    Price range estimate:
        estimated_move = atr * (abs(score) / 100.0)
        estimated_high = current_price + estimated_move
        estimated_low  = current_price - estimated_move

    Example:
        >>> engine = PredictionEngine()
        >>> result = engine.predict(inp)
        >>> result.direction
        '↑ BULLISH'
    """

    def predict(self, inp: PredictionInput) -> PredictionResult:
        """Generate a 5-bar forward forecast for a signal.

        Args:
            inp: PredictionInput with all required technical factors.

        Returns:
            PredictionResult: Scored forecast with direction, range, and key driver.

        Example:
            >>> result = engine.predict(PredictionInput(
            ...     symbol="AAPL", current_price=150.0,
            ...     distance_from_vwap=-0.012, rsi_current=22.0,
            ...     rsi_slope_5bar=0.8, volume_ratio=1.8,
            ...     atr_current=0.5, price_slope_5bar=0.02,
            ...     regime="BULL", layer_name="L3_RSI_SCALP"))
            # Returns PredictionResult with score ~+80, direction "↑ BULLISH"
        """
        score = 0
        drivers = []

        # VWAP distance scoring
        if inp.distance_from_vwap < -0.005:
            score += 20
            drivers.append("Price below VWAP (mean reversion setup)")
        elif inp.distance_from_vwap > 0.005 and inp.atr_current > 0:
            vwap_atr_ratio = (inp.distance_from_vwap * inp.current_price) / inp.atr_current
            if vwap_atr_ratio > 1.0:
                score -= 20
                drivers.append("Price extended above VWAP (fade risk)")

        # RSI slope scoring
        if inp.rsi_slope_5bar > 0.1:
            score += 15
            drivers.append("RSI recovering (positive slope)")
        elif inp.rsi_slope_5bar < -0.1:
            score -= 15
            drivers.append("RSI declining (negative slope)")

        # Volume conviction
        if inp.volume_ratio > 1.5:
            score += 15
            drivers.append(f"Volume conviction ({inp.volume_ratio:.1f}x avg)")
        elif inp.volume_ratio < 0.8:
            score -= 10
            drivers.append("Low volume conviction")

        # Price momentum slope
        if inp.price_slope_5bar > 0:
            score += 10
            drivers.append("Price slope positive")
        elif inp.price_slope_5bar < 0:
            score -= 10
            drivers.append("Price slope negative")

        # Layer 3 RSI oversold bonus
        if inp.layer_name == "L3_RSI_SCALP" and inp.rsi_current < 25:
            score += 10
            drivers.append("RSI(7) extreme oversold reversal")

        # Regime adjustment
        if inp.regime == "BEAR":
            score -= 20
            drivers.append("BEAR regime (size reduced)")
        elif inp.regime == "BULL":
            score += 5
            drivers.append("BULL regime")

        # Clamp score
        score = max(-100, min(100, score))

        # Determine direction
        if score >= 20:
            direction = "↑ BULLISH"
        elif score <= -20:
            direction = "↓ BEARISH"
        else:
            direction = "→ NEUTRAL"

        # Confidence percentage
        confidence_pct = float(abs(score))

        # Price range estimate
        if inp.atr_current > 0:
            estimated_move = inp.atr_current * (abs(score) / 100.0)
        else:
            estimated_move = inp.current_price * 0.005
        estimated_high = round(inp.current_price + estimated_move, 4)
        estimated_low = round(inp.current_price - estimated_move, 4)

        # Key driver — most impactful single factor
        key_driver = drivers[0] if drivers else "Mixed signals"

        logger.debug(
            "Prediction for %s [%s]: score=%d, dir=%s, drivers=%s",
            inp.symbol,
            inp.layer_name,
            score,
            direction,
            drivers,
        )

        return PredictionResult(
            direction=direction,
            score=score,
            confidence_pct=confidence_pct,
            estimated_low=estimated_low,
            estimated_high=estimated_high,
            key_driver=key_driver,
            bars_horizon=5,
            disclaimer=_DISCLAIMER,
        )

    def build_input_from_bar_store(
        self,
        symbol: str,
        bar: dict,
        bar_store,
        layer_name: str,
        regime: str = "NEUTRAL",
        rsi_period: int = 14,
    ) -> Optional[PredictionInput]:
        """Build a PredictionInput from current bar data and BarStore.

        Convenience method that computes all required factors automatically.

        Args:
            symbol: Ticker symbol.
            bar: Current bar dict (open, high, low, close, volume).
            bar_store: BarStore instance.
            layer_name: Strategy layer name.
            regime: Current regime string. Default "NEUTRAL".
            rsi_period: RSI period. Default 14.

        Returns:
            Optional[PredictionInput]: Input object or None if insufficient data.

        Example:
            >>> inp = engine.build_input_from_bar_store(
            ...     "AAPL", bar, store, "L1_VWAP_MR", "BULL")
        """
        try:
            from src.data.indicators import rsi, atr, slope, volume_ratio

            df = bar_store.get_bars(symbol, "1Min", 25)
            if df is None or len(df) < 20:
                logger.debug(
                    "%s: insufficient bars (%d < 20) for prediction — confidence=None",
                    symbol, len(df) if df is not None else 0,
                )
                return None

            closes = df["close"]
            highs = df["high"]
            lows = df["low"]
            vols = df["volume"]

            rsi_vals = rsi(closes, rsi_period)
            rsi_current = float(rsi_vals[-1]) if not any(
                v != v for v in [rsi_vals[-1]]
            ) else 50.0

            rsi_slope_5bar = 0.0
            valid_rsi = [v for v in rsi_vals[-5:] if v == v]
            if len(valid_rsi) >= 2:
                import pandas as pd
                rsi_slope_5bar = slope(pd.Series(valid_rsi), min(5, len(valid_rsi)))

            atr_vals = atr(highs, lows, closes, 14)
            atr_current = float(atr_vals[-1]) if not any(
                v != v for v in [atr_vals[-1]]
            ) else 0.0

            price_slope_5bar = slope(closes, 5)
            vol_ratio = volume_ratio(vols, 20)
            session_vwap = bar_store.get_session_vwap(symbol)
            current_price = float(bar.get("close", closes.iloc[-1]))

            dist_from_vwap = (
                (current_price - session_vwap) / session_vwap
                if session_vwap > 0
                else 0.0
            )

            return PredictionInput(
                symbol=symbol,
                current_price=current_price,
                distance_from_vwap=dist_from_vwap,
                rsi_current=rsi_current,
                rsi_slope_5bar=rsi_slope_5bar,
                volume_ratio=vol_ratio,
                atr_current=atr_current,
                price_slope_5bar=price_slope_5bar,
                regime=regime,
                layer_name=layer_name,
            )
        except Exception as e:
            logger.error(
                "Failed to build PredictionInput for %s: %s", symbol, str(e)
            )
            return None


_ENGINE_INSTANCE = None


def get_prediction_engine() -> PredictionEngine:
    """Get or create the singleton PredictionEngine.

    Returns:
        PredictionEngine: The prediction engine instance.

    Example:
        >>> engine = get_prediction_engine()
    """
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = PredictionEngine()
    return _ENGINE_INSTANCE
