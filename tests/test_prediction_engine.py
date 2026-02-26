"""Unit tests for the rule-based prediction engine.

Tests verify that each scoring factor produces reasonable outputs
and the composite score correctly weights and combines them.
"""

import pytest

from src.strategy.prediction_engine import PredictionEngine, PredictionInput
from src.strategy.base_strategy import PredictionResult


def _make_input(**overrides) -> PredictionInput:
    """Return a baseline PredictionInput, with any fields overridden by kwargs.

    Baseline represents a neutral-leaning, healthy setup:
        - price at VWAP (no distance),
        - RSI flat at 50,
        - volume ratio 1.0 (average),
        - flat price slope,
        - NEUTRAL regime,
        - L1 layer.
    """
    defaults = dict(
        symbol="AAPL",
        current_price=150.0,
        distance_from_vwap=0.0,
        rsi_current=50.0,
        rsi_slope_5bar=0.0,
        volume_ratio=1.0,
        atr_current=1.0,
        price_slope_5bar=0.0,
        regime="NEUTRAL",
        layer_name="L1_VWAP_MR",
    )
    defaults.update(overrides)
    return PredictionInput(**defaults)


@pytest.fixture
def engine():
    """Create a PredictionEngine instance with default config."""
    return PredictionEngine()


class TestPredictionEngine:
    """Tests for composite prediction scoring."""

    def test_predict_returns_prediction_result(self, engine):
        """predict() should always return a PredictionResult dataclass."""
        inp = _make_input()
        result = engine.predict(inp)
        assert isinstance(result, PredictionResult)

    def test_predict_result_has_required_fields(self, engine):
        """PredictionResult must expose all required fields."""
        inp = _make_input(
            distance_from_vwap=-0.012,
            rsi_current=28.0,
            rsi_slope_5bar=0.5,
            volume_ratio=1.8,
            price_slope_5bar=0.02,
            regime="BULL",
        )
        result = engine.predict(inp)
        assert hasattr(result, "direction")
        assert hasattr(result, "score")
        assert hasattr(result, "confidence_pct")
        assert hasattr(result, "estimated_low")
        assert hasattr(result, "estimated_high")
        assert hasattr(result, "key_driver")

    def test_predict_buy_bull_regime(self, engine):
        """Oversold setup in BULL regime should produce a bullish direction."""
        inp = _make_input(
            distance_from_vwap=-0.012,
            rsi_current=28.0,
            rsi_slope_5bar=0.5,
            volume_ratio=1.8,
            price_slope_5bar=0.02,
            regime="BULL",
        )
        result = engine.predict(inp)
        assert "BULLISH" in result.direction

    def test_predict_bull_regime_scores_higher_than_bear(self, engine):
        """BULL regime should produce a higher score than BEAR for the same setup."""
        base = dict(
            distance_from_vwap=-0.012,
            rsi_current=28.0,
            rsi_slope_5bar=0.5,
            volume_ratio=1.8,
            price_slope_5bar=0.02,
        )
        result_bull = engine.predict(_make_input(**base, regime="BULL"))
        result_bear = engine.predict(_make_input(**base, regime="BEAR"))
        assert result_bull.score > result_bear.score

    def test_predict_estimated_range_ordered(self, engine):
        """estimated_low must be strictly less than estimated_high."""
        inp = _make_input(
            distance_from_vwap=-0.012,
            rsi_current=28.0,
            rsi_slope_5bar=0.5,
            volume_ratio=1.8,
            atr_current=1.5,
            price_slope_5bar=0.02,
            regime="BULL",
        )
        result = engine.predict(inp)
        assert result.estimated_low < result.estimated_high

    def test_score_bounded_positive_extreme(self, engine):
        """Score must not exceed +100 for a maximally bullish input."""
        inp = _make_input(
            distance_from_vwap=-0.020,
            rsi_current=15.0,
            rsi_slope_5bar=2.0,
            volume_ratio=3.0,
            atr_current=1.0,
            price_slope_5bar=0.5,
            regime="BULL",
            layer_name="L3_RSI_SCALP",
        )
        result = engine.predict(inp)
        assert result.score <= 100

    def test_score_bounded_negative_extreme(self, engine):
        """Score must not go below -100 for a maximally bearish input."""
        inp = _make_input(
            distance_from_vwap=0.020,
            rsi_current=80.0,
            rsi_slope_5bar=-2.0,
            volume_ratio=0.3,
            atr_current=1.0,
            price_slope_5bar=-0.5,
            regime="BEAR",
        )
        result = engine.predict(inp)
        assert result.score >= -100

    def test_score_is_integer(self, engine):
        """Score must be an integer value."""
        result = engine.predict(_make_input())
        assert isinstance(result.score, int)

    def test_confidence_pct_equals_abs_score(self, engine):
        """confidence_pct should equal abs(score) for all results."""
        inp = _make_input(
            distance_from_vwap=-0.012,
            rsi_slope_5bar=0.5,
            volume_ratio=1.8,
            regime="BULL",
        )
        result = engine.predict(inp)
        assert result.confidence_pct == float(abs(result.score))

    def test_direction_bullish_when_score_high(self, engine):
        """score >= 20 should yield a BULLISH direction string."""
        inp = _make_input(
            distance_from_vwap=-0.012,
            rsi_slope_5bar=0.5,
            volume_ratio=2.0,
            price_slope_5bar=0.1,
            regime="BULL",
        )
        result = engine.predict(inp)
        # score = +20 (vwap) + 15 (rsi slope) + 15 (volume) + 10 (price slope) + 5 (bull) = 65
        assert result.score >= 20
        assert "BULLISH" in result.direction

    def test_direction_bearish_when_score_low(self, engine):
        """score <= -20 should yield a BEARISH direction string."""
        inp = _make_input(
            rsi_slope_5bar=-0.5,
            volume_ratio=0.5,
            price_slope_5bar=-0.1,
            regime="BEAR",
        )
        result = engine.predict(inp)
        # score = -15 (rsi slope) - 10 (volume) - 10 (price slope) - 20 (bear) = -55
        assert result.score <= -20
        assert "BEARISH" in result.direction

    def test_key_driver_is_string(self, engine):
        """key_driver must be a non-empty string."""
        result = engine.predict(_make_input())
        assert isinstance(result.key_driver, str)
        assert len(result.key_driver) > 0


class TestScoringFactors:
    """Tests for individual scoring factors via PredictionInput combinations."""

    def test_vwap_below_adds_score(self, engine):
        """Price significantly below VWAP should add +20 to score vs neutral."""
        neutral = engine.predict(_make_input(distance_from_vwap=0.0))
        oversold = engine.predict(_make_input(distance_from_vwap=-0.010))
        assert oversold.score > neutral.score

    def test_vwap_extended_above_subtracts_score(self, engine):
        """Price extended well above VWAP (> 1 ATR) should subtract score."""
        # distance_from_vwap=0.02 with price=150, atr=1.0 => ratio = 0.02*150/1.0 = 3.0 > 1
        extended = engine.predict(_make_input(distance_from_vwap=0.020, atr_current=1.0))
        neutral = engine.predict(_make_input(distance_from_vwap=0.0))
        assert extended.score < neutral.score

    def test_rsi_slope_positive_adds_score(self, engine):
        """Positive RSI slope should score higher than flat slope."""
        flat = engine.predict(_make_input(rsi_slope_5bar=0.0))
        recovering = engine.predict(_make_input(rsi_slope_5bar=0.5))
        assert recovering.score > flat.score

    def test_rsi_slope_negative_subtracts_score(self, engine):
        """Negative RSI slope should produce a lower score than flat slope."""
        flat = engine.predict(_make_input(rsi_slope_5bar=0.0))
        declining = engine.predict(_make_input(rsi_slope_5bar=-0.5))
        assert declining.score < flat.score

    def test_high_volume_adds_score(self, engine):
        """volume_ratio > 1.5 should produce a higher score than average volume."""
        avg_vol = engine.predict(_make_input(volume_ratio=1.0))
        high_vol = engine.predict(_make_input(volume_ratio=2.0))
        assert high_vol.score > avg_vol.score

    def test_low_volume_subtracts_score(self, engine):
        """volume_ratio < 0.8 should produce a lower score than average volume."""
        avg_vol = engine.predict(_make_input(volume_ratio=1.0))
        low_vol = engine.predict(_make_input(volume_ratio=0.5))
        assert low_vol.score < avg_vol.score

    def test_positive_price_slope_adds_score(self, engine):
        """Positive price slope should produce a higher score than flat."""
        flat = engine.predict(_make_input(price_slope_5bar=0.0))
        uptrend = engine.predict(_make_input(price_slope_5bar=0.1))
        assert uptrend.score > flat.score

    def test_negative_price_slope_subtracts_score(self, engine):
        """Negative price slope should produce a lower score than flat."""
        flat = engine.predict(_make_input(price_slope_5bar=0.0))
        downtrend = engine.predict(_make_input(price_slope_5bar=-0.1))
        assert downtrend.score < flat.score

    def test_l3_rsi_oversold_bonus(self, engine):
        """L3_RSI_SCALP layer with RSI < 25 should score higher than L1 same input."""
        l1 = engine.predict(_make_input(
            layer_name="L1_VWAP_MR",
            rsi_current=20.0,
        ))
        l3 = engine.predict(_make_input(
            layer_name="L3_RSI_SCALP",
            rsi_current=20.0,
        ))
        # L3 with RSI < 25 gets an extra +10
        assert l3.score > l1.score

    def test_l3_rsi_no_bonus_when_rsi_not_oversold(self, engine):
        """L3_RSI_SCALP layer with RSI >= 25 should NOT receive the oversold bonus."""
        l1 = engine.predict(_make_input(
            layer_name="L1_VWAP_MR",
            rsi_current=40.0,
        ))
        l3 = engine.predict(_make_input(
            layer_name="L3_RSI_SCALP",
            rsi_current=40.0,
        ))
        assert l3.score == l1.score

    def test_regime_bull_scores_higher_than_neutral(self, engine):
        """BULL regime should produce a higher score than NEUTRAL for identical input."""
        neutral = engine.predict(_make_input(regime="NEUTRAL"))
        bull = engine.predict(_make_input(regime="BULL"))
        assert bull.score > neutral.score

    def test_regime_bear_scores_lower_than_neutral(self, engine):
        """BEAR regime should produce a lower score than NEUTRAL for identical input."""
        neutral = engine.predict(_make_input(regime="NEUTRAL"))
        bear = engine.predict(_make_input(regime="BEAR"))
        assert bear.score < neutral.score

    def test_regime_bull_higher_than_bear(self, engine):
        """BULL regime should always score higher than BEAR regime."""
        bull = engine.predict(_make_input(regime="BULL"))
        bear = engine.predict(_make_input(regime="BEAR"))
        assert bull.score > bear.score

    def test_oversold_rsi_with_bear_regime_still_bounded(self, engine):
        """Even the most oversold RSI cannot push score above +100."""
        inp = _make_input(
            distance_from_vwap=-0.020,
            rsi_current=10.0,
            rsi_slope_5bar=3.0,
            volume_ratio=5.0,
            price_slope_5bar=1.0,
            regime="BULL",
            layer_name="L3_RSI_SCALP",
        )
        result = engine.predict(inp)
        assert -100 <= result.score <= 100
