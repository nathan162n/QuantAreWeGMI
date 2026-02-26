"""Tests for BarDispatcher — bar routing, deduplication, and cadence logic."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from src.data.bar_store import BarStore
from src.strategy.base_strategy import BaseStrategy, SignalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(symbol="AAPL", close=100.0, volume=5000):
    """Create a minimal bar dict."""
    return {
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": volume,
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc),
    }


def _make_mock_layer(layer_name: str, signal: str = "HOLD"):
    """Create a mock BaseStrategy layer that returns a given signal."""
    layer = MagicMock(spec=BaseStrategy)
    layer.layer_name = layer_name

    def _evaluate_bar(symbol, bar, bar_store, open_position=None):
        if signal == "BUY":
            return [SignalResult(
                symbol=symbol, signal="BUY", confidence=0.8,
                signal_price=bar["close"], layer_name=layer_name,
                stop_price=bar["close"] - 1.0,
            )]
        elif signal == "EXIT" and open_position is not None:
            return [SignalResult(
                symbol=symbol, signal="EXIT", confidence=1.0,
                signal_price=bar["close"], layer_name=layer_name,
                stop_price=0.0,
            )]
        return []

    layer.evaluate_bar.side_effect = _evaluate_bar
    return layer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dispatcher_imports():
    """BarDispatcher can be imported without errors."""
    from src.engine.bar_dispatcher import BarDispatcher
    assert BarDispatcher is not None


def test_dispatcher_returns_empty_on_no_signals():
    """When all layers return HOLD, dispatch returns empty list."""
    from src.engine.bar_dispatcher import BarDispatcher
    layers = [
        _make_mock_layer("L1_VWAP_MR", "HOLD"),
        _make_mock_layer("L2_ORB", "HOLD"),
        _make_mock_layer("L3_RSI_SCALP", "HOLD"),
        _make_mock_layer("L4_VOL_SURGE", "HOLD"),
    ]
    dispatcher = BarDispatcher(layers, max_positions=3)
    store = BarStore()
    bar = _make_bar("AAPL")
    # No opening range → L2 won't fire even if signal is BUY
    signals = dispatcher.dispatch("AAPL", bar, store, {}, set(), layer2_enabled=True)
    buy_signals = [s for s in signals if s.signal == "BUY"]
    assert len(buy_signals) == 0


def test_dispatcher_routes_buy_signal():
    """When L1 returns BUY with no open position, signal is included."""
    from src.engine.bar_dispatcher import BarDispatcher
    layers = [
        _make_mock_layer("L1_VWAP_MR", "BUY"),
        _make_mock_layer("L2_ORB", "HOLD"),
        _make_mock_layer("L3_RSI_SCALP", "HOLD"),
        _make_mock_layer("L4_VOL_SURGE", "HOLD"),
    ]
    dispatcher = BarDispatcher(layers, max_positions=3)
    store = BarStore()
    bar = _make_bar("AAPL")
    signals = dispatcher.dispatch("AAPL", bar, store, {}, set(), layer2_enabled=True)
    buy_signals = [s for s in signals if s.signal == "BUY"]
    assert len(buy_signals) >= 1
    assert buy_signals[0].layer_name == "L1_VWAP_MR"


def test_dispatcher_dedup_blocks_second_entry_same_layer():
    """If L1 already has a position in AAPL, a new BUY from L1 is blocked."""
    from src.engine.bar_dispatcher import BarDispatcher
    layers = [
        _make_mock_layer("L1_VWAP_MR", "BUY"),
        _make_mock_layer("L2_ORB", "HOLD"),
        _make_mock_layer("L3_RSI_SCALP", "HOLD"),
        _make_mock_layer("L4_VOL_SURGE", "HOLD"),
    ]
    dispatcher = BarDispatcher(layers, max_positions=3)
    store = BarStore()
    bar = _make_bar("AAPL")
    # Pre-existing open position for this layer
    open_positions = {
        "AAPL_L1_VWAP_MR": {
            "symbol": "AAPL", "entry_price": 99.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "qty": 5, "layer_name": "L1_VWAP_MR", "stop_price": 98.0,
        }
    }
    signals = dispatcher.dispatch("AAPL", bar, store, open_positions, set(), layer2_enabled=True)
    buy_signals = [s for s in signals if s.signal == "BUY" and s.layer_name == "L1_VWAP_MR"]
    assert len(buy_signals) == 0, "Should not generate BUY when L1 position already open in AAPL"


def test_dispatcher_blocks_all_entries_at_max_positions():
    """When 3 positions are open, no new BUY signals are generated."""
    from src.engine.bar_dispatcher import BarDispatcher
    layers = [
        _make_mock_layer("L1_VWAP_MR", "BUY"),
        _make_mock_layer("L2_ORB", "BUY"),
        _make_mock_layer("L3_RSI_SCALP", "BUY"),
        _make_mock_layer("L4_VOL_SURGE", "BUY"),
    ]
    dispatcher = BarDispatcher(layers, max_positions=3)
    store = BarStore()
    bar = _make_bar("AAPL")
    # 3 open positions in different symbols
    open_positions = {
        "MSFT_L1_VWAP_MR": {"symbol": "MSFT", "entry_price": 300.0, "qty": 1, "layer_name": "L1_VWAP_MR", "stop_price": 298.0, "entry_time": datetime.now(timezone.utc)},
        "NVDA_L3_RSI_SCALP": {"symbol": "NVDA", "entry_price": 500.0, "qty": 1, "layer_name": "L3_RSI_SCALP", "stop_price": 495.0, "entry_time": datetime.now(timezone.utc)},
        "TSLA_L4_VOL_SURGE": {"symbol": "TSLA", "entry_price": 200.0, "qty": 1, "layer_name": "L4_VOL_SURGE", "stop_price": 197.0, "entry_time": datetime.now(timezone.utc)},
    }
    signals = dispatcher.dispatch("AAPL", bar, store, open_positions, set(), layer2_enabled=True)
    buy_signals = [s for s in signals if s.signal == "BUY"]
    assert len(buy_signals) == 0, "Should block all BUY signals when at max positions"


def test_dispatcher_blocks_entry_when_symbol_has_open_order():
    """Symbol with an open (pending) order should not generate new BUY."""
    from src.engine.bar_dispatcher import BarDispatcher
    layers = [
        _make_mock_layer("L1_VWAP_MR", "BUY"),
        _make_mock_layer("L2_ORB", "HOLD"),
        _make_mock_layer("L3_RSI_SCALP", "HOLD"),
        _make_mock_layer("L4_VOL_SURGE", "HOLD"),
    ]
    dispatcher = BarDispatcher(layers, max_positions=3)
    store = BarStore()
    bar = _make_bar("AAPL")
    open_orders = {"AAPL"}  # AAPL has a pending order
    signals = dispatcher.dispatch("AAPL", bar, store, {}, open_orders, layer2_enabled=True)
    buy_signals = [s for s in signals if s.signal == "BUY"]
    assert len(buy_signals) == 0, "Should not generate BUY when open order exists for symbol"


def test_dispatcher_allows_exit_even_at_max_positions():
    """EXIT signals are always allowed regardless of position count."""
    from src.engine.bar_dispatcher import BarDispatcher

    # L1 generates EXIT when there's an open position for it
    layers = [
        _make_mock_layer("L1_VWAP_MR", "EXIT"),
        _make_mock_layer("L2_ORB", "HOLD"),
        _make_mock_layer("L3_RSI_SCALP", "HOLD"),
        _make_mock_layer("L4_VOL_SURGE", "HOLD"),
    ]
    dispatcher = BarDispatcher(layers, max_positions=3)
    store = BarStore()
    bar = _make_bar("AAPL")

    # 3 open positions including AAPL L1
    open_positions = {
        "AAPL_L1_VWAP_MR": {"symbol": "AAPL", "entry_price": 99.0, "qty": 5, "layer_name": "L1_VWAP_MR", "stop_price": 98.0, "entry_time": datetime.now(timezone.utc)},
        "MSFT_L3_RSI_SCALP": {"symbol": "MSFT", "entry_price": 300.0, "qty": 1, "layer_name": "L3_RSI_SCALP", "stop_price": 298.0, "entry_time": datetime.now(timezone.utc)},
        "NVDA_L4_VOL_SURGE": {"symbol": "NVDA", "entry_price": 500.0, "qty": 1, "layer_name": "L4_VOL_SURGE", "stop_price": 495.0, "entry_time": datetime.now(timezone.utc)},
    }
    signals = dispatcher.dispatch("AAPL", bar, store, open_positions, set(), layer2_enabled=True)
    exit_signals = [s for s in signals if s.signal == "EXIT"]
    assert len(exit_signals) >= 1, "EXIT signals should be allowed even at max positions"


def test_dispatcher_l2_blocked_when_layer2_disabled():
    """Layer 2 signals are suppressed when layer2_enabled=False (BEAR regime)."""
    from src.engine.bar_dispatcher import BarDispatcher
    store = BarStore()
    symbol = "SPY"
    # Create an opening range so L2 could theoretically fire
    base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    for i in range(35):
        bar_data = {"open": 400.0, "high": 402.0, "low": 399.0, "close": 400.5,
                    "volume": 2000, "symbol": symbol,
                    "timestamp": base_ts + timedelta(minutes=i)}
        store.update(symbol, "1Min", bar_data)
    store.finalize_opening_range(symbol)

    layers = [
        _make_mock_layer("L1_VWAP_MR", "HOLD"),
        _make_mock_layer("L2_ORB", "BUY"),
        _make_mock_layer("L3_RSI_SCALP", "HOLD"),
        _make_mock_layer("L4_VOL_SURGE", "HOLD"),
    ]
    dispatcher = BarDispatcher(layers, max_positions=3)
    bar = _make_bar(symbol, close=402.5)
    # layer2_enabled=False → L2 blocked
    signals = dispatcher.dispatch(symbol, bar, store, {}, set(), layer2_enabled=False)
    l2_signals = [s for s in signals if s.layer_name == "L2_ORB"]
    assert len(l2_signals) == 0, "L2 should be blocked when layer2_enabled=False"


def test_dispatcher_l3_cadence_skips_bars():
    """Layer 3 only fires on every 3rd bar (l3_cadence=3)."""
    from src.engine.bar_dispatcher import BarDispatcher

    l3_call_count = [0]
    original_layer = _make_mock_layer("L3_RSI_SCALP", "HOLD")
    original_evaluate = original_layer.evaluate_bar.side_effect

    def counting_evaluate(symbol, bar, bar_store, open_position=None):
        l3_call_count[0] += 1
        return original_evaluate(symbol, bar, bar_store, open_position)

    original_layer.evaluate_bar.side_effect = counting_evaluate

    layers = [
        _make_mock_layer("L1_VWAP_MR", "HOLD"),
        _make_mock_layer("L2_ORB", "HOLD"),
        original_layer,
        _make_mock_layer("L4_VOL_SURGE", "HOLD"),
    ]
    dispatcher = BarDispatcher(layers, max_positions=3, l3_cadence=3)
    store = BarStore()

    # Dispatch 6 bars to the same symbol
    for i in range(6):
        bar = _make_bar("AAPL")
        store.update("AAPL", "1Min", bar)
        dispatcher.dispatch("AAPL", bar, store, {}, set(), layer2_enabled=True)

    # With cadence=3, L3 should have been called at bars 0, 3 (every 3rd)
    # i.e., 2 out of 6 dispatches
    assert l3_call_count[0] <= 3, f"L3 called too many times: {l3_call_count[0]} (expected <= 3 for 6 bars with cadence 3)"


def test_get_open_position_count():
    """get_open_position_count returns correct count of unique positions."""
    from src.engine.bar_dispatcher import BarDispatcher
    layers = [_make_mock_layer("L1_VWAP_MR", "HOLD")]
    dispatcher = BarDispatcher(layers)

    open_positions = {
        "AAPL_L1_VWAP_MR": {"symbol": "AAPL"},
        "MSFT_L2_ORB": {"symbol": "MSFT"},
        "TSLA_L3_RSI_SCALP": {"symbol": "TSLA"},
    }
    count = dispatcher.get_open_position_count(open_positions)
    assert count == 3
