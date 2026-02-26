"""Shared pytest fixtures for the algorithmic trading test suite.

Provides mock clients, database sessions, and common test data
used across all test modules.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")
os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
os.environ.setdefault("TRADING_MODE", "paper")
os.environ.setdefault("LOG_LEVEL", "WARNING")
# Prevent DB connections during tests
os.environ.setdefault("SQLITE_FALLBACK_URL", "sqlite:///tests/test.db")


@pytest.fixture
def mock_config():
    """Provide a complete test configuration dictionary.

    Returns:
        dict: Full configuration matching the new config.yml structure.

    Example:
        >>> def test_something(mock_config):
        ...     assert mock_config["trading"]["max_positions"] == 3
    """
    return {
        "trading": {
            "mode": "paper",
            "max_positions": 3,
            "max_position_pct": 0.25,
            "allow_shorts": False,
            "starting_capital": 500.0,
            "dollar_risk_per_trade": 8.00,
        },
        "universe": {
            "base_list_path": "data/universe_largecap.csv",
            "min_price": 15.0,
            "min_adv_dollars": 50_000_000,
            "max_realized_vol": 1.50,
            "max_spread_pct": 0.0005,
            "earnings_blackout_days": 3,
            "min_universe_size": 10,
        },
        "strategy": {
            "l1_vwap_deviation_entry": 0.005,
            "l1_rsi_period": 14,
            "l1_rsi_entry": 42,
            "l1_rsi_exit": 62,
            "l1_max_hold_minutes": 180,
            "l1_stop_atr_multiplier": 1.0,
            "l2_range_computation_end_minutes": 30,
            "l2_breakout_buffer_pct": 0.001,
            "l2_volume_confirmation_ratio": 1.5,
            "l2_bar_position_threshold": 0.70,
            "l2_extension_target_multiplier": 2.0,
            "l3_rsi_period": 7,
            "l3_rsi_oversold": 25,
            "l3_rsi_recovery_exit": 55,
            "l3_max_hold_bars": 10,
            "l3_stop_atr_multiplier": 0.5,
            "l3_bar_cadence": 3,
            "l4_volume_surge_ratio": 3.0,
            "l4_bar_position_threshold": 0.80,
            "l4_rsi_exit_overbought": 75,
            "l4_max_hold_minutes": 120,
            "signal_confidence_notify_threshold": 0.35,
            "eod_hard_close_time": "15:30",
        },
        "risk": {
            "dollar_risk_per_trade": 8.00,
            "max_positions": 3,
            "max_position_pct": 0.25,
            "atr_period": 14,
            "daily_loss_limit": 0.03,
            "weekly_loss_limit": 0.08,
            "max_drawdown_limit": 0.15,
            "consecutive_loss_days": 3,
            "max_day_trades_per_week": 3,
        },
        "regime": {
            "spy_sma_period": 200,
            "vix_proxy_symbol": "VIXY",
            "vix_size_reduction_threshold": 40,
            "vix_size_reduction_scalar": 0.50,
            "vix_halt_threshold": 60,
        },
        "prediction": {
            "bars_horizon": 5,
            "min_confidence_to_display": 0.25,
        },
        "notifications": {
            "signal_throttle_seconds": 120,
            "queue_max_size": 1000,
            "discord_rate_limit_per_minute": 25,
        },
        "reporting": {
            "risk_free_rate": 0.05,
            "starting_capital": 500.0,
        },
        "backtest": {
            "default_slippage_pct": 0.0005,
            "default_lookback_days": 365,
            "target_trades_per_year": 300,
        },
        "scheduler": {
            "timezone": "America/New_York",
            "premarket_run_hour": 8,
            "premarket_run_minute": 0,
            "opening_range_close_hour": 10,
            "opening_range_close_minute": 0,
            "eod_close_hour": 15,
            "eod_close_minute": 30,
            "post_market_summary_hour": 16,
            "post_market_summary_minute": 30,
        },
        "database": {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_recycle_seconds": 1800,
            "connection_timeout": 10,
        },
        "bar_store": {
            "max_bars_per_symbol_per_timeframe": 200,
        },
    }


@pytest.fixture
def bar_store():
    """Provide a fresh BarStore instance.

    Returns:
        BarStore: Empty bar store for testing.

    Example:
        >>> def test_something(bar_store):
        ...     bar_store.update("AAPL", "1Min", {...})
    """
    from src.data.bar_store import BarStore, reset_bar_store
    reset_bar_store()
    return BarStore()


@pytest.fixture
def today():
    """Provide today's UTC datetime at market open.

    Returns:
        datetime: Today at 14:30 UTC (9:30 ET).

    Example:
        >>> def test_something(today):
        ...     assert today.hour == 14
    """
    now = datetime.now(timezone.utc)
    return now.replace(hour=14, minute=30, second=0, microsecond=0)


@pytest.fixture
def yesterday():
    """Provide yesterday's UTC datetime at market open.

    Returns:
        datetime: Yesterday at 14:30 UTC.

    Example:
        >>> def test_something(yesterday):
        ...     assert (datetime.now(timezone.utc) - yesterday).days >= 1
    """
    now = datetime.now(timezone.utc) - timedelta(days=1)
    return now.replace(hour=14, minute=30, second=0, microsecond=0)


@pytest.fixture
def sample_bar():
    """Provide a sample 1-minute bar dict.

    Returns:
        dict: A bar dict with OHLCV and timestamp.

    Example:
        >>> def test_something(sample_bar):
        ...     assert sample_bar["close"] == 150.0
    """
    return {
        "open": 149.5,
        "high": 151.0,
        "low": 149.0,
        "close": 150.0,
        "volume": 10000,
        "symbol": "AAPL",
        "timestamp": datetime.now(timezone.utc),
    }
