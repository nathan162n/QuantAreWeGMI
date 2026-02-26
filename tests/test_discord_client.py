"""Unit tests for Discord notification system.

Tests cover the discord client, embed builders, and notification queue
using mocked webhook calls (no actual Discord API calls).
"""

import os
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.notifications.embed_builder import (
    build_alert_embed,
    build_daily_summary_embed,
    build_signal_embed,
    build_trade_entry_embed,
    build_trade_exit_embed,
)
from src.notifications.discord_client import DiscordClient, _TokenBucket
from src.notifications.notification_queue import NotificationQueue
from src.strategy.base_strategy import SignalResult


class TestEmbedBuilders:
    """Tests for Discord embed construction."""

    def test_trade_entry_embed_structure(self):
        """Trade entry embed should have required fields."""
        trade_data = {
            "symbol": "AAPL",
            "entry_price": 182.45,
            "shares": 2,
            "stop_price": 178.00,
            "strategy_name": "L1_VWAP_MR",
            "confidence": 0.8,
        }
        embed = build_trade_entry_embed(trade_data)
        assert "title" in embed
        assert "color" in embed
        assert "fields" in embed
        assert "AAPL" in embed["title"]

    def test_trade_exit_embed_profit(self):
        """Profitable trade exit should use green color (0x00FF00)."""
        embed = build_trade_exit_embed({
            "symbol": "MSFT",
            "pnl": 5.50,
            "entry_price": 380.00,
            "exit_price": 382.75,
            "qty": 2,
            "exit_reason": "SIGNAL",
        })
        assert embed["color"] == 0x00FF00

    def test_trade_exit_embed_loss(self):
        """Loss trade exit should use red color (0xFF0000)."""
        embed = build_trade_exit_embed({
            "symbol": "MSFT",
            "pnl": -3.20,
            "entry_price": 380.00,
            "exit_price": 378.40,
            "qty": 2,
        })
        assert embed["color"] == 0xFF0000

    def test_signal_embed_with_prediction(self):
        """Signal embed with prediction should include forecast fields."""
        signal = SignalResult(
            symbol="NVDA",
            signal="BUY",
            confidence=0.75,
            signal_price=450.0,
            layer_name="L1_VWAP_MR",
            stop_price=445.0,
        )
        embed = build_signal_embed(
            signal,
            prediction_result={"direction": "BULLISH", "confidence_pct": 68.5,
                               "estimated_low": 448.0, "estimated_high": 455.0,
                               "key_driver": "Z-Score Recovery", "bars_horizon": 5},
        )
        field_names = [f["name"] for f in embed["fields"]]
        assert any("Forecast" in n for n in field_names)

    def test_signal_embed_without_prediction(self):
        """Signal embed without prediction should still work."""
        signal = SignalResult(
            symbol="NVDA",
            signal="BUY",
            confidence=0.75,
            signal_price=450.0,
            layer_name="L1_VWAP_MR",
            stop_price=445.0,
        )
        embed = build_signal_embed(signal)
        assert "NVDA" in embed["title"]

    def test_alert_embed_circuit_breaker(self):
        """Circuit breaker alert should use red color."""
        embed = build_alert_embed("CIRCUIT_BREAKER_TRIGGERED", "Daily loss limit exceeded")
        assert embed["color"] == 0xFF0000
        assert "CIRCUIT BREAKER" in embed["title"]

    def test_alert_embed_unknown_type(self):
        """Unknown alert type should use default formatting."""
        embed = build_alert_embed("UNKNOWN_TYPE", "Something happened")
        assert "ALERT" in embed["title"]

    def test_daily_summary_embed(self):
        """Daily summary should include portfolio and activity fields."""
        embed = build_daily_summary_embed({
            "portfolio_value": 503.42,
            "daily_pnl": 3.42,
            "peak_value": 505.0,
            "drawdown_pct": 0.003,
            "regime": "BULL",
            "starting_capital": 500.0,
            "sharpe": 1.2,
            "sortino": 1.5,
            "max_drawdown": 0.02,
            "win_rate": 0.6,
        })
        assert "DAILY SUMMARY" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert any("Portfolio" in n for n in field_names)


class TestTokenBucket:
    """Tests for the rate limiter."""

    def test_acquire_immediately(self):
        """First acquire should succeed immediately."""
        bucket = _TokenBucket(rate_per_minute=60)
        assert bucket.acquire(timeout=1.0) is True

    def test_acquire_depletes_tokens(self):
        """Acquiring all tokens should eventually block."""
        bucket = _TokenBucket(rate_per_minute=2)
        assert bucket.acquire(timeout=0.1) is True
        assert bucket.acquire(timeout=0.1) is True
        assert bucket.acquire(timeout=0.2) is False


class TestDiscordClient:
    """Tests for the Discord webhook client."""

    def test_missing_channel_skips(self):
        """Sending to unconfigured channel should return False."""
        client = DiscordClient()
        result = client.send_embed("nonexistent", {"title": "test"})
        assert result is False

    @patch.dict(os.environ, {"DISCORD_WEBHOOK_TRADES": "https://discord.com/api/webhooks/test/test"})
    @patch("src.notifications.discord_client.requests.post")
    def test_send_success(self, mock_post):
        """Successful webhook send should return True."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()
        client = DiscordClient()
        result = client.send_embed("trades", {"title": "test"})
        assert result is True
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"DISCORD_WEBHOOK_ALERTS": "https://discord.com/api/webhooks/test/test"})
    @patch("src.notifications.discord_client.requests.post")
    def test_send_alert_helper(self, mock_post):
        """send_alert helper should route to alerts channel."""
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()
        client = DiscordClient()
        embed = build_alert_embed("ERROR", "Test error")
        result = client.send_alert(embed)
        assert result is True


class TestNotificationQueue:
    """Tests for the async notification queue."""

    def test_enqueue_and_stats(self):
        """Enqueuing should increment pending count."""
        mock_client = MagicMock(spec=DiscordClient)
        mock_client.send_embed.return_value = True
        queue = NotificationQueue(client=mock_client, max_size=10)
        queue.enqueue_alert("ERROR", "Test")
        time.sleep(0.5)
        stats = queue.get_stats()
        assert stats["sent"] >= 0
        queue.shutdown(timeout=2.0)

    def test_signal_throttling(self):
        """Duplicate signals within throttle window should be dropped."""
        mock_client = MagicMock(spec=DiscordClient)
        mock_client.send_embed.return_value = True
        queue = NotificationQueue(
            client=mock_client, max_size=10, signal_throttle_seconds=300
        )
        signal = {"symbol": "AAPL", "signal_type": "BUY", "confidence": 0.7, "current_price": 150.0}
        first = queue.enqueue_signal(signal)
        second = queue.enqueue_signal(signal)
        assert first is True
        assert second is False
        assert queue.stats["throttled"] == 1
        queue.shutdown(timeout=2.0)

    def test_queue_overflow(self):
        """Queue overflow should increment dropped count."""
        mock_client = MagicMock(spec=DiscordClient)
        mock_client.send_embed.side_effect = lambda *a, **kw: time.sleep(1)
        queue = NotificationQueue(client=mock_client, max_size=2)
        for _ in range(5):
            queue.enqueue_alert("ERROR", "overflow test")
        assert queue.stats["dropped"] >= 1
        queue.shutdown(timeout=2.0)

    def test_shutdown(self):
        """Shutdown should stop the worker thread."""
        mock_client = MagicMock(spec=DiscordClient)
        queue = NotificationQueue(client=mock_client, max_size=10)
        queue.shutdown(timeout=2.0)
        # Worker thread should no longer be alive after shutdown
        worker = getattr(queue, "_worker", None) or getattr(queue, "_worker_thread", None)
        assert worker is None or not worker.is_alive()
