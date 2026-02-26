"""Discord webhook client with retry, rate limiting, and channel routing.

The only file in the codebase that communicates with Discord.
All embed formatting is delegated to embed_builder.py.
"""

import os
import time
import threading
from typing import Dict, Optional

import requests

from src.utils.logger import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger("notifications.discord_client")

# Channel name -> environment variable mapping
_CHANNEL_ENV_MAP: Dict[str, str] = {
    "trades": "DISCORD_WEBHOOK_TRADES",
    "signals": "DISCORD_WEBHOOK_SIGNALS",
    "alerts": "DISCORD_WEBHOOK_ALERTS",
    "daily": "DISCORD_WEBHOOK_DAILY",
}


class _TokenBucket:
    """Simple token bucket rate limiter for Discord webhook calls.

    Attributes:
        rate: Tokens added per second.
        capacity: Maximum tokens in the bucket.
        tokens: Current available tokens.
        last_refill: Timestamp of last refill.
        lock: Thread lock for concurrent access.
    """

    def __init__(self, rate_per_minute: int = 25):
        """Initialize the token bucket.

        Args:
            rate_per_minute: Maximum requests per minute. Default 25.
        """
        self.rate = rate_per_minute / 60.0
        self.capacity = rate_per_minute
        self.tokens = float(rate_per_minute)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """Block until a token is available or timeout.

        Args:
            timeout: Maximum seconds to wait. Default 30.

        Returns:
            bool: True if a token was acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_refill = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


class DiscordClient:
    """Discord webhook sender with per-channel routing and rate limiting.

    Loads webhook URLs from environment variables on init. If a channel
    URL is missing, messages to that channel are silently skipped with
    a warning logged once.

    Attributes:
        webhooks: Mapping of channel name to webhook URL.
        rate_limiter: Token bucket for rate limiting.
        _warned_channels: Set of channels already warned about missing URLs.
    """

    def __init__(self, rate_limit_per_minute: int = 25):
        """Initialize the Discord client.

        Args:
            rate_limit_per_minute: Maximum webhook calls per minute. Default 25.
        """
        self.webhooks: Dict[str, str] = {}
        self._warned_channels: set = set()
        self.rate_limiter = _TokenBucket(rate_per_minute=rate_limit_per_minute)

        for channel, env_var in _CHANNEL_ENV_MAP.items():
            url = os.environ.get(env_var, "")
            if url:
                self.webhooks[channel] = url
                logger.info("Discord channel '%s' configured", channel)
            else:
                logger.warning(
                    "Discord channel '%s' has no webhook URL (env: %s)",
                    channel,
                    env_var,
                )

    def is_channel_configured(self, channel: str) -> bool:
        """Check if a channel has a configured webhook URL.

        Args:
            channel: Channel name (trades, signals, alerts, daily).

        Returns:
            bool: True if the channel's webhook URL is set.
        """
        return channel in self.webhooks

    @retry_with_backoff(
        max_retries=3,
        base_delay=2.0,
        max_delay=30.0,
        retryable_exceptions=(requests.RequestException, ConnectionError, TimeoutError),
    )
    def _send_webhook(self, url: str, payload: dict) -> bool:
        """Send a payload to a Discord webhook URL.

        Args:
            url: Discord webhook URL.
            payload: JSON payload to send.

        Returns:
            bool: True if the webhook responded with 2xx.

        Raises:
            requests.RequestException: On network failure (retried).
        """
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 429:
            retry_after = resp.json().get("retry_after", 5.0)
            logger.warning("Discord rate limited, waiting %.1fs", retry_after)
            time.sleep(retry_after)
            resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True

    def send_embed(self, channel: str, embed: dict, content: str = "") -> bool:
        """Send an embed to a Discord channel via webhook.

        Args:
            channel: Target channel name (trades, signals, alerts, daily).
            embed: Discord embed dictionary (from embed_builder).
            content: Optional text content above the embed.

        Returns:
            bool: True if sent successfully, False if skipped or failed.
        """
        url = self.webhooks.get(channel)
        if not url:
            if channel not in self._warned_channels:
                logger.warning(
                    "Skipping Discord notification: channel '%s' not configured",
                    channel,
                )
                self._warned_channels.add(channel)
            return False

        if not self.rate_limiter.acquire(timeout=30.0):
            logger.warning("Discord rate limit timeout for channel '%s'", channel)
            return False

        payload: dict = {"embeds": [embed]}
        if content:
            payload["content"] = content

        try:
            self._send_webhook(url, payload)
            logger.debug("Discord notification sent to '%s'", channel)
            return True
        except Exception as e:
            logger.error(
                "Failed to send Discord notification to '%s': %s",
                channel,
                str(e),
            )
            return False

    def send_trade_entry(self, embed: dict) -> bool:
        """Send a trade entry notification to the trades channel.

        Args:
            embed: Trade entry embed from embed_builder.

        Returns:
            bool: True if sent successfully.
        """
        return self.send_embed("trades", embed)

    def send_trade_exit(self, embed: dict) -> bool:
        """Send a trade exit notification to the trades channel.

        Args:
            embed: Trade exit embed from embed_builder.

        Returns:
            bool: True if sent successfully.
        """
        return self.send_embed("trades", embed)

    def send_signal(self, embed: dict) -> bool:
        """Send a signal notification to the signals channel.

        Args:
            embed: Signal embed from embed_builder.

        Returns:
            bool: True if sent successfully.
        """
        return self.send_embed("signals", embed)

    def send_alert(self, embed: dict) -> bool:
        """Send an alert notification to the alerts channel.

        Args:
            embed: Alert embed from embed_builder.

        Returns:
            bool: True if sent successfully.
        """
        return self.send_embed("alerts", embed)

    def send_daily_summary(self, embed: dict) -> bool:
        """Send a daily summary to the daily channel.

        Args:
            embed: Daily summary embed from embed_builder.

        Returns:
            bool: True if sent successfully.
        """
        return self.send_embed("daily", embed)
