"""Non-blocking async notification queue with background worker.

Decouples the trading loop from Discord delivery. The trading thread
enqueues notifications; a background daemon thread drains the queue
and sends them via DiscordClient.
"""

import threading
import time
from queue import Full, Queue
from typing import Dict, Optional

from src.utils.logger import get_logger
from src.notifications.discord_client import DiscordClient
from src.notifications.embed_builder import (
    build_alert_embed,
    build_daily_summary_embed,
    build_signal_embed,
    build_trade_entry_embed,
    build_trade_exit_embed,
)

logger = get_logger("notifications.queue")


class NotificationQueue:
    """Thread-safe notification queue with a background consumer.

    The trading loop calls enqueue_* methods which are non-blocking.
    A daemon thread reads from the queue and calls DiscordClient.

    Attributes:
        client: DiscordClient instance for sending webhooks.
        queue: Thread-safe FIFO queue of (channel, embed) tuples.
        max_size: Maximum queue depth.
        _worker: Background daemon thread.
        _stop_event: Event to signal worker shutdown.
        _signal_throttle: Tracks last send time per (symbol, signal_type).
        signal_throttle_seconds: Minimum seconds between identical signals.
        stats: Counters for sent/failed/dropped notifications.
    """

    def __init__(
        self,
        client: Optional[DiscordClient] = None,
        max_size: int = 500,
        signal_throttle_seconds: int = 300,
        rate_limit_per_minute: int = 25,
    ):
        """Initialize the notification queue and start the background worker.

        Args:
            client: DiscordClient instance. Created with defaults if None.
            max_size: Maximum queue size. Default 500.
            signal_throttle_seconds: Throttle window for same-symbol signals. Default 300.
            rate_limit_per_minute: Rate limit for Discord client. Default 25.
        """
        self.client = client or DiscordClient(rate_limit_per_minute=rate_limit_per_minute)
        self.queue: Queue = Queue(maxsize=max_size)
        self.max_size = max_size
        self._stop_event = threading.Event()
        self._signal_throttle: Dict[str, float] = {}
        self.signal_throttle_seconds = signal_throttle_seconds
        self.stats = {"sent": 0, "failed": 0, "dropped": 0, "throttled": 0}

        self._worker = threading.Thread(target=self._consumer_loop, daemon=True)
        self._worker.start()
        logger.info(
            "Notification queue started (max_size=%d, throttle=%ds)",
            max_size,
            signal_throttle_seconds,
        )

    def _consumer_loop(self) -> None:
        """Background loop that drains the queue and sends notifications."""
        while not self._stop_event.is_set():
            try:
                if self.queue.empty():
                    time.sleep(0.1)
                    continue

                channel, embed = self.queue.get(timeout=1.0)
                success = self.client.send_embed(channel, embed)

                if success:
                    self.stats["sent"] += 1
                else:
                    self.stats["failed"] += 1

                self.queue.task_done()
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error("Notification consumer error: %s", str(e))
                    time.sleep(1.0)

    def _enqueue(self, channel: str, embed: dict) -> bool:
        """Add a notification to the queue.

        Args:
            channel: Target Discord channel name.
            embed: Embed dictionary to send.

        Returns:
            bool: True if enqueued, False if queue is full (dropped).
        """
        try:
            self.queue.put_nowait((channel, embed))
            return True
        except Full:
            self.stats["dropped"] += 1
            logger.warning(
                "Notification queue full (%d items), dropping %s notification",
                self.max_size,
                channel,
            )
            return False

    def _is_throttled(self, symbol: str, signal_type: str) -> bool:
        """Check if a signal for this symbol is within the throttle window.

        Args:
            symbol: Ticker symbol.
            signal_type: Signal type (BUY, SELL, etc).

        Returns:
            bool: True if the signal should be throttled.
        """
        key = f"{symbol}:{signal_type}"
        now = time.time()
        last_sent = self._signal_throttle.get(key, 0)
        if now - last_sent < self.signal_throttle_seconds:
            return True
        self._signal_throttle[key] = now
        return False

    def enqueue_trade_entry(
        self, trade_data: dict, prediction: Optional[dict] = None
    ) -> bool:
        """Enqueue a trade entry notification.

        Args:
            trade_data: Trade entry data for embed_builder.
            prediction: Optional prediction data.

        Returns:
            bool: True if enqueued successfully.
        """
        embed = build_trade_entry_embed(trade_data, prediction)
        return self._enqueue("trades", embed)

    def enqueue_trade_exit(self, trade_data: dict) -> bool:
        """Enqueue a trade exit notification.

        Args:
            trade_data: Trade exit data for embed_builder.

        Returns:
            bool: True if enqueued successfully.
        """
        embed = build_trade_exit_embed(trade_data)
        return self._enqueue("trades", embed)

    def enqueue_signal(
        self,
        signal_data: dict,
        prediction: Optional[dict] = None,
        portfolio_state: Optional[dict] = None,
    ) -> bool:
        """Enqueue a signal notification with throttling.

        Duplicate signals for the same symbol+type within the throttle
        window are silently dropped.

        Args:
            signal_data: Signal data for embed_builder.
            prediction: Optional prediction data.
            portfolio_state: Optional portfolio state.

        Returns:
            bool: True if enqueued, False if throttled or queue full.
        """
        symbol = signal_data.get("symbol", "")
        signal_type = signal_data.get("signal_type", "")

        if self._is_throttled(symbol, signal_type):
            self.stats["throttled"] += 1
            logger.debug(
                "Signal throttled: %s %s (within %ds window)",
                symbol,
                signal_type,
                self.signal_throttle_seconds,
            )
            return False

        embed = build_signal_embed(signal_data, prediction)
        return self._enqueue("signals", embed)

    def enqueue_alert(
        self, alert_type: str, message: str, data: Optional[dict] = None
    ) -> bool:
        """Enqueue an alert notification (never throttled).

        Args:
            alert_type: Alert type key for embed_builder.
            message: Alert message text.
            data: Optional additional alert data.

        Returns:
            bool: True if enqueued successfully.
        """
        embed = build_alert_embed(alert_type, message, data)
        return self._enqueue("alerts", embed)

    def enqueue_daily_summary(
        self, portfolio_state: dict, trades_today: list, metrics: dict
    ) -> bool:
        """Enqueue a daily summary notification.

        Args:
            portfolio_state: Portfolio state for embed_builder.
            trades_today: List of trades closed today.
            metrics: Performance metrics dictionary.

        Returns:
            bool: True if enqueued successfully.
        """
        embed = build_daily_summary_embed(portfolio_state, trades_today, metrics)
        return self._enqueue("daily", embed)

    def get_stats(self) -> dict:
        """Return queue statistics.

        Returns:
            dict: Counters for sent, failed, dropped, throttled, pending.
        """
        return {
            **self.stats,
            "pending": self.queue.qsize(),
        }

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the background worker and drain remaining items.

        Args:
            timeout: Maximum seconds to wait for queue drain. Default 10.
        """
        logger.info(
            "Shutting down notification queue (%d items pending)",
            self.queue.qsize(),
        )
        self._stop_event.set()
        self._worker.join(timeout=timeout)
        stats = self.get_stats()
        logger.info(
            "Notification queue stopped: sent=%d, failed=%d, dropped=%d, throttled=%d",
            stats["sent"],
            stats["failed"],
            stats["dropped"],
            stats["throttled"],
        )
