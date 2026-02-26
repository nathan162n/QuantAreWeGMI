"""Discord notification system for the algorithmic trading platform.

Provides non-blocking Discord webhook notifications with retry logic,
rich embed formatting, and an async queue that decouples trading from
notification delivery.
"""
