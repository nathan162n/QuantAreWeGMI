"""Logging configuration with rotating file and console handlers.

Provides a centralized logger factory that ensures consistent formatting,
rotation, and log levels across all modules in the trading system.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


_CONFIGURED = False


def setup_logger(
    name: str = "algotrader",
    log_level: str = "",
    log_dir: str = "",
) -> logging.Logger:
    """Create and configure a logger with rotating file and console handlers.

    Args:
        name: Logger name. Defaults to 'algotrader'.
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Falls back to LOG_LEVEL env var, then INFO.
        log_dir: Directory for log files. Falls back to LOG_DIR env var, then 'logs/'.

    Returns:
        logging.Logger: Configured logger instance with both file and console handlers.

    Example:
        >>> logger = setup_logger("algotrader.broker")
        >>> logger.info("Connected to Alpaca paper trading")
    """
    global _CONFIGURED

    level_str = log_level or os.environ.get("LOG_LEVEL", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    directory = log_dir or os.environ.get("LOG_DIR", "logs/")
    os.makedirs(directory, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    main_log_path = os.path.join(directory, "algotrader.log")
    file_handler = RotatingFileHandler(
        main_log_path,
        maxBytes=50 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    alert_log_path = os.path.join(directory, "ALERTS.log")
    alert_handler = RotatingFileHandler(
        alert_log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    alert_handler.setLevel(logging.CRITICAL)
    alert_handler.setFormatter(formatter)
    logger.addHandler(alert_handler)

    if not _CONFIGURED:
        _CONFIGURED = True
        logger.info("Logger initialized: level=%s, dir=%s", level_str.upper(), directory)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a child logger under the algotrader namespace.

    Args:
        module_name: Dotted module path (e.g. 'broker.alpaca_client').

    Returns:
        logging.Logger: Child logger that inherits algotrader configuration.

    Example:
        >>> logger = get_logger("risk.circuit_breaker")
        >>> logger.critical("Daily loss limit breached: -3.2%%")
    """
    parent = logging.getLogger("algotrader")
    if not parent.handlers:
        setup_logger()
    return logging.getLogger(f"algotrader.{module_name}")
