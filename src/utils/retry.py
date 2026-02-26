"""Exponential backoff retry decorator with jitter.

Provides a configurable retry mechanism for network operations,
API calls, and database connections that may transiently fail.
"""

import asyncio
import functools
import random
import time
import traceback
from typing import Any, Callable, Tuple, Type

from src.utils.logger import get_logger

logger = get_logger("utils.retry")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator that retries a function with exponential backoff on failure.

    Uses exponential backoff with optional jitter to avoid thundering herd.
    Delay formula: min(base_delay * exponential_base^attempt + jitter, max_delay)

    Args:
        max_retries: Maximum number of retry attempts. Default 3.
        base_delay: Initial delay in seconds. Default 1.0.
        max_delay: Maximum delay cap in seconds. Default 60.0.
        exponential_base: Multiplier for exponential growth. Default 2.0.
        jitter: Whether to add random jitter (0 to 1s). Default True.
        retryable_exceptions: Tuple of exception types to retry on.
            Default (Exception,) retries all exceptions.

    Returns:
        Callable: Decorated function with retry logic.

    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
        ... def fetch_data():
        ...     return requests.get("https://api.example.com/data")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Synchronous retry wrapper.

            Args:
                *args: Positional arguments passed to the wrapped function.
                **kwargs: Keyword arguments passed to the wrapped function.

            Returns:
                Any: Return value of the wrapped function on success.
            """
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            "All %d retries exhausted for %s: %s\n%s",
                            max_retries,
                            func.__name__,
                            str(e),
                            traceback.format_exc(),
                        )
                        raise
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay += random.uniform(0, 1.0)
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        func.__name__,
                        delay,
                        str(e),
                    )
                    time.sleep(delay)
            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Asynchronous retry wrapper.

            Args:
                *args: Positional arguments passed to the wrapped function.
                **kwargs: Keyword arguments passed to the wrapped function.

            Returns:
                Any: Return value of the wrapped function on success.
            """
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            "All %d retries exhausted for async %s: %s\n%s",
                            max_retries,
                            func.__name__,
                            str(e),
                            traceback.format_exc(),
                        )
                        raise
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay += random.uniform(0, 1.0)
                    logger.warning(
                        "Retry %d/%d for async %s after %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        func.__name__,
                        delay,
                        str(e),
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # type: ignore[misc]

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
