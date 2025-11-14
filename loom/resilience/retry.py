"""Retry strategy with exponential backoff.

Already exists in IMPLEMENTATION_SPEC but reproduced here for completeness.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    """Error that can be retried."""

    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    exponential_base: float = 2.0
    max_delay: float = 60.0
    retryable_exceptions: tuple[type[Exception], ...] = (RetryableError, asyncio.TimeoutError)


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    config: RetryConfig,
    *args: Any,
    **kwargs: Any
) -> T:
    """Execute function with exponential backoff retry.

    Args:
        func: Async function to execute
        config: Retry configuration
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function

    Returns:
        Result from function

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt < config.max_attempts - 1:
                delay = min(
                    config.initial_delay * (config.exponential_base**attempt),
                    config.max_delay,
                )
                await asyncio.sleep(delay)
            else:
                raise last_exception

    # This should never be reached, but mypy needs it
    raise RuntimeError("Retry loop completed without returning or raising")
