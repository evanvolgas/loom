"""Circuit breaker pattern implementation.

Prevents cascading failures by stopping calls to failing upstream services.

Based on Michael Nygard's "Release It!" patterns.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    """Number of consecutive failures before opening circuit."""

    success_threshold: int = 3
    """Number of consecutive successes in half-open to close circuit."""

    timeout_seconds: int = 60
    """Seconds to wait before transitioning from open to half-open."""

    half_open_max_requests: int = 3
    """Maximum concurrent requests allowed in half-open state."""

    excluded_exceptions: tuple[type[Exception], ...] = ()
    """Exception types that don't count as failures (e.g., validation errors)."""


class CircuitBreakerOpenError(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, circuit_name: str, open_until: float):
        self.circuit_name = circuit_name
        self.open_until = open_until
        remaining = max(0, open_until - time.time())
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. "
            f"Retry after {remaining:.1f}s at {time.strftime('%H:%M:%S', time.localtime(open_until))}"
        )


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: dict[str, int] = field(default_factory=dict)

    def record_success(self) -> None:
        """Record successful call."""
        self.total_calls += 1
        self.successful_calls += 1

    def record_failure(self) -> None:
        """Record failed call."""
        self.total_calls += 1
        self.failed_calls += 1

    def record_rejection(self) -> None:
        """Record rejected call (circuit open)."""
        self.rejected_calls += 1

    def record_state_change(self, from_state: CircuitState, to_state: CircuitState) -> None:
        """Record state transition."""
        key = f"{from_state.value} -> {to_state.value}"
        self.state_transitions[key] = self.state_transitions.get(key, 0) + 1


class CircuitBreaker:
    """Circuit breaker for protecting upstream services from cascading failures.

    Example:
        ```python
        # Create circuit breaker for LLM API
        llm_breaker = CircuitBreaker(
            name="arbiter_llm",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=60
            )
        )

        # Use circuit breaker
        try:
            result = await llm_breaker.call(llm_client.generate, prompt=prompt)
        except CircuitBreakerOpenError:
            # Circuit open, use fallback
            result = get_cached_response(prompt)
        ```
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker identifier for logging/monitoring
            config: Configuration, or use defaults
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.open_until: Optional[float] = None
        self.half_open_requests = 0
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"timeout_seconds={self.config.timeout_seconds}"
        )

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result from function

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        # Check state and potentially reject
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self.open_until is not None and time.time() < self.open_until:
                    # Still in timeout period, reject request
                    self.stats.record_rejection()
                    raise CircuitBreakerOpenError(self.name, self.open_until)
                else:
                    # Timeout expired, transition to half-open
                    await self._transition_to(CircuitState.HALF_OPEN)

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.half_open_max_requests:
                    # Too many concurrent half-open requests
                    self.stats.record_rejection()
                    raise CircuitBreakerOpenError(
                        self.name, time.time() + self.config.timeout_seconds
                    )
                self.half_open_requests += 1

        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result

        except Exception as e:
            # Check if exception should be excluded
            if isinstance(e, self.config.excluded_exceptions):
                logger.debug(
                    f"Circuit '{self.name}': Excluded exception {type(e).__name__}"
                )
                raise  # Don't count as failure

            await self._on_failure(e)
            raise

        finally:
            if self.state == CircuitState.HALF_OPEN:
                async with self._lock:
                    self.half_open_requests -= 1

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self.stats.record_success()

            if self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

            elif self.state == CircuitState.HALF_OPEN:
                # Count successes in half-open state
                self.success_count += 1

                logger.info(
                    f"Circuit '{self.name}' half-open success: "
                    f"{self.success_count}/{self.config.success_threshold}"
                )

                if self.success_count >= self.config.success_threshold:
                    # Enough successes, close circuit
                    await self._transition_to(CircuitState.CLOSED)

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self.stats.record_failure()

            logger.warning(
                f"Circuit '{self.name}' failure in {self.state.value} state: "
                f"{type(exception).__name__}: {exception}"
            )

            if self.state == CircuitState.CLOSED:
                self.failure_count += 1

                if self.failure_count >= self.config.failure_threshold:
                    # Threshold exceeded, open circuit
                    await self._transition_to(CircuitState.OPEN)

            elif self.state == CircuitState.HALF_OPEN:
                # Failure in half-open, immediately reopen circuit
                logger.warning(
                    f"Circuit '{self.name}': Failure in half-open state, reopening circuit"
                )
                await self._transition_to(CircuitState.OPEN)

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state.

        Args:
            new_state: Target state
        """
        old_state = self.state
        self.state = new_state
        self.stats.record_state_change(old_state, new_state)

        if new_state == CircuitState.OPEN:
            self.open_until = time.time() + self.config.timeout_seconds
            self.failure_count = 0

            logger.error(
                f"Circuit '{self.name}' OPENED: timeout={self.config.timeout_seconds}s, "
                f"retry at {time.strftime('%H:%M:%S', time.localtime(self.open_until))}"
            )

        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
            self.half_open_requests = 0

            logger.info(
                f"Circuit '{self.name}' HALF-OPEN: Testing if service recovered"
            )

        elif new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
            self.open_until = None

            logger.info(
                f"Circuit '{self.name}' CLOSED: Service recovered, resuming normal operation"
            )

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self.stats

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            logger.info(
                f"Circuit '{self.name}' manually reset to CLOSED state"
            )
            await self._transition_to(CircuitState.CLOSED)


# Global circuit breaker registry for sharing across components
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create circuit breaker.

    Args:
        name: Circuit breaker identifier
        config: Configuration for new circuit breaker

    Returns:
        Existing or new circuit breaker
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all registered circuit breakers.

    Returns:
        Dictionary of circuit breakers by name
    """
    return _circuit_breakers.copy()
