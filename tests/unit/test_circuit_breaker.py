"""Tests for circuit breaker pattern."""

import pytest
import asyncio
from loom.resilience import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)


class TestCircuitBreaker:
    """Test circuit breaker state transitions and behavior."""

    @pytest.mark.asyncio
    async def test_starts_in_closed_state(self):
        """Circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))

        assert breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_calls_keep_circuit_closed(self):
        """Successful calls keep circuit in CLOSED state."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))

        async def success_func():
            return "ok"

        # Multiple successful calls
        for _ in range(10):
            result = await breaker.call(success_func)
            assert result == "ok"

        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.stats.successful_calls == 10

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self):
        """Circuit opens after failure threshold exceeded."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))

        async def failing_func():
            raise Exception("Service unavailable")

        # Fail 3 times (threshold)
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        # Circuit should be OPEN now
        assert breaker.get_state() == CircuitState.OPEN
        assert breaker.stats.failed_calls == 3

    @pytest.mark.asyncio
    async def test_rejects_requests_when_open(self):
        """Circuit rejects requests when OPEN."""
        breaker = CircuitBreaker(
            "test", CircuitBreakerConfig(failure_threshold=2, timeout_seconds=10)
        )

        async def failing_func():
            raise Exception("Service unavailable")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        # Subsequent call should be rejected immediately
        async def any_func():
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await breaker.call(any_func)

        assert "Circuit 'test' is OPEN" in str(exc_info.value)
        assert breaker.stats.rejected_calls == 1

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after timeout expires."""
        breaker = CircuitBreaker(
            "test", CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.1)
        )

        async def failing_func():
            raise Exception("Service unavailable")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        # Wait for timeout to expire
        await asyncio.sleep(0.2)

        # Next call should transition to HALF_OPEN
        async def success_func():
            return "ok"

        # First call in half-open should succeed and start counting successes
        result = await breaker.call(success_func)
        assert result == "ok"
        assert breaker.get_state() == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_threshold_in_half_open(self):
        """Circuit closes after success threshold in HALF_OPEN."""
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=2, timeout_seconds=0.1, success_threshold=3
            ),
        )

        async def failing_func():
            raise Exception("Service unavailable")

        async def success_func():
            return "ok"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Succeed 3 times to close circuit
        for i in range(3):
            result = await breaker.call(success_func)
            assert result == "ok"

        # Circuit should be CLOSED now
        assert breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_failure_in_half_open(self):
        """Circuit reopens immediately on failure in HALF_OPEN."""
        breaker = CircuitBreaker(
            "test", CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.1)
        )

        async def failing_func():
            raise Exception("Service unavailable")

        async def success_func():
            return "ok"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        # Wait for timeout to enter half-open
        await asyncio.sleep(0.2)

        # Succeed once
        await breaker.call(success_func)
        assert breaker.get_state() == CircuitState.HALF_OPEN

        # Fail once - should immediately reopen
        with pytest.raises(Exception):
            await breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_limits_concurrent_half_open_requests(self):
        """Circuit limits concurrent requests in HALF_OPEN state."""
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=2,
                timeout_seconds=0.1,
                half_open_max_requests=2,
            ),
        )

        async def failing_func():
            raise Exception("Service unavailable")

        async def slow_func():
            await asyncio.sleep(0.5)
            return "ok"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Start 2 concurrent requests (at limit)
        task1 = asyncio.create_task(breaker.call(slow_func))
        task2 = asyncio.create_task(breaker.call(slow_func))

        # Small delay to let tasks start
        await asyncio.sleep(0.01)

        # Third concurrent request should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(slow_func)

        # Wait for original tasks to complete
        await task1
        await task2

    @pytest.mark.asyncio
    async def test_excluded_exceptions_dont_count_as_failures(self):
        """Excluded exceptions don't count toward failure threshold."""
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=3, excluded_exceptions=(ValueError,)
            ),
        )

        async def validation_error():
            raise ValueError("Invalid input")

        # ValueError should be excluded
        for _ in range(5):
            with pytest.raises(ValueError):
                await breaker.call(validation_error)

        # Circuit should still be CLOSED
        assert breaker.get_state() == CircuitState.CLOSED

        # But non-excluded exceptions should count
        async def service_error():
            raise Exception("Service error")

        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(service_error)

        # Now circuit should be OPEN
        assert breaker.get_state() == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_manual_reset(self):
        """Circuit can be manually reset to CLOSED."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))

        async def failing_func():
            raise Exception("Service unavailable")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        # Manual reset
        await breaker.reset()

        assert breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Circuit breaker tracks statistics correctly."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))

        async def success_func():
            return "ok"

        async def failing_func():
            raise Exception("Error")

        # 5 successes
        for _ in range(5):
            await breaker.call(success_func)

        # 3 failures (opens circuit)
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        # 2 rejections
        for _ in range(2):
            with pytest.raises(CircuitBreakerOpenError):
                await breaker.call(success_func)

        stats = breaker.get_stats()
        assert stats.successful_calls == 5
        assert stats.failed_calls == 3
        assert stats.rejected_calls == 2
        assert stats.total_calls == 8  # 5 + 3
        assert "closed -> open" in stats.state_transitions


class TestCircuitBreakerIntegration:
    """Integration tests with real-world scenarios."""

    @pytest.mark.asyncio
    async def test_llm_api_protection_scenario(self):
        """Circuit breaker protects against LLM API failures."""
        breaker = CircuitBreaker(
            "llm_api",
            CircuitBreakerConfig(
                failure_threshold=5, timeout_seconds=0.5, success_threshold=2
            ),
        )

        call_count = 0

        async def llm_api_call(prompt: str):
            nonlocal call_count
            call_count += 1

            # Simulate API degradation
            if call_count <= 5:
                # First 5 calls fail
                raise Exception("API rate limit exceeded")
            else:
                # API recovered
                return f"Response to: {prompt}"

        # First 5 calls fail and open circuit
        for i in range(5):
            with pytest.raises(Exception):
                await breaker.call(llm_api_call, prompt=f"prompt {i}")

        assert breaker.get_state() == CircuitState.OPEN

        # Next calls are rejected without hitting API
        api_calls_before = call_count
        for i in range(3):
            with pytest.raises(CircuitBreakerOpenError):
                await breaker.call(llm_api_call, prompt=f"prompt {i}")

        # Verify no API calls were made
        assert call_count == api_calls_before

        # Wait for circuit to test recovery
        await asyncio.sleep(0.6)

        # Succeed twice to close circuit
        result1 = await breaker.call(llm_api_call, prompt="test1")
        assert "Response" in result1

        result2 = await breaker.call(llm_api_call, prompt="test2")
        assert "Response" in result2

        # Circuit should be CLOSED
        assert breaker.get_state() == CircuitState.CLOSED
