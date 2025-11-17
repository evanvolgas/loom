"""Unit tests for retry mechanism."""

import asyncio
import pytest
import time

from loom.resilience.retry import retry_with_backoff, RetryConfig, RetryableError


class TestRetryWithBackoff:
    """Test retry_with_backoff functionality."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = [0]

        async def successful_func():
            call_count[0] += 1
            return "success"

        config = RetryConfig(max_attempts=3)
        result = await retry_with_backoff(successful_func, config)

        assert result == "success"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """Test retry with eventual success."""
        call_count = [0]

        async def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RetryableError("Temporary failure")
            return "success"

        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        result = await retry_with_backoff(flaky_func, config)

        assert result == "success"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test all retries exhausted raises exception."""
        call_count = [0]

        async def always_fails():
            call_count[0] += 1
            raise RetryableError("Persistent failure")

        config = RetryConfig(max_attempts=3, initial_delay=0.01)

        with pytest.raises(RetryableError) as exc_info:
            await retry_with_backoff(always_fails, config)

        assert "Persistent failure" in str(exc_info.value)
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff delays."""
        call_times = []

        async def flaky_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise RetryableError("Retry me")
            return "success"

        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.1,
            exponential_base=2.0,
        )

        start = time.time()
        result = await retry_with_backoff(flaky_func, config)
        elapsed = time.time() - start

        assert result == "success"
        assert len(call_times) == 3

        # Should have delays of ~0.1s, ~0.2s between attempts
        # Total should be at least 0.3s but less than 0.5s
        assert elapsed >= 0.3
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        call_count = [0]

        async def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RetryableError("Retry me")
            return "success"

        config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            exponential_base=10.0,  # Would be huge without cap
            max_delay=0.1,  # Cap at 0.1s
        )

        start = time.time()
        result = await retry_with_backoff(flaky_func, config)
        elapsed = time.time() - start

        assert result == "success"
        # Even with exponential_base=10, should only take ~0.2s total (2 * 0.1s)
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_timeout_error_is_retryable(self):
        """Test that TimeoutError is retryable by default."""
        call_count = [0]

        async def timeout_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise asyncio.TimeoutError("Timeout")
            return "success"

        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        result = await retry_with_backoff(timeout_func, config)

        assert result == "success"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately."""
        call_count = [0]

        async def bad_func():
            call_count[0] += 1
            raise ValueError("Not retryable")

        config = RetryConfig(max_attempts=3)

        with pytest.raises(ValueError) as exc_info:
            await retry_with_backoff(bad_func, config)

        assert "Not retryable" in str(exc_info.value)
        assert call_count[0] == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self):
        """Test custom retryable exception configuration."""
        call_count = [0]

        class CustomError(Exception):
            pass

        async def custom_error_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise CustomError("Custom error")
            return "success"

        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            retryable_exceptions=(CustomError,),
        )

        result = await retry_with_backoff(custom_error_func, config)

        assert result == "success"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_retry_with_args_and_kwargs(self):
        """Test retry with function arguments."""
        call_count = [0]

        async def func_with_args(x, y, z=0):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RetryableError("Retry")
            return x + y + z

        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        result = await retry_with_backoff(func_with_args, config, 1, 2, z=3)

        assert result == 6
        assert call_count[0] == 2
