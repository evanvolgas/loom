"""Example demonstrating retry mechanism."""

import asyncio
from loom.resilience.retry import retry_with_backoff, RetryConfig, RetryableError


async def flaky_api_call(attempt_counter):
    """Simulates a flaky API that fails first 2 times."""
    attempt_counter[0] += 1
    print(f"  Attempt {attempt_counter[0]}...", end=" ")

    if attempt_counter[0] < 3:
        print("❌ Failed (simulated error)")
        raise RetryableError("API temporarily unavailable")

    print("✅ Success!")
    return {"status": "ok", "data": "API response"}


async def main():
    print("🔄 Testing Retry Mechanism\n")

    # Example 1: Successful retry
    print("Example 1: Flaky API that succeeds on 3rd attempt")
    attempt_counter = [0]
    config = RetryConfig(
        max_attempts=5,
        initial_delay=0.5,
        exponential_base=2.0
    )

    result = await retry_with_backoff(flaky_api_call, config, attempt_counter)
    print(f"  Result: {result}\n")

    # Example 2: Fast retries
    print("Example 2: Fast retries with short delay")
    attempt_counter2 = [0]
    config2 = RetryConfig(
        max_attempts=3,
        initial_delay=0.1,
        exponential_base=1.5
    )

    result2 = await retry_with_backoff(flaky_api_call, config2, attempt_counter2)
    print(f"  Result: {result2}\n")

    print("✅ All retry examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
