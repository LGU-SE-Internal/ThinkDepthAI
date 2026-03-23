"""Rate limiter with exponential backoff retry for API calls."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 20
DEFAULT_MIN_WAIT = 1  # seconds
DEFAULT_MAX_WAIT = 60  # seconds
DEFAULT_JITTER = 1  # seconds


def _get_retryable_exceptions() -> tuple:
    """Dynamically collect retryable exception types."""
    exceptions: list[type[Exception]] = [asyncio.TimeoutError]

    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        exceptions.extend([RateLimitError, APITimeoutError, APIConnectionError, InternalServerError])
    except ImportError:
        pass

    try:
        from anthropic import InternalServerError as AnthropicInternalServerError

        exceptions.append(AnthropicInternalServerError)
    except ImportError:
        pass

    try:
        from google.genai.errors import ServerError as GeminiServerError

        exceptions.append(GeminiServerError)
    except ImportError:
        pass

    try:
        from pydantic import ValidationError as PydanticValidationError

        exceptions.append(PydanticValidationError)
    except ImportError:
        pass

    return tuple(exceptions)


RETRYABLE_EXCEPTIONS = _get_retryable_exceptions()


class RateLimiter:
    """Rate limiter with RPM (requests per minute) and TPM (tokens per minute) support."""

    def __init__(self, rpm: int | None = None, tpm: int | None = None, burst_multiplier: float = 1.0):
        self._rpm = rpm
        self._tpm = tpm
        self._request_limiter: AsyncLimiter | None = None
        self._token_limiter: AsyncLimiter | None = None

        if rpm is not None and rpm > 0:
            self._request_limiter = AsyncLimiter(max_rate=rpm * burst_multiplier, time_period=60)
        if tpm is not None and tpm > 0:
            self._token_limiter = AsyncLimiter(max_rate=tpm * burst_multiplier, time_period=60)

    async def acquire_request(self) -> None:
        if self._request_limiter is not None:
            await self._request_limiter.acquire()

    async def __aenter__(self):
        await self.acquire_request()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


async def retry_with_backoff[T](
    func: Callable[[], Awaitable[T]],
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_wait: float = DEFAULT_MIN_WAIT,
    max_wait: float = DEFAULT_MAX_WAIT,
    jitter: float = DEFAULT_JITTER,
) -> T:
    """Execute an async function with exponential backoff retry."""
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await func()
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)
                wait_time += random.uniform(0, jitter)
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed with {type(e).__name__}: {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} retry attempts failed. Last error: {e}")

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in retry_with_backoff")
