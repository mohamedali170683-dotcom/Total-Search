"""Base API client with common functionality."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    pass


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, calls_per_minute: int, burst_size: int | None = None):
        self.calls_per_minute = calls_per_minute
        self.burst_size = burst_size or calls_per_minute
        self.tokens = self.burst_size
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * (self.calls_per_minute / 60.0),
            )
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.calls_per_minute / 60.0)
                logger.debug(f"Rate limiter waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class BaseAPIClient(ABC):
    """Abstract base class for API clients."""

    def __init__(
        self,
        base_url: str,
        settings: Settings | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.settings = settings or get_settings()
        self.rate_limiter = rate_limiter
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers=self._get_default_headers(),
            )
        return self._client

    @abstractmethod
    def _get_default_headers(self) -> dict[str, str]:
        """Get default headers for requests."""
        pass

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "BaseAPIClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict | None = None,
        json_data: dict | list | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic."""
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        logger.debug(f"Making {method} request to {url}")

        response = await self.client.request(
            method=method,
            url=endpoint,
            params=params,
            json=json_data,
            headers=headers,
        )

        return self._handle_response(response)

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and raise appropriate errors."""
        try:
            data = response.json()
        except Exception:
            data = {"raw": response.text}

        if response.status_code == 401:
            raise AuthenticationError(
                "Authentication failed",
                status_code=401,
                response_data=data,
            )

        if response.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded",
                status_code=429,
                response_data=data,
            )

        if response.status_code >= 400:
            raise APIError(
                f"API request failed: {response.status_code}",
                status_code=response.status_code,
                response_data=data,
            )

        return data

    async def get(
        self,
        endpoint: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """Make a GET request."""
        return await self._request("GET", endpoint, params=params, headers=headers)

    async def post(
        self,
        endpoint: str,
        json_data: dict | list | None = None,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """Make a POST request."""
        return await self._request(
            "POST", endpoint, json_data=json_data, params=params, headers=headers
        )


def batch_items(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into batches of specified size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


async def process_in_batches(
    items: list[T],
    batch_size: int,
    processor: Callable[[list[T]], Any],
    delay_between_batches: float = 0.0,
) -> list[Any]:
    """Process items in batches with optional delay between batches."""
    results = []
    batches = batch_items(items, batch_size)

    for i, batch in enumerate(batches):
        result = await processor(batch)
        results.append(result)

        if delay_between_batches > 0 and i < len(batches) - 1:
            await asyncio.sleep(delay_between_batches)

    return results
