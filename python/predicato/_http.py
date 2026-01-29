"""
Internal HTTP client implementation.

This module provides the low-level HTTP functionality for the Predicato client,
including request/response handling, error translation, and retry logic.
"""

import logging
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from predicato.exceptions import (
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class HTTPClientConfig:
    """Configuration for the HTTP client."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay


class SyncHTTPClient:
    """
    Synchronous HTTP client for Predicato API.

    Handles request/response processing, error handling, and retry logic
    for synchronous operations.
    """

    def __init__(self, config: HTTPClientConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            path: URL path (will be appended to base_url).
            json: JSON body for the request.
            params: Query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            ConnectionError: If the server cannot be reached.
            ValidationError: If the request is invalid (4xx).
            ServerError: If the server returns an error (5xx).
            RateLimitError: If rate limited (429).
            NotFoundError: If resource not found (404).
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(
                    "Request %s %s (attempt %d/%d)",
                    method,
                    path,
                    attempt + 1,
                    self.config.max_retries + 1,
                )

                response = self._client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                )

                logger.debug(
                    "Response %s %s: status=%d",
                    method,
                    path,
                    response.status_code,
                )

                return self._handle_response(response)

            except httpx.TimeoutException as e:
                last_error = ConnectionError(
                    f"Request timed out after {self.config.timeout}s",
                    cause=e,
                )
            except httpx.ConnectError as e:
                last_error = ConnectionError(
                    f"Failed to connect to {self.config.base_url}",
                    cause=e,
                )
            except httpx.RequestError as e:
                last_error = ConnectionError(
                    f"Request failed: {e}",
                    cause=e,
                )
            except ServerError as e:
                # Retry on 5xx errors
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        "Server error (status=%d), retrying in %.1fs",
                        e.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise

            # For non-retryable errors, raise immediately
            if isinstance(last_error, (ValidationError, NotFoundError, RateLimitError)):
                raise last_error

            # For connection errors, retry
            if attempt < self.config.max_retries:
                delay = self._calculate_retry_delay(attempt)
                logger.warning(
                    "Connection error, retrying in %.1fs: %s",
                    delay,
                    last_error,
                )
                time.sleep(delay)

        # All retries exhausted
        if last_error is not None:
            raise last_error

        raise ConnectionError("Request failed after retries")

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle HTTP response and translate errors."""
        if response.status_code == 204:
            return {}

        # Try to parse response body
        body: dict[str, Any]
        try:
            body = dict(response.json())
        except Exception:
            body = {"raw": response.text}

        # Success responses
        if 200 <= response.status_code < 300:
            return body

        # Error responses
        error_message = self._extract_error_message(body)

        if response.status_code == 400:
            raise ValidationError(error_message, details=body)

        if response.status_code == 404:
            raise NotFoundError(error_message, details=body)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_seconds = float(retry_after) if retry_after else None
            raise RateLimitError(error_message, retry_after=retry_seconds, details=body)

        if response.status_code >= 500:
            raise ServerError(
                error_message,
                status_code=response.status_code,
                response_body=response.text,
                details=body,
            )

        # Other 4xx errors
        raise ValidationError(
            f"Request failed with status {response.status_code}: {error_message}",
            details=body,
        )

    def _extract_error_message(self, body: dict[str, Any]) -> str:
        """Extract error message from response body."""
        if isinstance(body, dict):
            # Try common error message fields
            for field in ["message", "error", "detail", "msg"]:
                if field in body and isinstance(body[field], str):
                    return str(body[field])
        return "Unknown error"

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.config.retry_base_delay * (2**attempt)
        return float(min(delay, self.config.retry_max_delay))


class AsyncHTTPClient:
    """
    Asynchronous HTTP client for Predicato API.

    Handles request/response processing, error handling, and retry logic
    for asynchronous operations.
    """

    def __init__(self, config: HTTPClientConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make an async HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            path: URL path (will be appended to base_url).
            json: JSON body for the request.
            params: Query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            ConnectionError: If the server cannot be reached.
            ValidationError: If the request is invalid (4xx).
            ServerError: If the server returns an error (5xx).
            RateLimitError: If rate limited (429).
            NotFoundError: If resource not found (404).
        """
        import asyncio

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(
                    "Request %s %s (attempt %d/%d)",
                    method,
                    path,
                    attempt + 1,
                    self.config.max_retries + 1,
                )

                response = await self._client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                )

                logger.debug(
                    "Response %s %s: status=%d",
                    method,
                    path,
                    response.status_code,
                )

                return self._handle_response(response)

            except httpx.TimeoutException as e:
                last_error = ConnectionError(
                    f"Request timed out after {self.config.timeout}s",
                    cause=e,
                )
            except httpx.ConnectError as e:
                last_error = ConnectionError(
                    f"Failed to connect to {self.config.base_url}",
                    cause=e,
                )
            except httpx.RequestError as e:
                last_error = ConnectionError(
                    f"Request failed: {e}",
                    cause=e,
                )
            except ServerError as e:
                # Retry on 5xx errors
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        "Server error (status=%d), retrying in %.1fs",
                        e.status_code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            # For non-retryable errors, raise immediately
            if isinstance(last_error, (ValidationError, NotFoundError, RateLimitError)):
                raise last_error

            # For connection errors, retry
            if attempt < self.config.max_retries:
                delay = self._calculate_retry_delay(attempt)
                logger.warning(
                    "Connection error, retrying in %.1fs: %s",
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        if last_error is not None:
            raise last_error

        raise ConnectionError("Request failed after retries")

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle HTTP response and translate errors."""
        if response.status_code == 204:
            return {}

        # Try to parse response body
        body: dict[str, Any]
        try:
            body = dict(response.json())
        except Exception:
            body = {"raw": response.text}

        # Success responses
        if 200 <= response.status_code < 300:
            return body

        # Error responses
        error_message = self._extract_error_message(body)

        if response.status_code == 400:
            raise ValidationError(error_message, details=body)

        if response.status_code == 404:
            raise NotFoundError(error_message, details=body)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_seconds = float(retry_after) if retry_after else None
            raise RateLimitError(error_message, retry_after=retry_seconds, details=body)

        if response.status_code >= 500:
            raise ServerError(
                error_message,
                status_code=response.status_code,
                response_body=response.text,
                details=body,
            )

        # Other 4xx errors
        raise ValidationError(
            f"Request failed with status {response.status_code}: {error_message}",
            details=body,
        )

    def _extract_error_message(self, body: dict[str, Any]) -> str:
        """Extract error message from response body."""
        if isinstance(body, dict):
            # Try common error message fields
            for field in ["message", "error", "detail", "msg"]:
                if field in body and isinstance(body[field], str):
                    return str(body[field])
        return "Unknown error"

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.config.retry_base_delay * (2**attempt)
        return float(min(delay, self.config.retry_max_delay))
