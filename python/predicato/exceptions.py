"""
Exceptions for the Predicato Python client.

This module defines a hierarchy of exceptions for handling various error
conditions that may occur when interacting with the Predicato API.
"""

from typing import Any


class PredicatoError(Exception):
    """
    Base exception for all Predicato client errors.

    All exceptions raised by the Predicato client inherit from this class,
    making it easy to catch any client-related error.

    Args:
        message: Human-readable error message.
        details: Optional additional details about the error.

    Example:
        >>> try:
        ...     client.add_episode(...)
        ... except PredicatoError as e:
        ...     print(f"Predicato error: {e}")
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(PredicatoError):
    """
    Raised when the client is misconfigured.

    This typically occurs when required configuration like base_url is not
    provided either through constructor parameters or environment variables.

    Example:
        >>> # Without PREDICATO_URL set
        >>> client = PredicatoClient()  # Raises ConfigurationError
    """

    pass


class ValidationError(PredicatoError):
    """
    Raised when request data fails validation.

    This can occur either during client-side validation (before making the API
    call) or when the server returns a 400 Bad Request response.

    Args:
        message: Description of the validation failure.
        field: The field that failed validation (if applicable).
        details: Additional validation error details.

    Example:
        >>> episode = Episode(name="x" * 2000, ...)  # Raises ValidationError
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.field = field


class ConnectionError(PredicatoError):
    """
    Raised when the client cannot connect to the server.

    This includes network errors, DNS resolution failures, connection timeouts,
    and other transport-level issues.

    Args:
        message: Description of the connection failure.
        cause: The underlying exception that caused this error.

    Example:
        >>> client = PredicatoClient(base_url="http://nonexistent:8080")
        >>> client.search("query")  # Raises ConnectionError
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause


class ServerError(PredicatoError):
    """
    Raised when the server returns a 5xx error.

    This indicates an internal server error that is not caused by the client's
    request. The client may retry the request after some delay.

    Args:
        message: Description of the server error.
        status_code: HTTP status code from the server.
        response_body: Raw response body if available.

    Example:
        >>> # Server returns 500 Internal Server Error
        >>> client.add_episode(...)  # Raises ServerError
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        response_body: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(PredicatoError):
    """
    Raised when the server returns a 429 Too Many Requests error.

    This indicates the client has exceeded the rate limit and should wait
    before making additional requests.

    Args:
        message: Description of the rate limit error.
        retry_after: Number of seconds to wait before retrying (if provided).

    Example:
        >>> try:
        ...     client.add_episode(...)
        ... except RateLimitError as e:
        ...     if e.retry_after:
        ...         time.sleep(e.retry_after)
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.retry_after = retry_after


class NotFoundError(PredicatoError):
    """
    Raised when the requested resource is not found.

    This corresponds to a 404 Not Found response from the server.

    Args:
        message: Description of what was not found.
        resource_type: Type of resource that was not found (e.g., "node", "group").
        resource_id: Identifier of the resource that was not found.

    Example:
        >>> client.get_node("nonexistent-uuid")  # Raises NotFoundError
    """

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id
