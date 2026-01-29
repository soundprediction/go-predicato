"""
Tests for exception classes.
"""

import pytest

from predicato.exceptions import (
    ConfigurationError,
    ConnectionError,
    NotFoundError,
    PredicatoError,
    RateLimitError,
    ServerError,
    ValidationError,
)


class TestPredicatoError:
    """Tests for the base PredicatoError."""

    def test_basic_error(self):
        """Test creating a basic error."""
        error = PredicatoError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_error_with_details(self):
        """Test creating an error with details."""
        error = PredicatoError("Error", details={"code": 123, "extra": "info"})
        assert "code" in error.details
        assert error.details["code"] == 123
        assert "details" in str(error)

    def test_error_is_exception(self):
        """Test that error can be raised and caught."""
        with pytest.raises(PredicatoError) as exc_info:
            raise PredicatoError("Test error")
        assert "Test error" in str(exc_info.value)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_is_predicato_error(self):
        """Test that it inherits from PredicatoError."""
        error = ConfigurationError("Missing config")
        assert isinstance(error, PredicatoError)

    def test_can_catch_as_base(self):
        """Test catching as PredicatoError."""
        with pytest.raises(PredicatoError):
            raise ConfigurationError("Missing URL")


class TestValidationError:
    """Tests for ValidationError."""

    def test_basic_validation_error(self):
        """Test creating a validation error."""
        error = ValidationError("Invalid data")
        assert error.message == "Invalid data"
        assert error.field is None

    def test_validation_error_with_field(self):
        """Test validation error with field name."""
        error = ValidationError("Too long", field="name")
        assert error.field == "name"

    def test_validation_error_with_details(self):
        """Test validation error with details."""
        error = ValidationError(
            "Invalid input",
            field="content",
            details={"max_length": 1000, "actual": 1500},
        )
        assert error.field == "content"
        assert error.details["max_length"] == 1000


class TestConnectionError:
    """Tests for ConnectionError."""

    def test_basic_connection_error(self):
        """Test creating a connection error."""
        error = ConnectionError("Cannot reach server")
        assert error.message == "Cannot reach server"
        assert error.cause is None

    def test_connection_error_with_cause(self):
        """Test connection error with underlying cause."""
        cause = OSError("Network unreachable")
        error = ConnectionError("Connection failed", cause=cause)
        assert error.cause is cause
        assert error.__cause__ is cause

    def test_exception_chaining(self):
        """Test that exception chaining works."""
        cause = TimeoutError("Timed out")
        error = ConnectionError("Request failed", cause=cause)

        try:
            raise error
        except ConnectionError as e:
            assert e.__cause__ is cause


class TestServerError:
    """Tests for ServerError."""

    def test_basic_server_error(self):
        """Test creating a server error."""
        error = ServerError("Internal error")
        assert error.message == "Internal error"
        assert error.status_code == 500
        assert error.response_body is None

    def test_server_error_with_details(self):
        """Test server error with full details."""
        error = ServerError(
            "Gateway timeout",
            status_code=504,
            response_body='{"error": "upstream timeout"}',
        )
        assert error.status_code == 504
        assert error.response_body == '{"error": "upstream timeout"}'


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_basic_rate_limit_error(self):
        """Test creating a rate limit error."""
        error = RateLimitError("Too many requests")
        assert error.message == "Too many requests"
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self):
        """Test rate limit error with retry after."""
        error = RateLimitError("Rate limited", retry_after=30.0)
        assert error.retry_after == 30.0


class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_basic_not_found_error(self):
        """Test creating a not found error."""
        error = NotFoundError("Resource not found")
        assert error.message == "Resource not found"
        assert error.resource_type is None
        assert error.resource_id is None

    def test_not_found_error_with_details(self):
        """Test not found error with resource details."""
        error = NotFoundError(
            "Node not found",
            resource_type="node",
            resource_id="node-123",
        )
        assert error.resource_type == "node"
        assert error.resource_id == "node-123"


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_inherit_from_base(self):
        """Test all exceptions inherit from PredicatoError."""
        exceptions = [
            ConfigurationError("test"),
            ValidationError("test"),
            ConnectionError("test"),
            ServerError("test"),
            RateLimitError("test"),
            NotFoundError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, PredicatoError)

    def test_can_catch_all_with_base(self):
        """Test catching all with PredicatoError."""
        exceptions = [
            ConfigurationError,
            ValidationError,
            ConnectionError,
            ServerError,
            RateLimitError,
            NotFoundError,
        ]
        for exc_class in exceptions:
            with pytest.raises(PredicatoError):
                raise exc_class("test")
