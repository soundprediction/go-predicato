"""
Test fixtures and configuration for Predicato tests.
"""

import os
from datetime import datetime, timezone

import pytest


@pytest.fixture
def base_url() -> str:
    """Base URL for the Predicato server."""
    return os.environ.get("PREDICATO_URL", "http://localhost:8080")


@pytest.fixture
def group_id() -> str:
    """Default group ID for tests."""
    return "test-group"


@pytest.fixture
def sample_episode_data() -> dict:
    """Sample episode data for testing."""
    return {
        "name": "Test Episode",
        "content": "This is a test episode about project management and team collaboration.",
        "source": "test",
        "group_id": "test-group",
        "metadata": {"author": "test", "version": 1},
    }


@pytest.fixture
def sample_messages() -> list[dict]:
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "What is the project status?"},
        {"role": "assistant", "content": "The project is on track for the deadline."},
        {"role": "user", "content": "Great, what are the next steps?"},
    ]


@pytest.fixture
def sample_entity_data() -> dict:
    """Sample entity data for testing."""
    return {
        "name": "Alice Smith",
        "entity_type": "Person",
        "group_id": "test-group",
        "attributes": {"role": "Engineer", "department": "Platform", "level": "Senior"},
    }


@pytest.fixture
def sample_datetime() -> datetime:
    """Sample datetime for testing."""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clear_env_vars(monkeypatch):
    """Clear Predicato environment variables before each test."""
    monkeypatch.delenv("PREDICATO_URL", raising=False)
    monkeypatch.delenv("PREDICATO_GROUP_ID", raising=False)
