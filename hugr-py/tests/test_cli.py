"""Tests for the CLI bindings."""

import pytest

from hugr._hugr import cli_with_input
from hugr.package import Package


def test_validate_with_bytes():
    """Test validating a HUGR package using the programmatic bytes API."""

    # Validate using the programmatic API
    result = cli_with_input(["validate"], Package([]).to_bytes())

    # Should return empty bytes for successful validation
    assert result == b""


def test_validate_with_bytes_invalid():
    """Test that invalid packages raise errors through the programmatic API."""
    # Create some invalid bytes
    invalid_bytes = b"not a valid hugr package"

    # Should raise an error
    with pytest.raises(ValueError, match=r"(?i)(error|invalid|magic)"):
        cli_with_input(["validate"], invalid_bytes)
