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


def test_convert_format():
    """Test converting a HUGR package between formats."""
    # Create a simple package
    package = Package([])
    input_bytes = package.to_bytes()

    # Convert to JSON format
    output_bytes = cli_with_input(["convert", "--format", "json"], input_bytes)

    # Output should not be empty
    assert len(output_bytes) > 0

    # Output should be a valid HUGR package
    output_package = Package.from_bytes(output_bytes)
    assert len(output_package.modules) == len(package.modules)


def test_convert_binary_to_text():
    """Test converting a HUGR package from binary to text format."""
    package = Package([])
    input_bytes = package.to_bytes()

    # Convert to text format using --text flag
    output_bytes = cli_with_input(["convert", "--text"], input_bytes)

    # Output should not be empty
    assert len(output_bytes) > 0

    # Should be able to parse back as a package
    output_package = Package.from_bytes(output_bytes)
    assert len(output_package.modules) == len(package.modules)
