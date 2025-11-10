"""Tests for the CLI bindings."""

import pytest

from hugr import cli
from hugr.build import Module
from hugr.package import Package


@pytest.fixture
def simple_hugr_bytes() -> bytes:
    """Create a simple HUGR package as bytes for testing."""
    return Package([Module().hugr]).to_bytes()


def test_validate_with_bytes(simple_hugr_bytes):
    """Test validating a HUGR package using the programmatic API."""
    # Validate using the programmatic API
    cli.validate(simple_hugr_bytes)  # Should not raise


def test_validate_with_bytes_invalid():
    """Test that invalid packages raise errors through the programmatic API."""
    # We need to pass invalid bytes through cli_with_io directly
    # since Package construction would fail first

    invalid_bytes = b"not a valid hugr package"

    # Should raise an error when trying to validate
    with pytest.raises(ValueError, match="Bad magic number"):
        cli.cli_with_io(["validate"], invalid_bytes)


def test_validate_quiet(simple_hugr_bytes):
    """Test validate with quiet flag."""
    # Validate with quiet mode
    cli.validate(simple_hugr_bytes, quiet=True)  # Should not raise


def test_validate_no_std(simple_hugr_bytes):
    """Test validate with no_std flag."""
    # Validate without standard extensions
    cli.validate(simple_hugr_bytes, no_std=True)  # Should not raise


def test_convert_format(simple_hugr_bytes):
    """Test converting a HUGR package between formats."""
    # Convert to JSON format
    output_bytes = cli.convert(simple_hugr_bytes, format="json")

    # Output should be valid bytes
    assert isinstance(output_bytes, bytes)
    assert len(output_bytes) > 0

    # Should be parseable as a package
    output_package = Package.from_bytes(output_bytes)
    input_package = Package.from_bytes(simple_hugr_bytes)
    assert len(output_package.modules) == len(input_package.modules)


def test_convert_binary_to_text(simple_hugr_bytes):
    """Test converting a HUGR package from binary to text format."""
    # Convert to text format using --text flag
    output_bytes = cli.convert(simple_hugr_bytes, text=True)

    # Should be valid bytes
    assert isinstance(output_bytes, bytes)
    assert len(output_bytes) > 0

    # Should be parseable as a package
    output_package = Package.from_bytes(output_bytes)
    input_package = Package.from_bytes(simple_hugr_bytes)
    assert len(output_package.modules) == len(input_package.modules)


def test_convert_with_compression(simple_hugr_bytes):
    """Test converting with compression enabled."""
    # Convert with compression
    output_bytes = cli.convert(simple_hugr_bytes, compress=True, compression_level=9)

    # Should be valid bytes
    assert isinstance(output_bytes, bytes)
    assert len(output_bytes) > 0

    # Should be parseable as a package
    output_package = Package.from_bytes(output_bytes)
    input_package = Package.from_bytes(simple_hugr_bytes)
    assert len(output_package.modules) == len(input_package.modules)


def test_mermaid_output(simple_hugr_bytes):
    """Test generating mermaid diagrams from a HUGR package."""
    # Generate mermaid diagram
    output = cli.mermaid(simple_hugr_bytes)

    # Should produce mermaid output
    assert isinstance(output, str)
    assert "graph LR" in output


def test_mermaid_with_validation(simple_hugr_bytes):
    """Test generating mermaid diagrams with validation."""
    # Generate mermaid diagram with validation
    output = cli.mermaid(simple_hugr_bytes, validate=True)
    assert "graph LR" in output


def test_describe_output(simple_hugr_bytes):
    """Test describing a HUGR package."""
    # Describe the package
    output_text = cli.describe(simple_hugr_bytes)

    # Output should not be empty
    assert len(output_text) > 0
    assert isinstance(output_text, str)

    # Should contain package information
    assert "Package contains" in output_text or "module" in output_text.lower()


def test_describe_json_output(simple_hugr_bytes):
    """Test describing a HUGR package in JSON format."""
    import json

    # Describe the package in JSON format
    output_text = cli.describe(simple_hugr_bytes, json=True)

    # Output should be valid JSON
    assert len(output_text) > 0
    description = json.loads(output_text)

    # Should have expected fields
    assert "header" in description
    assert "modules" in description


def test_describe_with_options(simple_hugr_bytes):
    """Test describe with various options."""
    # Test with packaged_extensions flag
    output_text = cli.describe(simple_hugr_bytes, packaged_extensions=True)
    assert "Packaged extensions:" in output_text

    # Test with no_resolved_extensions flag
    output_text = cli.describe(simple_hugr_bytes, no_resolved_extensions=True)
    assert "resolved" not in output_text

    # Test with public_symbols flag
    output_text = cli.describe(simple_hugr_bytes, public_symbols=True)
    assert len(output_text) > 0

    # Test with generator_claimed_extensions flag
    output_text = cli.describe(simple_hugr_bytes, generator_claimed_extensions=True)
    assert len(output_text) > 0
