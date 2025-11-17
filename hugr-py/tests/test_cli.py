"""Tests for the CLI bindings."""

from typing import Any

import pytest

from hugr import cli, tys
from hugr.build import Dfg, Module
from hugr.ext import Extension
from hugr.package import Package

from .serialization.test_extension import EXAMPLE


@pytest.fixture
def simple_hugr_bytes() -> bytes:
    """Create a simple HUGR package as bytes for testing."""
    return Package([Module().hugr]).to_bytes()


@pytest.fixture
def hugr_with_extension_bytes() -> bytes:
    """Create a HUGR package with an extension as bytes for testing."""
    ext = Extension.from_json(EXAMPLE)
    module = Module()
    return Package([module.hugr], [ext]).to_bytes()


def test_validate_with_bytes(simple_hugr_bytes: bytes):
    """Test validating a HUGR package using the programmatic API."""
    cli.validate(simple_hugr_bytes)


def test_validate_with_bytes_invalid():
    """Test that invalid packages raise errors through the programmatic API."""
    # We need to pass invalid bytes through cli_with_io directly
    # since Package construction would fail first

    invalid_bytes = b"not a valid hugr package"

    with pytest.raises(cli.HugrCliError, match="Bad magic number"):
        cli.cli_with_io(["validate"], invalid_bytes)


def test_validate_no_std(simple_hugr_bytes: bytes):
    """Test validate with no_std flag."""
    cli.validate(simple_hugr_bytes, no_std=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "json"},  # convert to JSON format
        {"text": True},  # convert to text format
        {"compress": True, "compression_level": 9},  # convert with compression
    ],
)
def test_convert_format(hugr_with_extension_bytes: bytes, kwargs: dict[str, Any]):
    """Test converting a HUGR package between formats."""
    output_bytes = cli.convert(hugr_with_extension_bytes, **kwargs)

    output_package = Package.from_bytes(output_bytes)
    input_package = Package.from_bytes(hugr_with_extension_bytes)
    assert output_package == input_package


def test_mermaid_output(simple_hugr_bytes: bytes):
    """Test generating mermaid diagrams from a HUGR package."""
    output = cli.mermaid(simple_hugr_bytes)

    assert "graph LR" in output


def test_mermaid_with_validation(simple_hugr_bytes: bytes):
    """Test generating mermaid diagrams with validation."""
    output = cli.mermaid(simple_hugr_bytes, validate=True)
    assert "graph LR" in output


def test_describe_output(simple_hugr_bytes: bytes):
    """Test describing a HUGR package."""
    output_text = cli.describe_str(simple_hugr_bytes)

    # Should contain package information
    assert "Package contains" in output_text


def test_describe_with_options(hugr_with_extension_bytes: bytes):
    """Test describe with various options."""
    output_text = cli.describe_str(hugr_with_extension_bytes, packaged_extensions=True)
    assert "Packaged extensions:" in output_text

    output_text = cli.describe_str(
        hugr_with_extension_bytes, no_resolved_extensions=True
    )
    assert "resolved" not in output_text

    output_text = cli.describe_str(hugr_with_extension_bytes, public_symbols=True)
    assert len(output_text) > 0

    # Test with generator_claimed_extensions flag
    output_text = cli.describe_str(
        hugr_with_extension_bytes, generator_claimed_extensions=True
    )
    assert len(output_text) > 0


def test_describe_json_basic(simple_hugr_bytes: bytes):
    """Test describe_json returns structured PackageDesc."""
    desc = cli.describe(simple_hugr_bytes)

    assert isinstance(desc, cli.PackageDesc)

    # Should have expected fields
    assert desc.header is not None
    assert isinstance(desc.modules, list)
    assert len(desc.modules) == 1

    # Module should have properties
    module = desc.modules[0]
    assert module is not None
    assert isinstance(module, cli.ModuleDesc)
    assert module.num_nodes is not None
    assert module.num_nodes > 0


def test_describe_json_with_packaged_extensions(hugr_with_extension_bytes: bytes):
    """Test describe_json with packaged_extensions flag."""
    desc = cli.describe(hugr_with_extension_bytes, packaged_extensions=True)

    # Should have packaged_extensions field populated
    assert isinstance(desc, cli.PackageDesc)
    assert desc.packaged_extensions is not None

    mod = desc.modules[0]
    assert mod is not None
    # mock use of extension in module
    mod.used_extensions_resolved = desc.packaged_extensions  # type: ignore[assignment]

    assert desc.uses_extension("ext")
    assert not desc.uses_extension("nonexistent_extension")


@pytest.fixture
def hugr_using_ext() -> bytes:
    """A simple HUGR package that uses an extension, but doesn't package it."""
    ext = Extension.from_json(EXAMPLE)
    u_t = tys.USize()
    op = ext.get_op("New").instantiate(
        [u_t.type_arg()], concrete_signature=tys.FunctionType([u_t], [])
    )
    h = Dfg(u_t)
    a = h.inputs()[0]
    h.add_op(op, a)
    h.set_outputs()

    package = Package([h.hugr], [])

    return package.to_bytes()


def test_failed_describe(hugr_using_ext):
    """Json description still succeeds, with error field populated"""
    desc = cli.describe(hugr_using_ext)
    mod = desc.modules[0]
    assert mod is not None
    assert mod.num_nodes == 8  # computed before error
    assert isinstance(desc.error, str)
    assert "requires extension ext" in desc.error

    assert desc.uses_extension("ext")
