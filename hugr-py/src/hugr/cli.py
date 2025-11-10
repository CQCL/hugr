"""Python interface for the HUGR CLI subcommands.

Provides programmatic access to the HUGR CLI via Rust bindings.
Exposes a generic `cli_with_io` function and helpers for the main subcommands:
validate, describe, convert, and mermaid.
"""

from hugr._hugr import cli_with_io

__all__ = ["cli_with_io", "validate", "describe", "convert", "mermaid"]


def validate(
    hugr_bytes: bytes,
    *,
    quiet: bool = False,
    no_std: bool = False,
    extensions: str | None = None,
) -> None:
    """Validate a HUGR package.

    Args:
        hugr_bytes: The HUGR package as bytes.
        quiet: Suppress success messages (default: False).
        no_std: Don't include the standard extension (default: False).
        extensions: Path to additional extensions registry file.

    Raises:
        ValueError: On validation failure.
    """
    args = ["validate"]
    if quiet:
        args.extend(["-q"])
    if no_std:
        args.append("--no-std")
    if extensions:
        args.extend(["--extensions", extensions])
    cli_with_io(args, hugr_bytes)


def describe(
    hugr_bytes: bytes,
    *,
    json: bool = False,
    packaged_extensions: bool = False,
    no_resolved_extensions: bool = False,
    public_symbols: bool = False,
    generator_claimed_extensions: bool = False,
) -> str:
    """Describe a HUGR package.

    Args:
        hugr_bytes: The HUGR package as bytes.
        json: Output as JSON (default: False).
        packaged_extensions: Show packaged extensions (default: False).
        no_resolved_extensions: Hide resolved extensions (default: False).
        public_symbols: Show public symbols (default: False).
        generator_claimed_extensions: Show generator claimed extensions
            (default: False).

    Returns:
        Description output as a string (text or JSON).
    """
    args = ["describe"]
    if json:
        args.append("--json")
    if packaged_extensions:
        args.append("--packaged-extensions")
    if no_resolved_extensions:
        args.append("--no-resolved-extensions")
    if public_symbols:
        args.append("--public-symbols")
    if generator_claimed_extensions:
        args.append("--generator-claimed-extensions")
    return cli_with_io(args, hugr_bytes).decode("utf-8")


def convert(
    hugr_bytes: bytes,
    *,
    format: str | None = None,
    text: bool = False,
    binary: bool = False,
    compress: bool = False,
    compression_level: int | None = None,
) -> bytes:
    """Convert a HUGR package between formats.

    Args:
        hugr_bytes: The HUGR package as bytes.
        format: Target format ("json" or "model", default: auto-detect).
        text: Output as text (default: auto-detect).
        binary: Output as binary (default: auto-detect).
        compress: Compress the output (default: False).
        compression_level: Compression level 0-9 (default: 6).

    Returns:
        Converted package as bytes.
    """
    args = ["convert"]
    if format:
        args.extend(["--format", format])
    if text:
        args.append("--text")
    if binary:
        args.append("--binary")
    if compress:
        args.append("--compress")
    if compression_level is not None:
        args.extend(["--compression-level", str(compression_level)])
    return cli_with_io(args, hugr_bytes)


def mermaid(hugr_bytes: bytes, *, validate: bool = False) -> str:
    """Generate mermaid diagrams from a HUGR package.

    Args:
        hugr_bytes: The HUGR package as bytes.
        validate: Validate the HUGR before generating diagram (default: False).

    Returns:
        Mermaid diagram output as a string.
    """
    args = ["mermaid"]
    if validate:
        args.append("--validate")
    return cli_with_io(args, hugr_bytes).decode("utf-8")
