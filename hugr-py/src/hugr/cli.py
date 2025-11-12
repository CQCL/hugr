"""Python interface for the HUGR CLI subcommands.

Provides programmatic access to the HUGR CLI via Rust bindings.
Exposes a generic `cli_with_io` function and helpers for the main subcommands:
validate, describe, convert, and mermaid.
"""

from pydantic import BaseModel

from hugr._hugr import cli_with_io

__all__ = [
    "cli_with_io",
    "validate",
    "describe_str",
    "describe",
    "convert",
    "mermaid",
    "PackageDesc",
    "ModuleDesc",
    "ExtensionDesc",
    "EntrypointDesc",
]


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


class EntrypointDesc(BaseModel):
    """Description of a module's entrypoint node.

    Attributes:
        node: The node index of the entrypoint.
        optype: String representation of the operation type.
    """

    node: int
    optype: str


class ExtensionDesc(BaseModel):
    """Description of a HUGR extension.

    Attributes:
        name: The name of the extension.
        version: The version string of the extension.
    """

    name: str
    version: str


class ModuleDesc(BaseModel):
    """Description of a HUGR module.

    Attributes:
        entrypoint: The entrypoint node of the module, if present.
        generator: Name and version of the generator that created this module.
        num_nodes: Total number of nodes in the module.
        public_symbols: List of public symbol names exported by the module.
        used_extensions_generator: Extensions claimed by the generator in metadata.
        used_extensions_resolved: Extensions actually used by the module operations.
    """

    entrypoint: EntrypointDesc | None = None
    generator: str | None = None
    num_nodes: int | None = None
    public_symbols: list[str] | None = None
    used_extensions_generator: list[ExtensionDesc] | None = None
    used_extensions_resolved: list[ExtensionDesc] | None = None

    def uses_extension(self, extension_name: str) -> bool:
        """Check if this module uses a specific extension.

        Args:
            extension_name: The name of the extension to check.

        Returns:
            True if the module uses the extension, False otherwise.
        """
        return any(
            ext.name == extension_name for ext in self.used_extensions_resolved or []
        )


class PackageDesc(BaseModel):
    """Description of a HUGR package.

    Attributes:
        error: Error message if the package failed to load.
        header: String representation of the envelope header.
        modules: List of module descriptions in the package.
        packaged_extensions: Extensions bundled with the package.
    """

    error: str | None = None
    header: str
    modules: list[ModuleDesc | None]
    packaged_extensions: list[ExtensionDesc | None] | None = None

    def uses_extension(self, extension_name: str) -> bool:
        """Check if any module in this package uses a specific extension.

        Args:
            extension_name: The name of the extension to check.

        Returns:
            True if any module uses the extension, False otherwise.
        """
        return any(
            module.uses_extension(extension_name)
            for module in self.modules
            if module is not None
        )


def describe_str(
    hugr_bytes: bytes,
    *,
    packaged_extensions: bool = False,
    no_resolved_extensions: bool = False,
    public_symbols: bool = False,
    generator_claimed_extensions: bool = False,
    _json: bool = False,  # only used by describe()
) -> str:
    """Describe a HUGR package with a string.

    Args:
        hugr_bytes: The HUGR package as bytes.
        packaged_extensions: Show packaged extensions (default: False).
        no_resolved_extensions: Hide resolved extensions (default: False).
        public_symbols: Show public symbols (default: False).
        generator_claimed_extensions: Show generator claimed extensions
            (default: False).

    Returns:
        Text description of the package.
    """
    args = ["describe"]
    if _json:
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


def describe(
    hugr_bytes: bytes,
    *,
    packaged_extensions: bool = False,
    no_resolved_extensions: bool = False,
    public_symbols: bool = False,
    generator_claimed_extensions: bool = False,
) -> PackageDesc:
    """Describe a HUGR package.

    Args:
        hugr_bytes: The HUGR package as bytes.
        packaged_extensions: Show packaged extensions (default: False).
        no_resolved_extensions: Hide resolved extensions (default: False).
        public_symbols: Show public symbols (default: False).
        generator_claimed_extensions: Show generator claimed extensions
            (default: False).

    Returns:
        Structured package description as a PackageDesc object.
    """
    output = describe_str(
        hugr_bytes,
        _json=True,
        packaged_extensions=packaged_extensions,
        no_resolved_extensions=no_resolved_extensions,
        public_symbols=public_symbols,
        generator_claimed_extensions=generator_claimed_extensions,
    )
    return PackageDesc.model_validate_json(output)


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
