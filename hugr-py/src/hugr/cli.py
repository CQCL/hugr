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


def _add_input_args(
    args: list[str], no_std: bool, extensions: list[str] | None
) -> list[str]:
    """Add common HugrInputArgs parameters to the argument list."""
    if no_std:
        args.append("--no-std")
    if extensions is not None:
        for ext in extensions:
            args.extend(["--extensions", ext])
    return args


def validate(
    hugr_bytes: bytes,
    *,
    no_std: bool = False,
    extensions: list[str] | None = None,
) -> None:
    """Validate a HUGR package.

    Args:
        hugr_bytes: The HUGR package as bytes.
        no_std: Don't use standard extensions when validating hugrs.
            Prelude is still used (default: False).
        extensions: Paths to additional serialised extensions needed to load the HUGR.

    Raises:
        ValueError: On validation failure.
    """
    args = _add_input_args(["validate"], no_std, extensions)
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
    no_std: bool = False,
    extensions: list[str] | None = None,
    _json: bool = False,  # only used by describe()
) -> str:
    """Describe the contents of a HUGR package as text.

    If an error occurs during loading, partial descriptions are printed.
    For example, if the first module is loaded and the second fails,
    then only the first module will be described.

    Args:
        hugr_bytes: The HUGR package as bytes.
        packaged_extensions: Enumerate packaged extensions (default: False).
        no_resolved_extensions: Don't display resolved extensions used by the module
            (default: False).
        public_symbols: Display public symbols in the module (default: False).
        generator_claimed_extensions: Display claimed extensions set by generator
            in module metadata (default: False).
        no_std: Don't use standard extensions when validating hugrs.
            Prelude is still used (default: False).
        extensions: Paths to additional serialised extensions needed to load the HUGR.

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
    args = _add_input_args(args, no_std, extensions)
    return cli_with_io(args, hugr_bytes).decode("utf-8")


def describe(
    hugr_bytes: bytes,
    *,
    packaged_extensions: bool = False,
    no_resolved_extensions: bool = False,
    public_symbols: bool = False,
    generator_claimed_extensions: bool = False,
    no_std: bool = False,
    extensions: list[str] | None = None,
) -> PackageDesc:
    """Describe the contents of a HUGR package.

    If an error occurs during loading, partial descriptions are returned.
    For example, if the first module is loaded and the second fails,
    then only the first module will be described.

    Args:
        hugr_bytes: The HUGR package as bytes.
        packaged_extensions: Enumerate packaged extensions (default: False).
        no_resolved_extensions: Don't display resolved extensions used by the module
            (default: False).
        public_symbols: Display public symbols in the module (default: False).
        generator_claimed_extensions: Display claimed extensions set by generator
            in module metadata (default: False).
        no_std: Don't use standard extensions when validating hugrs.
            Prelude is still used (default: False).
        extensions: Paths to additional serialised extensions needed to load the HUGR.

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
        no_std=no_std,
        extensions=extensions,
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
    no_std: bool = False,
    extensions: list[str] | None = None,
) -> bytes:
    """Convert between different HUGR envelope formats.

    Args:
        hugr_bytes: The HUGR package as bytes.
        format: Output format. One of: json, model, model-exts, model-text,
            model-text-exts (default: None, meaning same format as input).
        text: Use default text-based envelope configuration. Cannot be combined
            with format or binary (default: False).
        binary: Use default binary envelope configuration. Cannot be combined
            with format or text (default: False).
        compress: Enable zstd compression for the output (default: False).
        compression_level: Zstd compression level (1-22, where 1 is fastest and
            22 is best compression). (default None, uses the zstd default).
        no_std: Don't use standard extensions when validating hugrs.
            Prelude is still used (default: False).
        extensions: Paths to additional serialised extensions needed to load the HUGR.

    Returns:
        Converted package as bytes.
    """
    args = ["convert"]
    if format is not None:
        args.extend(["--format", format])
    if text:
        args.append("--text")
    if binary:
        args.append("--binary")
    if compress:
        args.append("--compress")
    if compression_level is not None:
        args.extend(["--compression-level", str(compression_level)])
    args = _add_input_args(args, no_std, extensions)
    return cli_with_io(args, hugr_bytes)


def mermaid(
    hugr_bytes: bytes,
    *,
    validate: bool = False,
    no_std: bool = False,
    extensions: list[str] | None = None,
) -> str:
    """Generate mermaid diagrams from a HUGR package.

    Args:
        hugr_bytes: The HUGR package as bytes.
        validate: Validate before rendering, includes extension inference
            (default: False).
        no_std: Don't use standard extensions when validating hugrs.
            Prelude is still used (default: False).
        extensions: Paths to additional serialised extensions needed to load the HUGR.

    Returns:
        Mermaid diagram output as a string.
    """
    args = ["mermaid"]
    if validate:
        args.append("--validate")
    args = _add_input_args(args, no_std, extensions)
    return cli_with_io(args, hugr_bytes).decode("utf-8")
