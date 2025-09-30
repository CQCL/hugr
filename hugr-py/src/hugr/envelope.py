"""Generic serialization of HUGRs and Packages.

The format is designed to be extensible and backwards-compatible. It
consists of a header declaring the format used to encode the HUGR, followed
by the encoded HUGR itself.

## Payload formats

The envelope may encode the HUGR in different formats, listed in
:class:`hugr.envelope.EnvelopeFormat`. The payload may also be compressed with zstd.

Some formats can be represented as ASCII, as indicated by the
:meth:`hugr.envelope.EnvelopeFormat.ascii_printable` method. When this is the case, the
whole envelope can be stored in a string.

## Envelope header

The binary header format is 10 bytes, with the following fields:

| Field  | Size (bytes) | Description |
|--------|--------------|-------------|
| Magic  | 8            | :class:`hugr.envelope.MAGIC_NUMBERS` constant identifying the envelope format. |
| Format | 1            | :class:`hugr.envelope.EnvelopeFormat` describing the payload format. |
| Flags  | 1            | Additional configuration flags. |

Flags:

- Bit 0: Whether the payload is compressed with zstd.
- Bits 1-5: Reserved for future use.
- Bit 7,6: Constant "01" to make some headers ascii-printable.
"""  # noqa: E501

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import pyzstd

import hugr._hugr as rust

if TYPE_CHECKING:
    from hugr.hugr.base import Hugr
    from hugr.package import Package

# This is a hard-coded magic number that identifies the start of a HUGR envelope.
MAGIC_NUMBERS = b"HUGRiHJv"

# The all-unset header flags configuration.
# Bit 7 is always set to ensure we have a printable ASCII character.
_DEFAULT_FLAGS = 0b0100_0000
# The ZSTD flag bit in the header's flags.
_ZSTD_FLAG = 0b0000_0001


def make_envelope(package: Package | Hugr, config: EnvelopeConfig) -> bytes:
    """Encode a HUGR or Package into an envelope, using the given configuration."""
    from hugr.package import Package

    envelope = bytearray(config._make_header().to_bytes())

    if not isinstance(package, Package):
        package = Package(modules=[package], extensions=[])

    # Currently only uncompressed JSON is supported.
    payload: bytes
    match config.format:
        case EnvelopeFormat.JSON:
            json_str = package._to_serial().model_dump_json()
            # This introduces an extra encode/decode roundtrip when calling
            # `make_envelope_str`, but we prioritize speed for binary formats.
            payload = json_str.encode("utf-8")

        case EnvelopeFormat.MODEL:
            payload = bytes(package.to_model())

        case EnvelopeFormat.MODEL_WITH_EXTS:
            package_bytes = bytes(package.to_model())
            extension_str = json.dumps(
                [ext._to_serial().model_dump(mode="json") for ext in package.extensions]
            )
            extension_bytes = extension_str.encode("utf8")
            payload = package_bytes + extension_bytes

    if config.zstd is not None:
        payload = pyzstd.compress(payload, config.zstd)

    envelope += payload
    return bytes(envelope)


def make_envelope_str(package: Package | Hugr, config: EnvelopeConfig) -> str:
    """Encode a HUGR or Package into an envelope, using the given configuration."""
    if not config.format.ascii_printable():
        msg = "Only ascii-printable envelope formats can be encoded into a string."
        raise ValueError(msg)
    envelope = make_envelope(package, config)
    return envelope.decode("utf-8")


def read_envelope(envelope: bytes) -> Package:
    """Decode a HUGR package from an envelope."""
    import hugr._serialization.extension as ext_s

    header = EnvelopeHeader.from_bytes(envelope)
    payload = envelope[10:]

    if header.zstd:
        payload = pyzstd.decompress(payload)

    match header.format:
        case EnvelopeFormat.JSON:
            return ext_s.Package.model_validate_json(payload).deserialize()
        case EnvelopeFormat.MODEL | EnvelopeFormat.MODEL_WITH_EXTS:
            # TODO Going via JSON is a temporary solution, until we get model import to
            # python properly implemented.
            # https://github.com/CQCL/hugr/issues/2287
            json_data = rust.to_json_envelope(envelope)
            return read_envelope(json_data)


def read_envelope_hugr(envelope: bytes) -> Hugr:
    """Decode a HUGR from an envelope.

    Raises:
        ValueError: If the envelope does not contain a single module.
    """
    pkg = read_envelope(envelope)
    if len(pkg.modules) != 1:
        msg = (
            "Expected a single module in the envelope, but got "
            + f"{len(pkg.modules)} modules."
        )
        raise ValueError(msg)
    return pkg.modules[0]


def read_envelope_str(envelope: str) -> Package:
    """Decode a HUGR package from an envelope."""
    return read_envelope(envelope.encode("utf-8"))


def read_envelope_hugr_str(envelope: str) -> Hugr:
    """Decode a HUGR from an envelope.

    Raises:
        ValueError: If the envelope does not contain a single module.
    """
    pkg = read_envelope_str(envelope)
    if len(pkg.modules) != 1:
        msg = (
            "Expected a single module in the envelope, but got "
            + f"{len(pkg.modules)} modules."
        )
        raise ValueError(msg)
    return pkg.modules[0]


class EnvelopeFormat(Enum):
    """Format used to encode a HUGR envelope."""

    MODEL = 1
    """A capnp-encoded hugr-model."""
    MODEL_WITH_EXTS = 2
    """A capnp-encoded hugr-model, immediately followed by a json-encoded
    extension registry."""
    JSON = 63  # '?' in ASCII
    """A json-encoded hugr-package. This format is ASCII-printable."""

    def ascii_printable(self) -> bool:
        return self in {EnvelopeFormat.JSON}


@dataclass
class EnvelopeHeader:
    """Header of a HUGR envelope.

    See the module docstring for the binary format.

    Attributes:
        format: The format used to encode the HUGR.
        zstd: Whether the payload is compressed with zstd.
            The compression level is detected from the payload.
    """

    format: EnvelopeFormat
    zstd: bool = False

    def to_bytes(self) -> bytes:
        header_bytes = bytearray(MAGIC_NUMBERS)
        header_bytes.append(self.format.value)
        flags = _DEFAULT_FLAGS
        if self.zstd:
            flags |= _ZSTD_FLAG
        header_bytes.append(flags)
        return bytes(header_bytes)

    @staticmethod
    def from_bytes(data: bytes) -> EnvelopeHeader:
        if len(data) < 10:
            msg = (
                "Invalid HUGR envelope."
                + f"Expected at least 10 bytes, but got {len(data)} bytes."
            )
            raise ValueError(msg)
        if data[:8] != MAGIC_NUMBERS:
            msg = (
                "Invalid magic number for HUGR envelope."
                + f"Expected {MAGIC_NUMBERS!r} but got {data[:8]!r}"
            )
            raise ValueError(msg)

        format: EnvelopeFormat = EnvelopeFormat(data[8])

        flags = data[9]
        zstd = bool(flags & _ZSTD_FLAG)
        other_flags = (flags ^ _DEFAULT_FLAGS) & ~_ZSTD_FLAG
        if other_flags:
            flag_ids = [i for i in range(8) if other_flags & (1 << i)]
            msg = (
                f"Unrecognised Envelope flags {flag_ids}."
                + " Please update your HUGR version."
            )
            raise ValueError(msg)

        return EnvelopeHeader(format=format, zstd=zstd)


@dataclass
class EnvelopeConfig:
    """Configuration for writing a HUGR envelope.

    Attributes:
        format: The format to use for encoding the HUGR.
        zstd: The compression level to use, or None for no compression.
            Use 0 for the default level.
    """

    format: EnvelopeFormat = EnvelopeFormat.JSON
    zstd: int | None = None

    TEXT: ClassVar[EnvelopeConfig]
    BINARY: ClassVar[EnvelopeConfig]

    def _make_header(self) -> EnvelopeHeader:
        return EnvelopeHeader(format=self.format, zstd=self.zstd is not None)


# Set EnvelopeConfig's class variables.
# These can only be initialized _after_ the class is defined.
EnvelopeConfig.TEXT = EnvelopeConfig(format=EnvelopeFormat.JSON, zstd=None)
EnvelopeConfig.BINARY = EnvelopeConfig(format=EnvelopeFormat.MODEL_WITH_EXTS, zstd=0)
