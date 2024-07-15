"""Floating point types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from hugr import tys, val

#: HUGR 64-bit IEEE 754-2019 floating point type.
FLOAT_EXT_ID = "arithmetic.float.types"
FLOAT_T = tys.Opaque(
    extension=FLOAT_EXT_ID,
    id="float64",
    args=[],
    bound=tys.TypeBound.Copyable,
)


@dataclass
class FloatVal(val.ExtensionValue):
    """Custom value for a floating point number."""

    v: float

    def to_value(self) -> val.Extension:
        return val.Extension("float", FLOAT_T, self.v)
