"""Floating point types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from hugr import val
from hugr.std import _load_extension

EXTENSION = _load_extension("arithmetic.float.types")

FLOAT_T = EXTENSION.types["float64"].instantiate([])


@dataclass
class FloatVal(val.ExtensionValue):
    """Custom value for a floating point number."""

    v: float

    def to_value(self) -> val.Extension:
        name = "ConstF64"
        payload = {"value": self.v}
        return val.Extension(
            name, typ=FLOAT_T, val=payload, extensions=[EXTENSION.name]
        )
