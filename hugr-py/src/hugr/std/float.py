"""Floating point types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from semver import Version

from hugr import ext, tys, val

Extension = ext.Extension("arithmetic.float.types", Version(0, 1, 0))
#: HUGR 64-bit IEEE 754-2019 floating point type.
FLOAT_T_DEF = Extension.add_type_def(
    ext.TypeDef(
        name="float64",
        description="64-bit IEEE 754-2019 floating point number",
        params=[],
        bound=ext.ExplicitBound(tys.TypeBound.Copyable),
    )
)

FLOAT_T = FLOAT_T_DEF.instantiate([])


@dataclass
class FloatVal(val.ExtensionValue):
    """Custom value for a floating point number."""

    v: float

    def to_value(self) -> val.Extension:
        return val.Extension("float", FLOAT_T, self.v)
