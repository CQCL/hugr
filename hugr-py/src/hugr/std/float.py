"""Floating point types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from hugr import ext, tys, val

EXTENSION = ext.Extension("arithmetic.float.types", ext.Version(0, 1, 0))
#: HUGR 64-bit IEEE 754-2019 floating point type.
FLOAT_T_DEF = EXTENSION.add_type_def(
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
        name = "ConstF64"
        payload = {"value": self.v}
        return val.Extension(
            name, typ=FLOAT_T, val=payload, extensions=[EXTENSION.name]
        )
