"""HUGR integer types and operations."""

from __future__ import annotations

from dataclasses import dataclass, field

from hugr import tys, val
from hugr.ops import Custom


def int_t(width: int) -> tys.Opaque:
    """Create an integer type with a given log bit width.


    Args:
        width: The log bit width of the integer.

    Returns:
        The integer type.

    Examples:
        >>> int_t(5).id # 32 bit integer
        'int'
    """
    return tys.Opaque(
        extension="arithmetic.int.types",
        id="int",
        args=[tys.BoundedNatArg(n=width)],
        bound=tys.TypeBound.Eq,
    )


#: HUGR 32-bit integer type.
INT_T = int_t(5)


@dataclass
class IntVal(val.ExtensionValue):
    """Custom value for an integer."""

    v: int

    def to_value(self) -> val.Extension:
        return val.Extension("int", INT_T, self.v)


@dataclass(frozen=True)
class IntOps(Custom):
    """Base class for integer operations."""

    extension: tys.ExtensionId = "arithmetic.int"


_ARG_I32 = tys.BoundedNatArg(n=5)


@dataclass(frozen=True)
class _DivModDef(IntOps):
    """DivMod operation, has two inputs and two outputs."""

    num_out: int = 2
    extension: tys.ExtensionId = "arithmetic.int"
    op_name: str = "idivmod_u"
    signature: tys.FunctionType = field(
        default_factory=lambda: tys.FunctionType(input=[INT_T] * 2, output=[INT_T] * 2)
    )
    args: list[tys.TypeArg] = field(default_factory=lambda: [_ARG_I32, _ARG_I32])


#: DivMod operation.
DivMod = _DivModDef()
