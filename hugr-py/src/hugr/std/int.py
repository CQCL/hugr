"""HUGR integer types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from typing_extensions import Self

from hugr import tys, val
from hugr.ops import AsCustomOp, Custom, DataflowOp

if TYPE_CHECKING:
    from hugr.ops import Command, ComWire


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
        bound=tys.TypeBound.Copyable,
    )


#: HUGR 32-bit integer type.
INT_T = int_t(5)


@dataclass
class IntVal(val.ExtensionValue):
    """Custom value for an integer."""

    v: int

    def to_value(self) -> val.Extension:
        return val.Extension("int", INT_T, self.v)


OPS_EXTENSION: tys.ExtensionId = "arithmetic.int"


@dataclass(frozen=True)
class _DivModDef(AsCustomOp):
    """DivMod operation, has two inputs and two outputs."""

    name: ClassVar[str] = "idivmod_u"
    arg1: int = 5
    arg2: int = 5

    def to_custom(self) -> Custom:
        return Custom(
            "idivmod_u",
            tys.FunctionType(
                input=[int_t(self.arg1)] * 2, output=[int_t(self.arg2)] * 2
            ),
            extension=OPS_EXTENSION,
            args=[tys.BoundedNatArg(n=self.arg1), tys.BoundedNatArg(n=self.arg2)],
        )

    @classmethod
    def from_custom(cls, custom: Custom) -> Self | None:
        if not custom.check_id(OPS_EXTENSION, "idivmod_u"):
            return None
        match custom.args:
            case [tys.BoundedNatArg(n=a1), tys.BoundedNatArg(n=a2)]:
                return cls(arg1=a1, arg2=a2)
            case _:
                msg = f"Invalid args: {custom.args}"
                raise AsCustomOp.InvalidCustomOp(msg)

    def __call__(self, a: ComWire, b: ComWire) -> Command:
        return DataflowOp.__call__(self, a, b)


#: DivMod operation.
DivMod = _DivModDef()
