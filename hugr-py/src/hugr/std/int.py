"""HUGR integer types and operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Self

from hugr import ext, tys, val
from hugr.ops import AsExtOp, DataflowOp, ExtOp, RegisteredOp

if TYPE_CHECKING:
    from hugr.ops import Command, ComWire

INT_TYPES_EXTENSION = ext.Extension("arithmetic.int.types", ext.Version(0, 1, 0))
_INT_PARAM = tys.BoundedNatParam(7)
INT_T_DEF = INT_TYPES_EXTENSION.add_type_def(
    ext.TypeDef(
        name="int",
        description="Variable-width integer.",
        bound=ext.ExplicitBound(tys.TypeBound.Copyable),
        params=[_INT_PARAM],
    )
)


def int_t(width: int) -> tys.ExtType:
    """Create an integer type with a fixed log bit width.


    Args:
        width: The log bit width of the integer.

    Returns:
        The integer type.

    Examples:
        >>> int_t(5).type_def.name # 32 bit integer
        'int'
    """
    return INT_T_DEF.instantiate(
        [tys.BoundedNatArg(n=width)],
    )


def _int_tv(index: int) -> tys.ExtType:
    return INT_T_DEF.instantiate(
        [tys.VariableArg(idx=index, param=_INT_PARAM)],
    )


#: HUGR 32-bit integer type.
INT_T = int_t(5)


@dataclass
class IntVal(val.ExtensionValue):
    """Custom value for an integer."""

    v: int
    width: int = field(default=5)

    def to_value(self) -> val.Extension:
        name = "ConstInt"
        payload = {"log_width": self.width, "value": self.v}
        return val.Extension(
            name,
            typ=int_t(self.width),
            val=payload,
            extensions=[INT_TYPES_EXTENSION.name],
        )


INT_OPS_EXTENSION = ext.Extension("arithmetic.int", ext.Version(0, 1, 0))


@INT_OPS_EXTENSION.register_op(
    signature=ext.OpDefSig(
        tys.FunctionType([_int_tv(0), _int_tv(1)], [_int_tv(0), _int_tv(1)])
    ),
)
@dataclass(frozen=True)
class idivmod_u(RegisteredOp):
    """DivMod operation, has two inputs and two outputs."""

    arg1: int = 5
    arg2: int = 5

    def type_args(self) -> list[tys.TypeArg]:
        return [tys.BoundedNatArg(n=self.arg1), tys.BoundedNatArg(n=self.arg2)]

    def cached_signature(self) -> tys.FunctionType | None:
        row: list[tys.Type] = [int_t(self.arg1), int_t(self.arg2)]
        return tys.FunctionType.endo(row)

    @classmethod
    def from_ext(cls, custom: ExtOp) -> Self | None:
        if custom.op_def() != cls.const_op_def:
            return None
        match custom.args:
            case [tys.BoundedNatArg(n=a1), tys.BoundedNatArg(n=a2)]:
                return cls(arg1=a1, arg2=a2)
            case _:
                msg = f"Invalid args: {custom.args}"
                raise AsExtOp.InvalidExtOp(msg)

    def __call__(self, a: ComWire, b: ComWire) -> Command:
        return DataflowOp.__call__(self, a, b)


#: DivMod operation.
DivMod = idivmod_u()
