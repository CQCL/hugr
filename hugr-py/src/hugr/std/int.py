"""HUGR integer types and operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from typing_extensions import Self

import hugr.model as model
from hugr import ext, tys, val
from hugr.ops import AsExtOp, DataflowOp, ExtOp, RegisteredOp
from hugr.std import _load_extension

if TYPE_CHECKING:
    from hugr.ops import Command, ComWire

CONVERSIONS_EXTENSION = _load_extension("arithmetic.conversions")

INT_TYPES_EXTENSION = _load_extension("arithmetic.int.types")
_INT_PARAM = tys.BoundedNatParam(7)

INT_T_DEF = INT_TYPES_EXTENSION.types["int"]


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
        )

    def __str__(self) -> str:
        return f"{self.v}"

    def to_model(self) -> model.Term:
        return model.Apply(
            "arithmetic.int.const", [model.Literal(self.width), model.Literal(self.v)]
        )


INT_OPS_EXTENSION = _load_extension("arithmetic.int")


@dataclass(frozen=True)
class _DivModDef(RegisteredOp):
    """DivMod operation, has two inputs and two outputs."""

    width: int = 5
    const_op_def: ClassVar[ext.OpDef] = INT_OPS_EXTENSION.operations["idivmod_u"]

    def type_args(self) -> list[tys.TypeArg]:
        return [tys.BoundedNatArg(n=self.width)]

    def cached_signature(self) -> tys.FunctionType | None:
        row: list[tys.Type] = [int_t(self.width)] * 2
        return tys.FunctionType.endo(row)

    @classmethod
    def from_ext(cls, custom: ExtOp) -> Self | None:
        if custom.op_def() != cls.op_def():
            return None
        match custom.args:
            case [tys.BoundedNatArg(n=a1)]:
                return cls(width=a1)
            case _:
                msg = f"Invalid args: {custom.args}"
                raise AsExtOp.InvalidExtOp(msg)

    def __call__(self, a: ComWire, b: ComWire) -> Command:
        return DataflowOp.__call__(self, a, b)


#: DivMod operation.
DivMod = _DivModDef()
