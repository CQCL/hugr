"""HUGR logic operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import ext, tys
from hugr.ops import AsExtOp, Command, DataflowOp, ExtOp

if TYPE_CHECKING:
    from hugr.ops import ComWire


EXTENSION = ext.Extension("logic", ext.Version(0, 1, 0))

_NotDef = EXTENSION.add_op_def(
    ext.OpDef(
        name="Not",
        description="Logical NOT operation.",
        signature=ext.OpDefSig(tys.FunctionType.endo([tys.Bool])),
    )
)


@dataclass(frozen=True)
class _NotOp(AsExtOp):
    def to_ext(self) -> ExtOp:
        return ExtOp(_NotDef)

    def __call__(self, a: ComWire) -> Command:
        return DataflowOp.__call__(self, a)


#: Not operation
Not = _NotOp()
