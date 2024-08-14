"""HUGR logic operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import ext, tys
from hugr.ops import Command, DataflowOp, RegisteredOp

if TYPE_CHECKING:
    from hugr.ops import ComWire


EXTENSION = ext.Extension("logic", ext.Version(0, 1, 0))


@EXTENSION.register_op(
    name="Not",
    signature=ext.OpDefSig(tys.FunctionType.endo([tys.Bool])),
)
@dataclass(frozen=True)
class _NotOp(RegisteredOp):
    """Logical NOT operation."""

    def __call__(self, a: ComWire) -> Command:
        return DataflowOp.__call__(self, a)


#: Not operation
Not = _NotOp()
