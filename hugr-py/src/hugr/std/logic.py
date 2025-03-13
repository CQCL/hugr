"""HUGR logic operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from hugr.ops import Command, DataflowOp, RegisteredOp
from hugr.std import _load_extension

if TYPE_CHECKING:
    from hugr import ext
    from hugr.ops import ComWire


EXTENSION = _load_extension("logic")


@dataclass(frozen=True)
class _NotOp(RegisteredOp):
    """Logical NOT operation."""

    const_op_def: ClassVar[ext.OpDef] = EXTENSION.operations["Not"]

    def __call__(self, a: ComWire) -> Command:
        return DataflowOp.__call__(self, a)


#: Not operation
Not = _NotOp()
