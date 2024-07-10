"""HUGR logic operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import tys
from hugr.ops import Command, Custom

if TYPE_CHECKING:
    from hugr.ops import ComWire


@dataclass(frozen=True)
class LogicOps(Custom):
    """Base class for logic operations."""

    extension: tys.ExtensionId = "logic"


_NotSig = tys.FunctionType.endo([tys.Bool])


@dataclass(frozen=True)
class _NotDef(LogicOps):
    """Not operation."""

    num_out: int = 1
    op_name: str = "Not"
    signature: tys.FunctionType = _NotSig

    def __call__(self, a: ComWire) -> Command:
        return super().__call__(a)


#: Not operation
Not = _NotDef()
