"""HUGR logic operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import tys
from hugr.ops import AsCustomOp, Command, Custom, DataflowOp

if TYPE_CHECKING:
    from hugr.ops import ComWire


EXTENSION_ID: tys.ExtensionId = "logic"


@dataclass(frozen=True)
class _NotDef(AsCustomOp):
    """Not operation."""

    def to_custom(self) -> Custom:
        return Custom("Not", tys.FunctionType.endo([tys.Bool]), extension=EXTENSION_ID)

    def __call__(self, a: ComWire) -> Command:
        return DataflowOp.__call__(self, a)


#: Not operation
Not = _NotDef()
