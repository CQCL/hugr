from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, TYPE_CHECKING
from hugr.serialization.ops import BaseOp
import hugr.serialization.ops as sops
import hugr.serialization.tys as tys

if TYPE_CHECKING:
    from hugr._hugr import Hugr, Node, Wire


class Op(Protocol):
    @property
    def num_out(self) -> int | None:
        return None

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> BaseOp: ...

    def __call__(self, *args) -> Command:
        return Command(self, list(args))


@dataclass(frozen=True)
class Command:
    op: Op
    incoming: list[Wire]


T = TypeVar("T", bound=BaseOp)


@dataclass()
class SerWrap(Op, Generic[T]):
    # catch all for serial ops that don't have a corresponding Op class
    _serial_op: T

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> T:
        root = self._serial_op.model_copy()
        root.parent = parent.idx
        return root


@dataclass()
class Input(Op):
    types: list[tys.Type]

    @property
    def num_out(self) -> int | None:
        return len(self.types)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Input:
        return sops.Input(parent=parent.idx, types=self.types)

    def __call__(self) -> Command:
        return super().__call__()


@dataclass()
class Output(Op):
    types: list[tys.Type]

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Output:
        return sops.Output(parent=parent.idx, types=self.types)


@dataclass()
class Custom(Op):
    op_name: str
    signature: tys.FunctionType = field(default_factory=tys.FunctionType.empty)
    description: str = ""
    extension: tys.ExtensionId = ""
    args: list[tys.TypeArg] = field(default_factory=list)

    @property
    def num_out(self) -> int | None:
        return len(self.signature.output)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.CustomOp:
        return sops.CustomOp(
            parent=parent.idx,
            extension=self.extension,
            op_name=self.op_name,
            signature=self.signature,
            description=self.description,
            args=self.args,
        )


@dataclass()
class MakeTuple(Op):
    types: list[tys.Type]
    num_out: int | None = 1

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.MakeTuple:
        return sops.MakeTuple(
            parent=parent.idx,
            tys=self.types,
        )

    def __call__(self, *elements: Wire) -> Command:
        return super().__call__(*elements)


@dataclass()
class UnpackTuple(Op):
    types: list[tys.Type]

    @property
    def num_out(self) -> int | None:
        return len(self.types)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.UnpackTuple:
        return sops.UnpackTuple(
            parent=parent.idx,
            tys=self.types,
        )

    def __call__(self, tuple_: Wire) -> Command:
        return super().__call__(tuple_)


@dataclass()
class DFG(Op):
    signature: tys.FunctionType = field(default_factory=tys.FunctionType.empty)

    @property
    def num_out(self) -> int | None:
        return len(self.signature.output)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.DFG:
        return sops.DFG(
            parent=parent.idx,
            signature=self.signature,
        )
