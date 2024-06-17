from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, TYPE_CHECKING
from hugr.serialization.ops import BaseOp
import hugr.serialization.ops as sops
from hugr.utils import ser_it
import hugr._tys as tys

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
    types: tys.TypeRow

    @property
    def num_out(self) -> int | None:
        return len(self.types)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Input:
        return sops.Input(parent=parent.idx, types=ser_it(self.types))

    def __call__(self) -> Command:
        return super().__call__()


@dataclass()
class Output(Op):
    types: tys.TypeRow

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Output:
        return sops.Output(parent=parent.idx, types=ser_it(self.types))


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
            signature=self.signature.to_serial(),
            description=self.description,
            args=ser_it(self.args),
        )


@dataclass()
class MakeTuple(Op):
    types: tys.TypeRow
    num_out: int | None = 1

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.MakeTuple:
        return sops.MakeTuple(
            parent=parent.idx,
            tys=ser_it(self.types),
        )

    def __call__(self, *elements: Wire) -> Command:
        return super().__call__(*elements)


@dataclass()
class UnpackTuple(Op):
    types: tys.TypeRow

    @property
    def num_out(self) -> int | None:
        return len(self.types)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.UnpackTuple:
        return sops.UnpackTuple(
            parent=parent.idx,
            tys=ser_it(self.types),
        )

    def __call__(self, tuple_: Wire) -> Command:
        return super().__call__(tuple_)


class DfParentOp(Op, Protocol):
    def input_types(self) -> tys.TypeRow: ...
    def output_types(self) -> tys.TypeRow: ...


@dataclass()
class DFG(DfParentOp):
    signature: tys.FunctionType = field(default_factory=tys.FunctionType.empty)

    @property
    def num_out(self) -> int | None:
        return len(self.signature.output)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.DFG:
        return sops.DFG(
            parent=parent.idx,
            signature=self.signature.to_serial(),
        )

    def input_types(self) -> tys.TypeRow:
        return self.signature.input

    def output_types(self) -> tys.TypeRow:
        return self.signature.output


@dataclass()
class CFG(Op):
    signature: tys.FunctionType = field(default_factory=tys.FunctionType.empty)

    @property
    def num_out(self) -> int | None:
        return len(self.signature.output)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.CFG:
        return sops.CFG(
            parent=parent.idx,
            signature=self.signature.to_serial(),
        )


@dataclass
class DataflowBlock(DfParentOp):
    inputs: tys.TypeRow
    sum_rows: list[tys.TypeRow]
    other_outputs: tys.TypeRow = field(default_factory=list)
    extension_delta: tys.ExtensionSet = field(default_factory=list)

    @property
    def num_out(self) -> int | None:
        return len(self.sum_rows)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.DataflowBlock:
        return sops.DataflowBlock(
            parent=parent.idx,
            inputs=ser_it(self.inputs),
            sum_rows=list(map(ser_it, self.sum_rows)),
            other_outputs=ser_it(self.other_outputs),
            extension_delta=self.extension_delta,
        )

    def input_types(self) -> tys.TypeRow:
        return self.inputs

    def output_types(self) -> tys.TypeRow:
        return [tys.Sum(self.sum_rows), *self.other_outputs]


@dataclass
class ExitBlock(Op):
    cfg_outputs: tys.TypeRow
    num_out: int | None = 0

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.ExitBlock:
        return sops.ExitBlock(
            parent=parent.idx,
            cfg_outputs=ser_it(self.cfg_outputs),
        )
