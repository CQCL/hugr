from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING, runtime_checkable, TypeVar
from hugr.serialization.ops import BaseOp
import hugr.serialization.ops as sops
from hugr.utils import ser_it
import hugr._tys as tys
import hugr._val as val
from ._exceptions import IncompleteOp

if TYPE_CHECKING:
    from hugr._hugr import Hugr, Node, Wire, InPort, OutPort


@runtime_checkable
class Op(Protocol):
    @property
    def num_out(self) -> int | None:
        return None

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> BaseOp: ...

    def port_kind(self, port: InPort | OutPort) -> tys.Kind: ...


@runtime_checkable
class DataflowOp(Op, Protocol):
    def outer_signature(self) -> tys.FunctionType: ...

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        if port.offset == -1:
            return tys.OrderKind()
        return tys.ValueKind(self.port_type(port))

    def port_type(self, port: InPort | OutPort) -> tys.Type:
        from hugr._hugr import Direction

        sig = self.outer_signature()
        if port.direction == Direction.INCOMING:
            return sig.input[port.offset]
        return sig.output[port.offset]

    def __call__(self, *args) -> Command:
        return Command(self, list(args))


@runtime_checkable
class PartialOp(Protocol):
    def set_in_types(self, types: tys.TypeRow) -> None: ...


@dataclass(frozen=True)
class Command:
    op: DataflowOp
    incoming: list[Wire]


@dataclass()
class Input(DataflowOp):
    types: tys.TypeRow

    @property
    def num_out(self) -> int | None:
        return len(self.types)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Input:
        return sops.Input(parent=parent.idx, types=ser_it(self.types))

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=self.types)

    def __call__(self) -> Command:
        return super().__call__()


V = TypeVar("V")


def _check_complete(v: V | None) -> V:
    if v is None:
        raise IncompleteOp()
    return v


@dataclass()
class Output(DataflowOp, PartialOp):
    _types: tys.TypeRow | None = None

    @property
    def types(self) -> tys.TypeRow:
        return _check_complete(self._types)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Output:
        return sops.Output(parent=parent.idx, types=ser_it(self.types))

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=self.types, output=[])

    def set_in_types(self, types: tys.TypeRow) -> None:
        self._types = types


@dataclass()
class Custom(DataflowOp):
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

    def outer_signature(self) -> tys.FunctionType:
        return self.signature


@dataclass()
class MakeTupleDef(DataflowOp, PartialOp):
    _types: tys.TypeRow | None = None
    num_out: int | None = 1

    @property
    def types(self) -> tys.TypeRow:
        return _check_complete(self._types)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.MakeTuple:
        return sops.MakeTuple(
            parent=parent.idx,
            tys=ser_it(self.types),
        )

    def __call__(self, *elements: Wire) -> Command:
        return super().__call__(*elements)

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=self.types, output=[tys.Tuple(*self.types)])

    def set_in_types(self, types: tys.TypeRow) -> None:
        self._types = types


MakeTuple = MakeTupleDef()


@dataclass()
class UnpackTupleDef(DataflowOp, PartialOp):
    _types: tys.TypeRow | None = None

    @property
    def types(self) -> tys.TypeRow:
        return _check_complete(self._types)

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

    def outer_signature(self) -> tys.FunctionType:
        return MakeTupleDef(self.types).outer_signature().flip()

    def set_in_types(self, types: tys.TypeRow) -> None:
        (t,) = types
        assert isinstance(t, tys.Sum), f"Expected unary Sum, got {t}"
        (row,) = t.variant_rows
        self._types = row


UnpackTuple = UnpackTupleDef()


@dataclass()
class Tag(DataflowOp):
    tag: int
    sum_ty: tys.Sum
    num_out: int | None = 1

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Tag:
        return sops.Tag(
            parent=parent.idx,
            tag=self.tag,
            variants=[ser_it(r) for r in self.sum_ty.variant_rows],
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(
            input=self.sum_ty.variant_rows[self.tag], output=[self.sum_ty]
        )


class DfParentOp(Op, Protocol):
    def inner_signature(self) -> tys.FunctionType: ...

    def _set_out_types(self, types: tys.TypeRow) -> None: ...

    def _inputs(self) -> tys.TypeRow: ...


@dataclass
class DFG(DfParentOp, DataflowOp):
    inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = None

    @property
    def outputs(self) -> tys.TypeRow:
        return _check_complete(self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        return tys.FunctionType(self.inputs, self.outputs)

    @property
    def num_out(self) -> int | None:
        return len(self.signature.output)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.DFG:
        return sops.DFG(
            parent=parent.idx,
            signature=self.signature.to_serial(),
        )

    def inner_signature(self) -> tys.FunctionType:
        return self.signature

    def outer_signature(self) -> tys.FunctionType:
        return self.signature

    def _set_out_types(self, types: tys.TypeRow) -> None:
        self._outputs = types

    def _inputs(self) -> tys.TypeRow:
        return self.inputs


@dataclass()
class CFG(DataflowOp):
    inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = None

    @property
    def outputs(self) -> tys.TypeRow:
        return _check_complete(self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        return tys.FunctionType(self.inputs, self.outputs)

    @property
    def num_out(self) -> int | None:
        return len(self.outputs)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.CFG:
        return sops.CFG(
            parent=parent.idx,
            signature=self.signature.to_serial(),
        )

    def outer_signature(self) -> tys.FunctionType:
        return self.signature


@dataclass
class DataflowBlock(DfParentOp):
    inputs: tys.TypeRow
    _sum: tys.Sum | None = None
    _other_outputs: tys.TypeRow | None = None
    extension_delta: tys.ExtensionSet = field(default_factory=list)

    @property
    def sum_ty(self) -> tys.Sum:
        return _check_complete(self._sum)

    @property
    def other_outputs(self) -> tys.TypeRow:
        return _check_complete(self._other_outputs)

    @property
    def num_out(self) -> int | None:
        return len(self.sum_ty.variant_rows)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.DataflowBlock:
        return sops.DataflowBlock(
            parent=parent.idx,
            inputs=ser_it(self.inputs),
            sum_rows=list(map(ser_it, self.sum_ty.variant_rows)),
            other_outputs=ser_it(self.other_outputs),
            extension_delta=self.extension_delta,
        )

    def inner_signature(self) -> tys.FunctionType:
        return tys.FunctionType(
            input=self.inputs, output=[self.sum_ty, *self.other_outputs]
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        return tys.CFKind()

    def _set_out_types(self, types: tys.TypeRow) -> None:
        (sum_, other) = tys.get_first_sum(types)
        self._sum = sum_
        self._other_outputs = other

    def _inputs(self) -> tys.TypeRow:
        return self.inputs

    def nth_outputs(self, n: int) -> tys.TypeRow:
        return [*self.sum_ty.variant_rows[n], *self.other_outputs]


@dataclass
class ExitBlock(Op):
    _cfg_outputs: tys.TypeRow | None = None
    num_out: int | None = 0

    @property
    def cfg_outputs(self) -> tys.TypeRow:
        return _check_complete(self._cfg_outputs)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.ExitBlock:
        return sops.ExitBlock(
            parent=parent.idx,
            cfg_outputs=ser_it(self.cfg_outputs),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        return tys.CFKind()


@dataclass
class Const(Op):
    val: val.Value
    num_out: int | None = 1

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Const:
        return sops.Const(
            parent=parent.idx,
            v=self.val.to_serial_root(),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        return tys.ConstKind(self.val.type_())


@dataclass
class LoadConst(DataflowOp):
    typ: tys.Type | None = None

    def type_(self) -> tys.Type:
        return _check_complete(self.typ)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.LoadConstant:
        return sops.LoadConstant(
            parent=parent.idx,
            datatype=self.type_().to_serial_root(),
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=[self.type_()])


@dataclass()
class Conditional(DataflowOp):
    sum_ty: tys.Sum
    other_inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = None

    @property
    def outputs(self) -> tys.TypeRow:
        return _check_complete(self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        inputs = [self.sum_ty, *self.other_inputs]
        return tys.FunctionType(inputs, self.outputs)

    @property
    def num_out(self) -> int | None:
        return len(self.outputs)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Conditional:
        return sops.Conditional(
            parent=parent.idx,
            sum_rows=[ser_it(r) for r in self.sum_ty.variant_rows],
            other_inputs=ser_it(self.other_inputs),
            outputs=ser_it(self.outputs),
        )

    def outer_signature(self) -> tys.FunctionType:
        return self.signature

    def nth_inputs(self, n: int) -> tys.TypeRow:
        return [*self.sum_ty.variant_rows[n], *self.other_inputs]


@dataclass
class Case(DfParentOp):
    inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = None

    @property
    def outputs(self) -> tys.TypeRow:
        return _check_complete(self._outputs)

    @property
    def num_out(self) -> int | None:
        return 0

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.Case:
        return sops.Case(
            parent=parent.idx, signature=self.inner_signature().to_serial()
        )

    def inner_signature(self) -> tys.FunctionType:
        return tys.FunctionType(self.inputs, self.outputs)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise NotImplementedError("Case nodes have no external ports.")

    def _set_out_types(self, types: tys.TypeRow) -> None:
        self._outputs = types

    def _inputs(self) -> tys.TypeRow:
        return self.inputs


@dataclass
class TailLoop(DfParentOp, DataflowOp):
    just_inputs: tys.TypeRow
    rest: tys.TypeRow
    _just_outputs: tys.TypeRow | None = None
    extension_delta: tys.ExtensionSet = field(default_factory=list)

    @property
    def just_outputs(self) -> tys.TypeRow:
        return _check_complete(self._just_outputs)

    @property
    def num_out(self) -> int | None:
        return len(self.just_outputs) + len(self.rest)

    def to_serial(self, node: Node, parent: Node, hugr: Hugr) -> sops.TailLoop:
        return sops.TailLoop(
            parent=parent.idx,
            just_inputs=ser_it(self.just_inputs),
            just_outputs=ser_it(self.just_outputs),
            rest=ser_it(self.rest),
            extension_delta=self.extension_delta,
        )

    def inner_signature(self) -> tys.FunctionType:
        return tys.FunctionType(
            self._inputs(), [tys.Sum([self.just_inputs, self.just_outputs]), *self.rest]
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(self._inputs(), self.just_outputs + self.rest)

    def _set_out_types(self, types: tys.TypeRow) -> None:
        (sum_, other) = tys.get_first_sum(types)
        just_ins, just_outs = sum_.variant_rows
        assert (
            just_ins == self.just_inputs
        ), "First sum variant rows don't match TailLoop inputs."
        self._just_outputs = just_outs

    def _inputs(self) -> tys.TypeRow:
        return self.just_inputs + self.rest
