from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable, TypeVar
from hugr.serialization.ops import BaseOp
import hugr.serialization.ops as sops
from hugr.utils import ser_it
import hugr.tys as tys
from hugr.node_port import Node, InPort, OutPort, Wire
import hugr.val as val


@dataclass
class InvalidPort(Exception):
    """Port is not valid for this operation."""

    port: InPort | OutPort
    op: Op

    @property
    def msg(self) -> str:
        return f"Port {self.port} is invalid for operation {self.op}."


@runtime_checkable
class Op(Protocol):
    @property
    def num_out(self) -> int: ...

    def to_serial(self, parent: Node) -> BaseOp: ...

    def port_kind(self, port: InPort | OutPort) -> tys.Kind: ...

    def _invalid_port(self, port: InPort | OutPort) -> InvalidPort:
        return InvalidPort(port, self)


def _sig_port_type(sig: tys.FunctionType, port: InPort | OutPort) -> tys.Type:
    from hugr.node_port import Direction

    if port.direction == Direction.INCOMING:
        return sig.input[port.offset]
    return sig.output[port.offset]


@runtime_checkable
class DataflowOp(Op, Protocol):
    def outer_signature(self) -> tys.FunctionType: ...

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        if port.offset == -1:
            return tys.OrderKind()
        return tys.ValueKind(self.port_type(port))

    def port_type(self, port: InPort | OutPort) -> tys.Type:
        return _sig_port_type(self.outer_signature(), port)

    def __call__(self, *args) -> Command:
        return Command(self, list(args))


@runtime_checkable
class _PartialOp(Protocol):
    def set_in_types(self, types: tys.TypeRow) -> None: ...


@dataclass
class IncompleteOp(Exception):
    """Op types have not been set during building."""

    op: Op

    @property
    def msg(self) -> str:
        return (
            f"Operation {self.op} is incomplete, may require set_in_types to be called."
        )


V = TypeVar("V")


def _check_complete(op, v: V | None) -> V:
    if v is None:
        raise IncompleteOp(op)
    return v


@dataclass(frozen=True)
class Command:
    op: DataflowOp
    incoming: list[Wire]


@dataclass()
class Input(DataflowOp):
    types: tys.TypeRow

    @property
    def num_out(self) -> int:
        return len(self.types)

    def to_serial(self, parent: Node) -> sops.Input:
        return sops.Input(parent=parent.idx, types=ser_it(self.types))

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=self.types)

    def __call__(self) -> Command:
        return super().__call__()


@dataclass()
class Output(DataflowOp, _PartialOp):
    _types: tys.TypeRow | None = None
    num_out: int = 0

    @property
    def types(self) -> tys.TypeRow:
        return _check_complete(self, self._types)

    def to_serial(self, parent: Node) -> sops.Output:
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
    def num_out(self) -> int:
        return len(self.signature.output)

    def to_serial(self, parent: Node) -> sops.CustomOp:
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
class MakeTupleDef(DataflowOp, _PartialOp):
    _types: tys.TypeRow | None = None
    num_out: int = 1

    @property
    def types(self) -> tys.TypeRow:
        return _check_complete(self, self._types)

    def to_serial(self, parent: Node) -> sops.MakeTuple:
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
class UnpackTupleDef(DataflowOp, _PartialOp):
    _types: tys.TypeRow | None = None

    @property
    def types(self) -> tys.TypeRow:
        return _check_complete(self, self._types)

    @property
    def num_out(self) -> int:
        return len(self.types)

    def to_serial(self, parent: Node) -> sops.UnpackTuple:
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
    num_out: int = 1

    def to_serial(self, parent: Node) -> sops.Tag:
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
    extension_delta: tys.ExtensionSet = field(default_factory=list)

    @property
    def outputs(self) -> tys.TypeRow:
        return _check_complete(self, self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        return tys.FunctionType(self.inputs, self.outputs, self.extension_delta)

    @property
    def num_out(self) -> int:
        return len(self.signature.output)

    def to_serial(self, parent: Node) -> sops.DFG:
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
        return _check_complete(self, self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        return tys.FunctionType(self.inputs, self.outputs)

    @property
    def num_out(self) -> int:
        return len(self.outputs)

    def to_serial(self, parent: Node) -> sops.CFG:
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
        return _check_complete(self, self._sum)

    @property
    def other_outputs(self) -> tys.TypeRow:
        return _check_complete(self, self._other_outputs)

    @property
    def num_out(self) -> int:
        return len(self.sum_ty.variant_rows)

    def to_serial(self, parent: Node) -> sops.DataflowBlock:
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
    num_out: int = 0

    @property
    def cfg_outputs(self) -> tys.TypeRow:
        return _check_complete(self, self._cfg_outputs)

    def to_serial(self, parent: Node) -> sops.ExitBlock:
        return sops.ExitBlock(
            parent=parent.idx,
            cfg_outputs=ser_it(self.cfg_outputs),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        return tys.CFKind()


@dataclass
class Const(Op):
    val: val.Value
    num_out: int = 1

    def to_serial(self, parent: Node) -> sops.Const:
        return sops.Const(
            parent=parent.idx,
            v=self.val.to_serial_root(),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case OutPort(_, 0):
                return tys.ConstKind(self.val.type_())
            case _:
                raise self._invalid_port(port)


@dataclass
class LoadConst(DataflowOp):
    typ: tys.Type | None = None
    num_out: int = 1

    def type_(self) -> tys.Type:
        return _check_complete(self, self.typ)

    def to_serial(self, parent: Node) -> sops.LoadConstant:
        return sops.LoadConstant(
            parent=parent.idx,
            datatype=self.type_().to_serial_root(),
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=[self.type_()])

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case InPort(_, 0):
                return tys.ConstKind(self.type_())
            case OutPort(_, 0):
                return tys.ValueKind(self.type_())
            case _:
                raise self._invalid_port(port)


@dataclass()
class Conditional(DataflowOp):
    sum_ty: tys.Sum
    other_inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = None

    @property
    def outputs(self) -> tys.TypeRow:
        return _check_complete(self, self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        inputs = [self.sum_ty, *self.other_inputs]
        return tys.FunctionType(inputs, self.outputs)

    @property
    def num_out(self) -> int:
        return len(self.outputs)

    def to_serial(self, parent: Node) -> sops.Conditional:
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
    num_out: int = 0

    @property
    def outputs(self) -> tys.TypeRow:
        return _check_complete(self, self._outputs)

    def to_serial(self, parent: Node) -> sops.Case:
        return sops.Case(
            parent=parent.idx, signature=self.inner_signature().to_serial()
        )

    def inner_signature(self) -> tys.FunctionType:
        return tys.FunctionType(self.inputs, self.outputs)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)

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
        return _check_complete(self, self._just_outputs)

    @property
    def num_out(self) -> int:
        return len(self.just_outputs) + len(self.rest)

    def to_serial(self, parent: Node) -> sops.TailLoop:
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


@dataclass
class FuncDefn(DfParentOp):
    name: str
    inputs: tys.TypeRow
    params: list[tys.TypeParam] = field(default_factory=list)
    _outputs: tys.TypeRow | None = None
    num_out: int = 1

    @property
    def outputs(self) -> tys.TypeRow:
        return _check_complete(self, self._outputs)

    @property
    def signature(self) -> tys.PolyFuncType:
        return tys.PolyFuncType(
            self.params, tys.FunctionType(self.inputs, self.outputs)
        )

    def to_serial(self, parent: Node) -> sops.FuncDefn:
        return sops.FuncDefn(
            parent=parent.idx,
            name=self.name,
            signature=self.signature.to_serial(),
        )

    def inner_signature(self) -> tys.FunctionType:
        return self.signature.body

    def _set_out_types(self, types: tys.TypeRow) -> None:
        self._outputs = types

    def _inputs(self) -> tys.TypeRow:
        return self.inputs

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case OutPort(_, 0):
                return tys.FunctionKind(self.signature)
            case _:
                raise self._invalid_port(port)


@dataclass
class FuncDecl(Op):
    name: str
    signature: tys.PolyFuncType
    num_out: int = 0

    def to_serial(self, parent: Node) -> sops.FuncDecl:
        return sops.FuncDecl(
            parent=parent.idx,
            name=self.name,
            signature=self.signature.to_serial(),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case OutPort(_, 0):
                return tys.FunctionKind(self.signature)
            case _:
                raise self._invalid_port(port)


@dataclass
class Module(Op):
    num_out: int = 0

    def to_serial(self, parent: Node) -> sops.Module:
        return sops.Module(parent=parent.idx)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)


class NoConcreteFunc(Exception):
    pass


def _fn_instantiation(
    signature: tys.PolyFuncType,
    instantiation: tys.FunctionType | None = None,
    type_args: Sequence[tys.TypeArg] | None = None,
) -> tuple[tys.FunctionType, list[tys.TypeArg]]:
    if len(signature.params) == 0:
        return signature.body, []

    else:
        # TODO substitute type args into signature to get instantiation
        if instantiation is None:
            raise NoConcreteFunc("Missing instantiation for polymorphic function.")
        type_args = type_args or []

        if len(signature.params) != len(type_args):
            raise NoConcreteFunc("Mismatched number of type arguments.")
        return instantiation, list(type_args)


@dataclass
class Call(Op):
    signature: tys.PolyFuncType
    instantiation: tys.FunctionType
    type_args: list[tys.TypeArg]

    def __init__(
        self,
        signature: tys.PolyFuncType,
        instantiation: tys.FunctionType | None = None,
        type_args: Sequence[tys.TypeArg] | None = None,
    ) -> None:
        self.signature = signature
        self.instantiation, self.type_args = _fn_instantiation(
            signature, instantiation, type_args
        )

    def to_serial(self, parent: Node) -> sops.Call:
        return sops.Call(
            parent=parent.idx,
            func_sig=self.signature.to_serial(),
            type_args=ser_it(self.type_args),
            instantiation=self.instantiation.to_serial(),
        )

    @property
    def num_out(self) -> int:
        return len(self.signature.body.output)

    def function_port_offset(self) -> int:
        return len(self.signature.body.input)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case InPort(_, offset) if offset == self.function_port_offset():
                return tys.FunctionKind(self.signature)
            case _:
                return tys.ValueKind(_sig_port_type(self.instantiation, port))


@dataclass()
class CallIndirectDef(DataflowOp, _PartialOp):
    _signature: tys.FunctionType | None = None

    @property
    def num_out(self) -> int:
        return len(self.signature.output)

    @property
    def signature(self) -> tys.FunctionType:
        return _check_complete(self, self._signature)

    def to_serial(self, parent: Node) -> sops.CallIndirect:
        return sops.CallIndirect(
            parent=parent.idx,
            signature=self.signature.to_serial(),
        )

    def __call__(self, function: Wire, *args: Wire) -> Command:  # type: ignore[override]
        return super().__call__(function, *args)

    def outer_signature(self) -> tys.FunctionType:
        sig = self.signature

        return tys.FunctionType(input=[sig, *sig.input], output=sig.output)

    def set_in_types(self, types: tys.TypeRow) -> None:
        func_sig, *_ = types
        assert isinstance(
            func_sig, tys.FunctionType
        ), f"Expected function type, got {func_sig}"
        self._signature = func_sig


# rename to eval?
CallIndirect = CallIndirectDef()


@dataclass
class LoadFunc(DataflowOp):
    signature: tys.PolyFuncType
    instantiation: tys.FunctionType
    type_args: list[tys.TypeArg]
    num_out: int = 1

    def __init__(
        self,
        signature: tys.PolyFuncType,
        instantiation: tys.FunctionType | None = None,
        type_args: Sequence[tys.TypeArg] | None = None,
    ) -> None:
        self.signature = signature
        self.instantiation, self.type_args = _fn_instantiation(
            signature, instantiation, type_args
        )

    def to_serial(self, parent: Node) -> sops.LoadFunction:
        return sops.LoadFunction(
            parent=parent.idx,
            func_sig=self.signature.to_serial(),
            type_args=ser_it(self.type_args),
            signature=self.outer_signature().to_serial(),
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=[self.instantiation])

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case InPort(_, 0):
                return tys.FunctionKind(self.signature)
            case OutPort(_, 0):
                return tys.ValueKind(self.instantiation)
            case _:
                raise self._invalid_port(port)


@dataclass
class NoopDef(DataflowOp, _PartialOp):
    _type: tys.Type | None = None
    num_out: int = 1

    @property
    def type_(self) -> tys.Type:
        return _check_complete(self, self._type)

    def to_serial(self, parent: Node) -> sops.Noop:
        return sops.Noop(parent=parent.idx, ty=self.type_.to_serial_root())

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType.endo([self.type_])

    def set_in_types(self, types: tys.TypeRow) -> None:
        (t,) = types
        self._type = t


Noop = NoopDef()


@dataclass
class Lift(DataflowOp, _PartialOp):
    new_extension: tys.ExtensionId
    _type_row: tys.TypeRow | None = None
    num_out: int = 1

    @property
    def type_row(self) -> tys.TypeRow:
        return _check_complete(self, self._type_row)

    def to_serial(self, parent: Node) -> sops.Lift:
        return sops.Lift(
            parent=parent.idx,
            new_extension=self.new_extension,
            type_row=ser_it(self.type_row),
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType.endo(self.type_row)

    def set_in_types(self, types: tys.TypeRow) -> None:
        self._type_row = types


@dataclass
class AliasDecl(Op):
    name: str
    bound: tys.TypeBound
    num_out: int = 0

    def to_serial(self, parent: Node) -> sops.AliasDecl:
        return sops.AliasDecl(
            parent=parent.idx,
            name=self.name,
            bound=self.bound,
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)


@dataclass
class AliasDefn(Op):
    name: str
    definition: tys.Type
    num_out: int = 0

    def to_serial(self, parent: Node) -> sops.AliasDefn:
        return sops.AliasDefn(
            parent=parent.idx,
            name=self.name,
            definition=self.definition.to_serial_root(),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)
