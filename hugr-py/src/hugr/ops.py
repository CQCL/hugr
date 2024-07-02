"""Definitions of HUGR operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import hugr.serialization.ops as sops
from hugr import tys, val
from hugr.node_port import Direction, InPort, Node, OutPort, Wire
from hugr.utils import ser_it

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hugr.serialization.ops import BaseOp


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
    """An abstract HUGR operation. Must be convertible
    to a serializable :class:`BaseOp`.
    """

    @property
    def num_out(self) -> int:
        """The number of output ports for this operation.

        Example:
            >>> op = Const(val.TRUE)
            >>> op.num_out
            1
        """
        ...  # pragma: no cover

    def to_serial(self, parent: Node) -> BaseOp:
        """Convert this operation to a serializable form."""
        ...  # pragma: no cover

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        """Get the kind of the given port.

        Example:
            >>> op = Const(val.TRUE)
            >>> op.port_kind(OutPort(Node(0), 0))
            ConstKind(Bool)
        """
        ...  # pragma: no cover

    def _invalid_port(self, port: InPort | OutPort) -> InvalidPort:
        return InvalidPort(port, self)


def _sig_port_type(sig: tys.FunctionType, port: InPort | OutPort) -> tys.Type:
    if port.direction == Direction.INCOMING:
        return sig.input[port.offset]
    return sig.output[port.offset]


@runtime_checkable
class DataflowOp(Op, Protocol):
    """Abstract dataflow operation. Can be assumed to have a signature and Value-
    kind ports.
    """

    def outer_signature(self) -> tys.FunctionType:
        """The external signature of this operation. Defines the valid external
        connectivity of the node the operation belongs to.
        """
        ...  # pragma: no cover

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        if port.offset == -1:
            return tys.OrderKind()
        return tys.ValueKind(self.port_type(port))

    def port_type(self, port: InPort | OutPort) -> tys.Type:
        """Get the type of the given dataflow port from the signature of the
        operation.

        Example:
            >>> op = Input([tys.Bool])
            >>> op.port_type(OutPort(Node(0), 0))
            Bool

        """
        return _sig_port_type(self.outer_signature(), port)

    def __call__(self, *args) -> Command:
        """Calling with incoming :class:`Wire` arguments returns a
        :class:`Command` which can be used to wire the operation into a
        dataflow graph.
        """
        return Command(self, list(args))


@runtime_checkable
class _PartialOp(Protocol):
    def _set_in_types(self, types: tys.TypeRow) -> None: ...


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
    """A :class:`DataflowOp` and its incoming :class:`Wire <hugr.nodeport.Wire>`
      arguments.

    Ephemeral: used to wire operations into a dataflow graph.

    Example:
        >>> Noop()(Node(0).out(0))
        Command(op=Noop, incoming=[OutPort(Node(0), 0)])
    """

    op: DataflowOp
    incoming: list[Wire]


@dataclass()
class Input(DataflowOp):
    """Input operation in dataflow graph. Outputs of this operation are the
    inputs to the graph.
    """

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
    """Output operation in dataflow graph. Inputs of this operation are the
    outputs of the graph.
    """

    _types: tys.TypeRow | None = field(default=None, repr=False)
    num_out: int = field(default=0, repr=False)

    @property
    def types(self) -> tys.TypeRow:
        return _check_complete(self, self._types)

    def to_serial(self, parent: Node) -> sops.Output:
        return sops.Output(parent=parent.idx, types=ser_it(self.types))

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=self.types, output=[])

    def _set_in_types(self, types: tys.TypeRow) -> None:
        self._types = types


@dataclass()
class Custom(DataflowOp):
    """A non-core dataflow operation defined in an extension."""

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
class MakeTuple(DataflowOp, _PartialOp):
    """Operation to create a tuple from a sequence of wires."""

    _types: tys.TypeRow | None = field(default=None, repr=False)
    num_out: int = field(default=1, repr=False)

    @property
    def types(self) -> tys.TypeRow:
        """If set, the types of the tuple elements.

        Raises:
            IncompleteOp: If the types have not been set.
        """
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

    def _set_in_types(self, types: tys.TypeRow) -> None:
        self._types = types

    def __repr__(self) -> str:
        return "MakeTuple" + (f"({self._types})" if self._types is not None else "")


@dataclass()
class UnpackTuple(DataflowOp, _PartialOp):
    """Operation to unpack a tuple into its elements."""

    _types: tys.TypeRow | None = field(default=None, repr=False)

    @property
    def types(self) -> tys.TypeRow:
        """If set, the types of the tuple elements.

        Raises:
            IncompleteOp: If the types have not been set.
        """
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
        return MakeTuple(self.types).outer_signature().flip()

    def _set_in_types(self, types: tys.TypeRow) -> None:
        (t,) = types
        assert isinstance(t, tys.Sum), f"Expected unary Sum, got {t}"
        (row,) = t.variant_rows
        self._types = row


@dataclass()
class Tag(DataflowOp):
    """Tag a row of incoming values to make them a variant of a sum type.

    Requires `sum_ty` to be set as it is not possible to extract all the variants from
    just the input wires for one variant.
    """

    tag: int
    sum_ty: tys.Sum
    num_out: int = field(default=1, repr=False)

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
    """Abstract parent of dataflow graph operations. Can be queried for the
    dataflow signature of its child graph.
    """

    def inner_signature(self) -> tys.FunctionType:
        """Inner signature of the child dataflow graph."""
        ...  # pragma: no cover

    def _set_out_types(self, types: tys.TypeRow) -> None: ...

    def _inputs(self) -> tys.TypeRow: ...


@dataclass
class DFG(DfParentOp, DataflowOp):
    """Simple dataflow graph operation. Outer signature matches inner signature."""

    #: Inputs types of the operation.
    inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = field(default=None, repr=False)
    _extension_delta: tys.ExtensionSet = field(default_factory=list, repr=False)

    @property
    def outputs(self) -> tys.TypeRow:
        """Output types of the operation.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
        return _check_complete(self, self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        """Signature of the operation.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
        return tys.FunctionType(self.inputs, self.outputs, self._extension_delta)

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
    """Parent operation of a control flow graph."""

    #: Inputs types of the operation.
    inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = field(default=None, repr=False)

    @property
    def outputs(self) -> tys.TypeRow:
        """Output types of the operation, if set.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
        return _check_complete(self, self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        """Dataflow signature of the CFG operation.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
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
    """Parent of non-entry basic block in a control flow graph."""

    #: Inputs types of the innner dataflow graph.
    inputs: tys.TypeRow
    _sum: tys.Sum | None = None
    _other_outputs: tys.TypeRow | None = field(default=None, repr=False)
    extension_delta: tys.ExtensionSet = field(default_factory=list)

    @property
    def sum_ty(self) -> tys.Sum:
        """If set, the sum type that defines the potential branching of the
        block.


        Raises:
            IncompleteOp: If the sum type has not been set.
        """
        return _check_complete(self, self._sum)

    @property
    def other_outputs(self) -> tys.TypeRow:
        """The non-branching outputs of the block which are passed to all
        successors.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
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
        """The outputs passed to the nth successor of the block.
        Concatenation of the nth variant of the sum type and the other outputs.
        """
        return [*self.sum_ty.variant_rows[n], *self.other_outputs]


@dataclass
class ExitBlock(Op):
    """Unique exit block of a control flow graph."""

    _cfg_outputs: tys.TypeRow | None = field(default=None, repr=False)
    num_out: int = field(default=0, repr=False)

    @property
    def cfg_outputs(self) -> tys.TypeRow:
        """Output types of the parent control flow graph of this exit block.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
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
    """A static constant value. Can be used with a :class:`LoadConst` to load into
    a dataflow graph.
    """

    val: val.Value
    num_out: int = field(default=1, repr=False)

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

    def __repr__(self) -> str:
        return f"Const({self.val})"


@dataclass
class LoadConst(DataflowOp):
    """Load a constant value into a dataflow graph. Connects to a :class:`Const`."""

    _typ: tys.Type | None = None
    num_out: int = field(default=1, repr=False)

    @property
    def type_(self) -> tys.Type:
        """The type of the loaded value.

        Raises:
            IncompleteOp: If the type has not been set.
        """
        return _check_complete(self, self._typ)

    def to_serial(self, parent: Node) -> sops.LoadConstant:
        return sops.LoadConstant(
            parent=parent.idx,
            datatype=self.type_.to_serial_root(),
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=[self.type_])

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case InPort(_, 0):
                return tys.ConstKind(self.type_)
            case OutPort(_, 0):
                return tys.ValueKind(self.type_)
            case _:
                raise self._invalid_port(port)

    def __repr__(self) -> str:
        return "LoadConst" + (f"({self._typ})" if self._typ is not None else "")


@dataclass()
class Conditional(DataflowOp):
    """'Switch' operation on the variants of an incoming sum type, evaluating the
    corresponding one of the child :class:`Case` operations.
    """

    #: Sum type to switch on.
    sum_ty: tys.Sum
    #: Non-sum inputs that are passed to all cases.
    other_inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = field(default=None, repr=False)

    @property
    def outputs(self) -> tys.TypeRow:
        """Outputs of the conditional, common to all cases.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
        return _check_complete(self, self._outputs)

    @property
    def signature(self) -> tys.FunctionType:
        """Dataflow signature of the conditional operation.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
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
        """The inputs passed to the nth child case.
        Concatenation of the nth variant of the sum type and the other inputs.
        """
        return [*self.sum_ty.variant_rows[n], *self.other_inputs]


@dataclass
class Case(DfParentOp):
    """Parent of a dataflow graph that is a branch of a :class:`Conditional`."""

    #: Inputs types of the innner dataflow graph.
    inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = field(default=None, repr=False)
    num_out: int = field(default=0, repr=False)

    @property
    def outputs(self) -> tys.TypeRow:
        """Outputs of the case operation.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
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
    """Tail controlled loop operation, child dataflow graph iterates while it
    outputs the first variant of a sum type.
    """

    #: Types that are only inputs of the child graph.
    just_inputs: tys.TypeRow
    #: Types that are appended to both inputs and outputs of the graph.
    rest: tys.TypeRow
    _just_outputs: tys.TypeRow | None = field(default=None, repr=False)
    extension_delta: tys.ExtensionSet = field(default_factory=list, repr=False)

    @property
    def just_outputs(self) -> tys.TypeRow:
        """Types that are only outputs of the child graph.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
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
    """Function definition operation, parent of a dataflow graph that defines
    the function.
    """

    #: function name
    name: str
    #: input types of the function
    inputs: tys.TypeRow
    # ? type parameters of the function if polymorphic
    params: list[tys.TypeParam] = field(default_factory=list)
    _outputs: tys.TypeRow | None = field(default=None, repr=False)
    num_out: int = field(default=1, repr=False)

    @property
    def outputs(self) -> tys.TypeRow:
        """Output types of the function.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
        return _check_complete(self, self._outputs)

    @property
    def signature(self) -> tys.PolyFuncType:
        """Polymorphic signature of the function.

        Raises:
            IncompleteOp: If the outputs have not been set.
        """
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
    """Function declaration operation, defines the signature of a function."""

    #: function name
    name: str
    #: polymorphic function signature
    signature: tys.PolyFuncType
    num_out: int = field(default=1, repr=False)

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
    """Root operation of a HUGR which corresponds to a full module definition."""

    num_out: int = field(default=0, repr=False)

    def to_serial(self, parent: Node) -> sops.Module:
        return sops.Module(parent=parent.idx)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)


class NoConcreteFunc(Exception):
    """Could not instantiate a polymorphic function."""


@dataclass
class _CallOrLoad:
    #: polymorphic function signature
    signature: tys.PolyFuncType
    #: concrete function signature
    instantiation: tys.FunctionType
    #: type arguments for polymorphic function
    type_args: list[tys.TypeArg]

    def __init__(
        self,
        signature: tys.PolyFuncType,
        instantiation: tys.FunctionType | None = None,
        type_args: Sequence[tys.TypeArg] | None = None,
    ) -> None:
        self.signature = signature

        if len(signature.params) == 0:
            self.instantiation = signature.body
            self.type_args = []

        else:
            # TODO substitute type args into signature to get instantiation
            if instantiation is None:
                msg = "Missing instantiation for polymorphic function."
                raise NoConcreteFunc(msg)
            type_args = type_args or []

            if len(signature.params) != len(type_args):
                msg = "Mismatched number of type arguments."
                raise NoConcreteFunc(msg)
            self.instantiation = instantiation
            self.type_args = list(type_args)


class Call(_CallOrLoad, Op):
    """Call a function inside a dataflow graph. Connects to :class:`FuncDefn` or
    :class:`FuncDecl` operations.

    Args:
        signature: Polymorphic function signature.
        instantiation: Concrete function signature. Defaults to None.
        type_args: Type arguments for polymorphic function. Defaults to None.

    Raises:
        NoConcreteFunc: If the signature is polymorphic and no instantiation
            is provided.
    """

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

    def _function_port_offset(self) -> int:
        return len(self.signature.body.input)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case InPort(_, offset) if offset == self._function_port_offset():
                return tys.FunctionKind(self.signature)
            case _:
                return tys.ValueKind(_sig_port_type(self.instantiation, port))


@dataclass()
class CallIndirect(DataflowOp, _PartialOp):
    """Higher order evaluation of a
    :class:`FunctionType <hugr.tys.FunctionType>` value.
    """

    _signature: tys.FunctionType | None = None

    @property
    def num_out(self) -> int:
        return len(self.signature.output)

    @property
    def signature(self) -> tys.FunctionType:
        """The signature of the function being called.

        Raises:
        IncompleteOp: If the signature has not been set.
        """
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

    def _set_in_types(self, types: tys.TypeRow) -> None:
        func_sig, *_ = types
        assert isinstance(
            func_sig, tys.FunctionType
        ), f"Expected function type, got {func_sig}"
        self._signature = func_sig


class LoadFunc(_CallOrLoad, DataflowOp):
    """Load a statically defined function as a higher order value.
      Connects to :class:`FuncDefn` or :class:`FuncDecl` operations.

    Args:
        signature: Polymorphic function signature.
        instantiation: Concrete function signature. Defaults to None.
        type_args: Type arguments for polymorphic function. Defaults to None.

    Raises:
        NoConcreteFunc: If the signature is polymorphic and no instantiation
            is provided.
    """

    num_out: int = field(default=1, repr=False)

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
class Noop(DataflowOp, _PartialOp):
    """Identity operation that passes through its input."""

    _type: tys.Type | None = None
    num_out: int = field(default=1, repr=False)

    @property
    def type_(self) -> tys.Type:
        """The type of the input and output of the operation."""
        return _check_complete(self, self._type)

    def to_serial(self, parent: Node) -> sops.Noop:
        return sops.Noop(parent=parent.idx, ty=self.type_.to_serial_root())

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType.endo([self.type_])

    def _set_in_types(self, types: tys.TypeRow) -> None:
        (t,) = types
        self._type = t

    def __repr__(self) -> str:
        return "Noop" + (f"({self._type})" if self._type is not None else "")


@dataclass
class Lift(DataflowOp, _PartialOp):
    """Add an extension requirement to input values and pass them through."""

    #: Extension added.
    new_extension: tys.ExtensionId
    _type_row: tys.TypeRow | None = field(default=None, repr=False)
    num_out: int = field(default=1, repr=False)

    @property
    def type_row(self) -> tys.TypeRow:
        """Types of the input and output of the operation."""
        return _check_complete(self, self._type_row)

    def to_serial(self, parent: Node) -> sops.Lift:
        return sops.Lift(
            parent=parent.idx,
            new_extension=self.new_extension,
            type_row=ser_it(self.type_row),
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType.endo(self.type_row)

    def _set_in_types(self, types: tys.TypeRow) -> None:
        self._type_row = types


@dataclass
class AliasDecl(Op):
    """Declare an external type alias."""

    #: Alias name.
    name: str
    #: Type bound.
    bound: tys.TypeBound
    num_out: int = field(default=0, repr=False)

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
    """Declare a type alias."""

    #: Alias name.
    name: str
    #: Type definition.
    definition: tys.Type
    num_out: int = field(default=0, repr=False)

    def to_serial(self, parent: Node) -> sops.AliasDefn:
        return sops.AliasDefn(
            parent=parent.idx,
            name=self.name,
            definition=self.definition.to_serial_root(),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)
