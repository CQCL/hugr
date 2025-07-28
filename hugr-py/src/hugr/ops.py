"""Definitions of HUGR operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import Self

import hugr._serialization.ops as sops
from hugr import tys, val
from hugr.hugr.node_port import Direction, InPort, Node, OutPort, PortOffset, Wire
from hugr.utils import comma_sep_str, ser_it

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hugr import ext
    from hugr._serialization.ops import BaseOp


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

    def _to_serial(self, parent: Node) -> BaseOp:
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

    def name(self) -> str:
        """Name of the operation."""
        return str(self)


@runtime_checkable
class DataflowOp(Op, Protocol):
    """Abstract dataflow operation. Can be assumed to have a signature and Value-
    kind ports.
    """

    #: When initializing a Hugr with a dataflow operation
    #: a function is defined in the root module containing the op,
    #: marked as entrypoint.
    #: If the operation's output types are only known _after_ the
    #: HUGR is defined, we need to wire up the function containing
    #: the entrypoint as soon as the outputs are set.
    #:
    #: This flag is set to True for such cases. It should never be set
    #: manually.
    _entrypoint_requires_wiring: bool = field(
        init=False, repr=False, default=False, compare=False
    )

    def _inputs(self) -> tys.TypeRow:
        """The external input row of this operation. Defines the valid external
        connectivity of the node the operation belongs to.

        Raises:
            IncompleteOp: If the operation's inputs have not been set.
        """
        ...  # pragma: no cover

    def outer_signature(self) -> tys.FunctionType:
        """The external signature of this operation. Defines the valid external
        connectivity of the node the operation belongs to.

        Raises:
            IncompleteOp: If the operation's inputs and outputs have not been set.
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
        sig = self.outer_signature()
        if port.offset == -1:
            # Order port
            msg = "Order port has no type."
            raise ValueError(msg)
        try:
            if port.direction == Direction.INCOMING:
                return sig.input[port.offset]
            return sig.output[port.offset]
        except IndexError as e:
            raise self._invalid_port(port) from e

    def __call__(self, *args) -> Command:
        """Calling with incoming :class:`Wire` arguments returns a
        :class:`Command` which can be used to wire the operation into a
        dataflow graph.
        """
        return Command(self, list(args))


def is_dataflow_op(op: Any) -> TypeGuard[DataflowOp]:
    """Returns `true` if the object is an instance of :class:`DataflowOp`.

    This is functionally equivalent to matching on `DataflowOp()` directly, but
    calling `isinstance(_, DataflowOp)` errors out in `python <=3.11` due to how
    runtime_checkable Protocols were implemented. See <https://github.com/python/cpython/issues/102433>
    """
    match op:
        case (
            Custom()
            | Tag()
            | DFG()
            | CFG()
            | LoadConst()
            | Conditional()
            | TailLoop()
            | CallIndirect()
            | LoadFunc()
            | AsExtOp()
        ):
            return True
        case _:
            return False


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


ComWire = Wire | int


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
    incoming: list[ComWire]


@dataclass()
class Input(DataflowOp):
    """Input operation in dataflow graph. Outputs of this operation are the
    inputs to the graph.
    """

    types: tys.TypeRow

    @property
    def num_out(self) -> int:
        return len(self.types)

    def _to_serial(self, parent: Node) -> sops.Input:
        return sops.Input(parent=parent.idx, types=ser_it(self.types))

    def _inputs(self) -> tys.TypeRow:
        return []

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=self.types)

    def __call__(self, *args) -> Command:
        return super().__call__()

    def name(self) -> str:
        return "Input"


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

    def _to_serial(self, parent: Node) -> sops.Output:
        return sops.Output(parent=parent.idx, types=ser_it(self.types))

    def _inputs(self) -> tys.TypeRow:
        return self.types

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=self.types, output=[])

    def _set_in_types(self, types: tys.TypeRow) -> None:
        self._types = types

    def name(self) -> str:
        return "Output"


@runtime_checkable
class AsExtOp(DataflowOp, Protocol):
    """Abstract interface that types can implement
    to behave as an extension dataflow operation.
    """

    @dataclass(frozen=True)
    class InvalidExtOp(Exception):
        """Extension operation does not match the expected type."""

        msg: str

    @cached_property
    def ext_op(self) -> ExtOp:
        """:class:`ExtOp` operation that this type represents.

        Computed once using :meth:`op_def` :meth:`type_args` and :meth:`type_args`.
        Each of those methods should be deterministic.
        """
        return self.op_def().instantiate(self.type_args(), self.cached_signature())

    def op_def(self) -> ext.OpDef:
        """The :class:`tys.OpDef` for this operation.


        Used by :attr:`ext_op`, so must be deterministic.
        """
        ...  # pragma: no cover

    def type_args(self) -> list[tys.TypeArg]:
        """Type arguments of the operation.

        Used by :attr:`op_def`, so must be deterministic.
        """
        return []

    def cached_signature(self) -> tys.FunctionType | None:
        """Cached signature of the operation, if there is one.


        Used by :attr:`op_def`, so must be deterministic.
        """
        return None

    @classmethod
    def from_ext(cls, ext_op: ExtOp) -> Self | None:
        """Load from a :class:`ExtOp` operation.


        By default assumes the type of `cls` is a singleton,
        and compares the result of :meth:`to_ext` with the given `ext_op`.

        If successful, returns the singleton, else None.

        Non-singleton types should override this method.

        Raises:
            InvalidCustomOp: If the given `ext_op` does not match the expected one for a
            given extension/operation name.
        """
        default = cls()
        if default.ext_op == ext_op:
            return default
        return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AsExtOp):
            return NotImplemented
        slf, other = self.ext_op, other.ext_op
        return (
            slf._op_def == other._op_def
            and slf.outer_signature() == other.outer_signature()
            and slf.args == other.args
        )

    def _inputs(self) -> tys.TypeRow:
        return self.outer_signature().input

    def outer_signature(self) -> tys.FunctionType:
        return self.ext_op.outer_signature()

    def _to_serial(self, parent: Node) -> sops.ExtensionOp:
        return self.ext_op._to_serial(parent)

    @property
    def num_out(self) -> int:
        return len(self.outer_signature().output)

    def name(self) -> str:
        name = self.ext_op._op_def.qualified_name()
        ta = self.type_args()
        if len(ta) == 0:
            return name
        return f"{name}<{comma_sep_str(self.type_args())}>"


@dataclass(frozen=True, eq=False)
class Custom(DataflowOp):
    """Serializable version of non-core dataflow operation defined in an extension."""

    op_name: str
    signature: tys.FunctionType = field(default_factory=tys.FunctionType.empty)
    extension: tys.ExtensionId = ""
    args: list[tys.TypeArg] = field(default_factory=list)

    def _to_serial(self, parent: Node) -> sops.ExtensionOp:
        return sops.ExtensionOp(
            parent=parent.idx,
            extension=self.extension,
            name=self.op_name,
            signature=self.signature._to_serial(),
            args=ser_it(self.args),
        )

    def _inputs(self) -> tys.TypeRow:
        return self.signature.input

    def outer_signature(self) -> tys.FunctionType:
        return self.signature

    @property
    def num_out(self) -> int:
        return len(self.outer_signature().output)

    def check_id(self, extension: tys.ExtensionId, name: str) -> bool:
        """Check if the operation matches the given extension and operation name."""
        return self.extension == extension and self.op_name == name

    def resolve(self, registry: ext.ExtensionRegistry) -> ExtOp | Custom:
        """Resolve the custom operation to an :class:`ExtOp`.

        If extension or operation is not found, returns itself.
        """
        from hugr.ext import ExtensionRegistry, Extension  # noqa: I001 # no circular import

        try:
            op_def = registry.get_extension(self.extension).get_op(self.op_name)
        except (
            Extension.OperationNotFound,
            ExtensionRegistry.ExtensionNotFound,
        ):
            return self

        signature = self.signature.resolve(registry)
        args = [arg.resolve(registry) for arg in self.args]
        # TODO check signature matches op_def reported signature
        # if/once op_def can compute signature from type scheme + args
        return ExtOp(op_def, signature, args)

    def name(self) -> str:
        return f"Custom({self.op_name})"


@dataclass(frozen=True, eq=False)
class ExtOp(AsExtOp):
    """A non-core dataflow operation defined in an extension."""

    _op_def: ext.OpDef
    signature: tys.FunctionType | None = None
    args: list[tys.TypeArg] = field(default_factory=list)

    def to_custom_op(self) -> Custom:
        ext = self._op_def._extension
        if self.signature is None:
            poly_func = self._op_def.signature.poly_func
            if poly_func is None or len(poly_func.params) > 0:
                msg = "For polymorphic ops signature must be cached."
                raise ValueError(msg)
            sig = poly_func.body
        else:
            sig = self.signature

        return Custom(
            op_name=self._op_def.name,
            signature=sig,
            extension=ext.name if ext else "",
            args=self.args,
        )

    def _to_serial(self, parent: Node) -> sops.ExtensionOp:
        return self.to_custom_op()._to_serial(parent)

    def op_def(self) -> ext.OpDef:
        return self._op_def

    def type_args(self) -> list[tys.TypeArg]:
        return self.args

    def cached_signature(self) -> tys.FunctionType | None:
        return self.signature

    @classmethod
    def from_ext(cls, custom: ExtOp) -> ExtOp:
        return custom

    def _inputs(self) -> tys.TypeRow:
        if self.signature is None:
            raise IncompleteOp(self)
        return self.signature.input

    def outer_signature(self) -> tys.FunctionType:
        if self.signature is not None:
            return self.signature
        poly_func = self._op_def.signature.poly_func
        if poly_func is None:
            msg = "Polymorphic signature must be cached."
            raise ValueError(msg)
        return poly_func.body


class RegisteredOp(AsExtOp):
    """Base class for operations that are registered with an extension using
    :meth:`Extension.register_op <hugr.ext.Extension.register_op>`.
    """

    #: Known operation definition.
    const_op_def: ClassVar[ext.OpDef]  # may be set by registered_op decorator.

    @classmethod
    def op_def(cls) -> ext.OpDef:
        # override for AsExtOp.op_def
        return cls.const_op_def


@dataclass()
class MakeTuple(AsExtOp, _PartialOp):
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

    def _inputs(self) -> tys.TypeRow:
        return self.types

    def op_def(self) -> ext.OpDef:
        from hugr import std  # no circular import

        return std.PRELUDE.get_op("MakeTuple")

    def cached_signature(self) -> tys.FunctionType | None:
        return tys.FunctionType(
            input=self.types,
            output=[tys.Tuple(*self.types)],
        )

    def type_args(self) -> list[tys.TypeArg]:
        return [tys.SequenceArg([t.type_arg() for t in self.types])]

    def __call__(self, *elements: ComWire) -> Command:
        return super().__call__(*elements)

    def _set_in_types(self, types: tys.TypeRow) -> None:
        self._types = types

    def __repr__(self) -> str:
        return "MakeTuple" + (f"({self._types})" if self._types is not None else "")

    def name(self) -> str:
        return "MakeTuple"


@dataclass()
class UnpackTuple(AsExtOp, _PartialOp):
    """Operation to unpack a tuple into its elements."""

    _types: tys.TypeRow | None = field(default=None, repr=False)

    @property
    def types(self) -> tys.TypeRow:
        """If set, the types of the tuple elements.

        Raises:
            IncompleteOp: If the types have not been set.
        """
        return _check_complete(self, self._types)

    def op_def(self) -> ext.OpDef:
        from hugr import std  # no circular import

        return std.PRELUDE.get_op("UnpackTuple")

    def cached_signature(self) -> tys.FunctionType | None:
        return tys.FunctionType(
            input=[tys.Tuple(*self.types)],
            output=self.types,
        )

    def type_args(self) -> list[tys.TypeArg]:
        return [tys.SequenceArg([t.type_arg() for t in self.types])]

    @property
    def num_out(self) -> int:
        return len(self.types)

    def __call__(self, tuple_: ComWire) -> Command:
        return super().__call__(tuple_)

    def _inputs(self) -> tys.TypeRow:
        return MakeTuple(self.types).outer_signature().output

    def outer_signature(self) -> tys.FunctionType:
        return MakeTuple(self.types).outer_signature().flip()

    def _set_in_types(self, types: tys.TypeRow) -> None:
        (t,) = types
        assert isinstance(t, tys.Sum), f"Expected unary Sum, got {t}"
        (row,) = t.variant_rows
        self._types = row

    def __repr__(self) -> str:
        return "UnpackTuple" + (f"({self._types})" if self._types is not None else "")

    def name(self) -> str:
        return "UnpackTuple"


@dataclass()
class Tag(DataflowOp):
    """Tag a row of incoming values to make them a variant of a sum type.

    Requires `sum_ty` to be set as it is not possible to extract all the variants from
    just the input wires for one variant.
    """

    tag: int
    sum_ty: tys.Sum
    num_out: int = field(default=1, repr=False)

    def _to_serial(self, parent: Node) -> sops.Tag:
        return sops.Tag(
            parent=parent.idx,
            tag=self.tag,
            variants=[ser_it(r) for r in self.sum_ty.variant_rows],
        )

    def _inputs(self) -> tys.TypeRow:
        return self.sum_ty.variant_rows[self.tag]

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(
            input=self.sum_ty.variant_rows[self.tag], output=[self.sum_ty]
        )

    def __repr__(self) -> str:
        return f"Tag({self.tag})"


@dataclass
class Some(Tag):
    """Tag operation for the `Some` variant of an Option type.

    Example:
        # construct a Some variant holding a row of Bool and Unit types
        >>> Some(tys.Bool, tys.Unit)
        Some
    """

    def __init__(self, *some_tys: tys.Type) -> None:
        super().__init__(1, tys.Option(*some_tys))

    def __repr__(self) -> str:
        return "Some"


@dataclass
class Right(Tag):
    """Tag operation for the `Right` variant of an type."""

    def __init__(self, either_type: tys.Either) -> None:
        super().__init__(1, either_type)

    def __repr__(self) -> str:
        return "Right"


@dataclass
class Left(Tag):
    """Tag operation for the `Left` variant of an type."""

    def __init__(self, either_type: tys.Either) -> None:
        super().__init__(0, either_type)

    def __repr__(self) -> str:
        return "Left"


class Continue(Left):
    """Tag operation for the `Continue` variant of a TailLoop
    controlling Either type.
    """

    def __repr__(self) -> str:
        return "Continue"


class Break(Right):
    """Tag operation for the `Break` variant of a TailLoop controlling Either type."""

    def __repr__(self) -> str:
        return "Break"


class DfParentOp(Op, Protocol):
    """Abstract parent of dataflow graph operations. Can be queried for the
    dataflow signature of its child graph.
    """

    def inner_signature(self) -> tys.FunctionType:
        """Inner signature of the child dataflow graph."""
        ...  # pragma: no cover

    def _set_out_types(self, types: tys.TypeRow) -> None: ...

    def _inputs(self) -> tys.TypeRow: ...


def is_df_parent_op(op: Any) -> TypeGuard[DfParentOp]:
    """Returns `true` if the object is an instance of :class:`DfParentOp`.

    This is functionally equivalent to matching on `DfParentOp()` directly, but
    calling `isinstance(_, DfParentOp)` errors out in `python <=3.11` due to how
    runtime_checkable Protocols were implemented.
    See <https://github.com/python/cpython/issues/102433>
    """
    match op:
        case DFG() | DataflowBlock() | Case() | TailLoop() | FuncDefn():
            return True
        case _:
            return False


@dataclass
class DFG(DfParentOp, DataflowOp):
    """Simple dataflow graph operation. Outer signature matches inner signature."""

    #: Inputs types of the operation.
    inputs: tys.TypeRow
    _outputs: tys.TypeRow | None = field(default=None, repr=False)

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
        return tys.FunctionType(self.inputs, self.outputs)

    @property
    def num_out(self) -> int:
        return len(self.signature.output)

    def _to_serial(self, parent: Node) -> sops.DFG:
        return sops.DFG(
            parent=parent.idx,
            signature=self.signature._to_serial(),
        )

    def inner_signature(self) -> tys.FunctionType:
        return self.signature

    def outer_signature(self) -> tys.FunctionType:
        return self.signature

    def _set_out_types(self, types: tys.TypeRow) -> None:
        self._outputs = types

    def _inputs(self) -> tys.TypeRow:
        return self.inputs

    def name(self) -> str:
        return "DFG"


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

    def _to_serial(self, parent: Node) -> sops.CFG:
        return sops.CFG(
            parent=parent.idx,
            signature=self.signature._to_serial(),
        )

    def outer_signature(self) -> tys.FunctionType:
        return self.signature

    def name(self) -> str:
        return "CFG"

    def _inputs(self) -> tys.TypeRow:
        return self.inputs


@dataclass
class DataflowBlock(DfParentOp):
    """Parent of non-entry basic block in a control flow graph."""

    #: Inputs types of the inner dataflow graph.
    inputs: tys.TypeRow
    _sum: tys.Sum | None = None
    _other_outputs: tys.TypeRow | None = field(default=None, repr=False)

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

    def _to_serial(self, parent: Node) -> sops.DataflowBlock:
        return sops.DataflowBlock(
            parent=parent.idx,
            inputs=ser_it(self.inputs),
            sum_rows=list(map(ser_it, self.sum_ty.variant_rows)),
            other_outputs=ser_it(self.other_outputs),
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

    def name(self) -> str:
        return "DataflowBlock"


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

    def _to_serial(self, parent: Node) -> sops.ExitBlock:
        return sops.ExitBlock(
            parent=parent.idx,
            cfg_outputs=ser_it(self.cfg_outputs),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        return tys.CFKind()

    def name(self) -> str:
        return "ExitBlock"


@dataclass
class Const(Op):
    """A static constant value. Can be used with a :class:`LoadConst` to load into
    a dataflow graph.
    """

    val: val.Value
    num_out: int = field(default=1, repr=False)

    def _to_serial(self, parent: Node) -> sops.Const:
        return sops.Const(
            parent=parent.idx,
            v=self.val._to_serial_root(),
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

    def _to_serial(self, parent: Node) -> sops.LoadConstant:
        return sops.LoadConstant(
            parent=parent.idx,
            datatype=self.type_._to_serial_root(),
        )

    def _inputs(self) -> tys.TypeRow:
        return []

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=[self.type_])

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case InPort(_, 0):
                return tys.ConstKind(self.type_)
            case _:
                return DataflowOp.port_kind(self, port)

    def __repr__(self) -> str:
        return "LoadConst" + (f"({self._typ})" if self._typ is not None else "")

    def name(self) -> str:
        return "LoadConst"


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
        return tys.FunctionType(self._inputs(), self.outputs)

    @property
    def num_out(self) -> int:
        return len(self.outputs)

    def _to_serial(self, parent: Node) -> sops.Conditional:
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

    def name(self) -> str:
        return "Conditional"

    def _inputs(self) -> tys.TypeRow:
        """Input row of the outer signature."""
        return [self.sum_ty, *self.other_inputs]


@dataclass
class Case(DfParentOp):
    """Parent of a dataflow graph that is a branch of a :class:`Conditional`."""

    #: Inputs types of the inner dataflow graph.
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

    def _to_serial(self, parent: Node) -> sops.Case:
        return sops.Case(
            parent=parent.idx, signature=self.inner_signature()._to_serial()
        )

    def inner_signature(self) -> tys.FunctionType:
        return tys.FunctionType(self.inputs, self.outputs)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)

    def _set_out_types(self, types: tys.TypeRow) -> None:
        self._outputs = types

    def _inputs(self) -> tys.TypeRow:
        return self.inputs

    def name(self) -> str:
        return "Case"


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

    def _to_serial(self, parent: Node) -> sops.TailLoop:
        return sops.TailLoop(
            parent=parent.idx,
            just_inputs=ser_it(self.just_inputs),
            just_outputs=ser_it(self.just_outputs),
            rest=ser_it(self.rest),
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

    def name(self) -> str:
        return "TailLoop"


@dataclass
class FuncDefn(DfParentOp):
    """Function definition operation, parent of a dataflow graph that defines
    the function.
    """

    #: function name
    f_name: str
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

    def _to_serial(self, parent: Node) -> sops.FuncDefn:
        return sops.FuncDefn(
            parent=parent.idx,
            name=self.f_name,
            signature=self.signature._to_serial(),
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

    def name(self) -> str:
        return f"FuncDefn({self.f_name})"


@dataclass
class FuncDecl(Op):
    """Function declaration operation, defines the signature of a function."""

    #: function name
    f_name: str
    #: polymorphic function signature
    signature: tys.PolyFuncType
    num_out: int = field(default=1, repr=False)

    def _to_serial(self, parent: Node) -> sops.FuncDecl:
        return sops.FuncDecl(
            parent=parent.idx,
            name=self.f_name,
            signature=self.signature._to_serial(),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case OutPort(_, 0):
                return tys.FunctionKind(self.signature)
            case _:
                raise self._invalid_port(port)

    def name(self) -> str:
        return f"FuncDecl({self.f_name})"


@dataclass
class Module(Op):
    """Root operation of a HUGR which corresponds to a full module definition."""

    num_out: int = field(default=0, repr=False)

    def _to_serial(self, parent: Node) -> sops.Module:
        return sops.Module(parent=parent.idx)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)

    def name(self) -> str:
        return "Module"


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


class Call(_CallOrLoad, DataflowOp):
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

    def _to_serial(self, parent: Node) -> sops.Call:
        return sops.Call(
            parent=parent.idx,
            func_sig=self.signature._to_serial(),
            type_args=ser_it(self.type_args),
            instantiation=self.instantiation._to_serial(),
        )

    @property
    def num_out(self) -> int:
        return len(self.signature.body.output)

    def _function_port_offset(self) -> PortOffset:
        return len(self.signature.body.input)

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case InPort(_, offset) if offset == self._function_port_offset():
                return tys.FunctionKind(self.signature)
            case _:
                return DataflowOp.port_kind(self, port)

    def name(self) -> str:
        return "Call"

    def _inputs(self) -> tys.TypeRow:
        return self.instantiation.input

    def outer_signature(self) -> tys.FunctionType:
        return self.instantiation


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

    def _to_serial(self, parent: Node) -> sops.CallIndirect:
        return sops.CallIndirect(
            parent=parent.idx,
            signature=self.signature._to_serial(),
        )

    def __call__(self, function: ComWire, *args: ComWire) -> Command:  # type: ignore[override]
        return super().__call__(function, *args)

    def _inputs(self) -> tys.TypeRow:
        sig = self.signature
        return [sig, *sig.input]

    def outer_signature(self) -> tys.FunctionType:
        sig = self.signature

        return tys.FunctionType(input=[sig, *sig.input], output=sig.output)

    def _set_in_types(self, types: tys.TypeRow) -> None:
        func_sig, *_ = types
        assert isinstance(
            func_sig, tys.FunctionType
        ), f"Expected function type, got {func_sig}"
        self._signature = func_sig

    def name(self) -> str:
        return "CallIndirect"


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

    def _to_serial(self, parent: Node) -> sops.LoadFunction:
        return sops.LoadFunction(
            parent=parent.idx,
            func_sig=self.signature._to_serial(),
            type_args=ser_it(self.type_args),
            instantiation=self.instantiation._to_serial(),
        )

    def _inputs(self) -> tys.TypeRow:
        return []

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType(input=[], output=[self.instantiation])

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        match port:
            case InPort(_, 0):
                return tys.FunctionKind(self.signature)
            case _:
                return DataflowOp.port_kind(self, port)

    def name(self) -> str:
        return "LoadFunc"


@dataclass
class Noop(AsExtOp, _PartialOp):
    """Identity operation that passes through its input."""

    _type: tys.Type | None = None
    num_out: int = field(default=1, repr=False)

    @property
    def type_(self) -> tys.Type:
        """The type of the input and output of the operation."""
        return _check_complete(self, self._type)

    def op_def(self) -> ext.OpDef:
        from hugr import std  # no circular import

        return std.PRELUDE.get_op("Noop")

    def type_args(self) -> list[tys.TypeArg]:
        return [tys.TypeTypeArg(self.type_)]

    def cached_signature(self) -> tys.FunctionType | None:
        return tys.FunctionType.endo(
            [self.type_],
        )

    def outer_signature(self) -> tys.FunctionType:
        return tys.FunctionType.endo(
            [self.type_],
        )

    def _set_in_types(self, types: tys.TypeRow) -> None:
        (t,) = types
        self._type = t

    def __repr__(self) -> str:
        return "Noop" + (f"({self._type})" if self._type is not None else "")

    def name(self) -> str:
        return "Noop"


@dataclass
class AliasDecl(Op):
    """Declare an external type alias."""

    #: Alias name.
    alias: str
    #: Type bound.
    bound: tys.TypeBound
    num_out: int = field(default=0, repr=False)

    def _to_serial(self, parent: Node) -> sops.AliasDecl:
        return sops.AliasDecl(
            parent=parent.idx,
            name=self.alias,
            bound=self.bound,
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)

    def name(self) -> str:
        return f"AliasDecl({self.alias})"


@dataclass
class AliasDefn(Op):
    """Declare a type alias."""

    #: Alias name.
    alias: str
    #: Type definition.
    definition: tys.Type
    num_out: int = field(default=0, repr=False)

    def _to_serial(self, parent: Node) -> sops.AliasDefn:
        return sops.AliasDefn(
            parent=parent.idx,
            name=self.alias,
            definition=self.definition._to_serial_root(),
        )

    def port_kind(self, port: InPort | OutPort) -> tys.Kind:
        raise self._invalid_port(port)

    def name(self) -> str:
        return f"AliasDefn({self.alias})"
