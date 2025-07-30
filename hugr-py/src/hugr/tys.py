"""HUGR edge kinds, types, type parameters and type arguments."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

import hugr._serialization.tys as stys
import hugr.model as model
from hugr.utils import comma_sep_repr, comma_sep_str, comma_sep_str_paren, ser_it

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from hugr import ext


ExtensionId = stys.ExtensionId
ExtensionSet = stys.ExtensionSet
TypeBound = stys.TypeBound
Visibility = Literal["Public", "Private"]


@runtime_checkable
class TypeParam(Protocol):
    """A HUGR type parameter."""

    def _to_serial(self) -> stys.BaseTypeParam:
        """Convert to serializable model."""
        ...  # pragma: no cover

    def _to_serial_root(self) -> stys.TypeParam:
        return stys.TypeParam(root=self._to_serial())  # type: ignore[arg-type]

    def to_model(self) -> model.Term:
        """Convert the type parameter to a model Term."""
        raise NotImplementedError(self)


@runtime_checkable
class TypeArg(Protocol):
    """A HUGR type argument, which can be bound to a :class:TypeParam."""

    def _to_serial(self) -> stys.BaseTypeArg:
        """Convert to serializable model."""
        ...  # pragma: no cover

    def _to_serial_root(self) -> stys.TypeArg:
        return stys.TypeArg(root=self._to_serial())  # type: ignore[arg-type]

    def resolve(self, registry: ext.ExtensionRegistry) -> TypeArg:
        """Resolve types in the argument using the given registry."""
        return self

    def to_model(self) -> model.Term | model.Splice:
        """Convert the type argument to a model Term."""
        raise NotImplementedError(self)


@runtime_checkable
class Type(Protocol):
    """A HUGR type."""

    def type_bound(self) -> stys.TypeBound:
        """The bound of this type.

        Example:
            >>> Tuple(Bool, Bool).type_bound()
            <TypeBound.Copyable: 'C'>
            >>> Tuple(Qubit, Bool).type_bound()
            <TypeBound.Linear: 'A'>
        """
        ...  # pragma: no cover

    def _to_serial(self) -> stys.BaseType:
        """Convert to serializable model."""
        ...  # pragma: no cover

    def _to_serial_root(self) -> stys.Type:
        return stys.Type(root=self._to_serial())  # type: ignore[arg-type]

    def type_arg(self) -> TypeTypeArg:
        """The :class:`TypeTypeArg` for this type.

        Example:
            >>> Qubit.type_arg()
            TypeTypeArg(ty=Qubit)
        """
        return TypeTypeArg(self)

    def resolve(self, registry: ext.ExtensionRegistry) -> Type:
        """Resolve types in the type using the given registry."""
        return self

    def to_model(self) -> model.Term | model.Splice:
        """Convert the type to a model Term."""
        raise NotImplementedError(self)


#: Row of types.
TypeRow = list[Type]

# --------------------------------------------
# --------------- TypeParam ------------------
# --------------------------------------------


@dataclass(frozen=True)
class TypeTypeParam(TypeParam):
    """A type parameter indicating a type with a given boumd."""

    bound: TypeBound

    def _to_serial(self) -> stys.TypeTypeParam:
        return stys.TypeTypeParam(b=self.bound)

    def __str__(self) -> str:
        return str(self.bound)

    def to_model(self) -> model.Term:
        # Note that we drop the bound.
        return model.Apply("core.type")


@dataclass(frozen=True)
class BoundedNatParam(TypeParam):
    """A type parameter indicating a natural number with an optional upper bound."""

    upper_bound: int | None = None

    def _to_serial(self) -> stys.BoundedNatParam:
        return stys.BoundedNatParam(bound=self.upper_bound)

    def __str__(self) -> str:
        if self.upper_bound is None:
            return "Nat"
        return f"Nat({self.upper_bound})"

    def to_model(self) -> model.Term:
        # Note that we drop the bound.
        return model.Apply("core.nat")


@dataclass(frozen=True)
class StringParam(TypeParam):
    """String type parameter."""

    def _to_serial(self) -> stys.StringParam:
        return stys.StringParam()

    def __str__(self) -> str:
        return "String"

    def to_model(self) -> model.Term:
        return model.Apply("core.str")


@dataclass(frozen=True)
class FloatParam(TypeParam):
    """Float type parameter."""

    def _to_serial(self) -> stys.FloatParam:
        return stys.FloatParam()

    def __str__(self) -> str:
        return "Float"

    def to_model(self) -> model.Term:
        return model.Apply("core.float")


@dataclass(frozen=True)
class BytesParam(TypeParam):
    """Bytes type parameter."""

    def _to_serial(self) -> stys.BytesParam:
        return stys.BytesParam()

    def __str__(self) -> str:
        return "Bytes"

    def to_model(self) -> model.Term:
        return model.Apply("core.bytes")


@dataclass(frozen=True)
class ListParam(TypeParam):
    """Type parameter which requires a list of type arguments."""

    param: TypeParam

    def _to_serial(self) -> stys.ListParam:
        return stys.ListParam(param=self.param._to_serial_root())

    def __str__(self) -> str:
        return f"[{self.param}]"

    def to_model(self) -> model.Term:
        item_type = self.param.to_model()
        return model.Apply("core.list", [item_type])


@dataclass(frozen=True)
class TupleParam(TypeParam):
    """Type parameter which requires a tuple of type arguments."""

    params: list[TypeParam]

    def _to_serial(self) -> stys.TupleParam:
        return stys.TupleParam(params=ser_it(self.params))

    def __str__(self) -> str:
        return f"({comma_sep_str(self.params)})"

    def to_model(self) -> model.Term:
        item_types = model.List([param.to_model() for param in self.params])
        return model.Apply("core.tuple", [item_types])


@dataclass(frozen=True)
class ConstParam(TypeParam):
    """Type parameter which requires a constant value."""

    ty: Type

    def _to_serial(self) -> stys.ConstParam:
        return stys.ConstParam(ty=self.ty._to_serial_root())

    def __str__(self) -> str:
        return f"Const({self.ty!s})"

    def to_model(self) -> model.Term:
        ty = cast(model.Term, self.ty.to_model())
        return model.Apply("core.const", [ty])


# ------------------------------------------
# --------------- TypeArg ------------------
# ------------------------------------------


@dataclass(frozen=True)
class TypeTypeArg(TypeArg):
    """A type argument for a :class:`TypeTypeParam`."""

    ty: Type

    def _to_serial(self) -> stys.TypeTypeArg:
        return stys.TypeTypeArg(ty=self.ty._to_serial_root())

    def resolve(self, registry: ext.ExtensionRegistry) -> TypeArg:
        return TypeTypeArg(self.ty.resolve(registry))

    def __str__(self) -> str:
        return f"Type({self.ty!s})"

    def to_model(self) -> model.Term | model.Splice:
        return self.ty.to_model()


@dataclass(frozen=True)
class BoundedNatArg(TypeArg):
    """A type argument for a :class:`BoundedNatParam`."""

    n: int

    def _to_serial(self) -> stys.BoundedNatArg:
        return stys.BoundedNatArg(n=self.n)

    def __str__(self) -> str:
        return str(self.n)

    def to_model(self) -> model.Term:
        return model.Literal(self.n)


@dataclass(frozen=True)
class StringArg(TypeArg):
    """A utf-8 encoded string type argument."""

    value: str

    def _to_serial(self) -> stys.StringArg:
        return stys.StringArg(arg=self.value)

    def __str__(self) -> str:
        return f'"{self.value}"'

    def to_model(self) -> model.Term:
        return model.Literal(self.value)


@dataclass(frozen=True)
class FloatArg(TypeArg):
    """A floating point type argument."""

    value: float

    def _to_serial(self) -> stys.FloatArg:
        return stys.FloatArg(value=self.value)

    def __str__(self) -> str:
        return f"{self.value}"

    def to_model(self) -> model.Term:
        return model.Literal(self.value)


@dataclass(frozen=True)
class BytesArg(TypeArg):
    """A bytes type argument."""

    value: bytes

    def _to_serial(self) -> stys.BytesArg:
        value = base64.b64encode(self.value).decode()
        return stys.BytesArg(value=value)

    def __str__(self) -> str:
        return "bytes"

    def to_model(self) -> model.Term:
        return model.Literal(self.value)


@dataclass(frozen=True)
class ListArg(TypeArg):
    """Sequence of type arguments for a :class:`ListParam`."""

    elems: list[TypeArg]

    def _to_serial(self) -> stys.ListArg:
        return stys.ListArg(elems=ser_it(self.elems))

    def resolve(self, registry: ext.ExtensionRegistry) -> TypeArg:
        return ListArg([arg.resolve(registry) for arg in self.elems])

    def __str__(self) -> str:
        return f"[{comma_sep_str(self.elems)}]"

    def to_model(self) -> model.Term:
        return model.List([elem.to_model() for elem in self.elems])


@dataclass(frozen=True)
class ListConcatArg(TypeArg):
    """Sequence of lists to concatenate for a :class:`ListParam`."""

    lists: list[TypeArg]

    def _to_serial(self) -> stys.ListConcatArg:
        return stys.ListConcatArg(lists=ser_it(self.lists))

    def resolve(self, registry: ext.ExtensionRegistry) -> TypeArg:
        return ListConcatArg([arg.resolve(registry) for arg in self.lists])

    def __str__(self) -> str:
        lists = comma_sep_str(f"... {list}" for list in self.lists)
        return f"[{lists}]"

    def to_model(self) -> model.Term:
        return model.List(
            [model.Splice(cast(model.Term, elem.to_model())) for elem in self.lists]
        )


@dataclass(frozen=True)
class TupleArg(TypeArg):
    """Sequence of type arguments for a :class:`TupleParam`."""

    elems: list[TypeArg]

    def _to_serial(self) -> stys.TupleArg:
        return stys.TupleArg(elems=ser_it(self.elems))

    def resolve(self, registry: ext.ExtensionRegistry) -> TypeArg:
        return TupleArg([arg.resolve(registry) for arg in self.elems])

    def __str__(self) -> str:
        return f"({comma_sep_str(self.elems)})"

    def to_model(self) -> model.Term:
        return model.Tuple([elem.to_model() for elem in self.elems])


@dataclass(frozen=True)
class TupleConcatArg(TypeArg):
    """Sequence of tuples to concatenate for a :class:`TupleParam`."""

    tuples: list[TypeArg]

    def _to_serial(self) -> stys.TupleConcatArg:
        return stys.TupleConcatArg(tuples=ser_it(self.tuples))

    def resolve(self, registry: ext.ExtensionRegistry) -> TypeArg:
        return TupleConcatArg([arg.resolve(registry) for arg in self.tuples])

    def __str__(self) -> str:
        tuples = comma_sep_str(f"... {tuple}" for tuple in self.tuples)
        return f"({tuples})"

    def to_model(self) -> model.Term:
        return model.Tuple(
            [model.Splice(cast(model.Term, elem.to_model())) for elem in self.tuples]
        )


@dataclass(frozen=True)
class VariableArg(TypeArg):
    """A type argument variable."""

    idx: int
    param: TypeParam

    def _to_serial(self) -> stys.VariableArg:
        return stys.VariableArg(idx=self.idx, cached_decl=self.param._to_serial_root())

    def __str__(self) -> str:
        return f"${self.idx}"

    def to_model(self) -> model.Term:
        return model.Var(str(self.idx))


# ----------------------------------------------
# --------------- Type -------------------------
# ----------------------------------------------


@dataclass()
class Sum(Type):
    """Algebraic sum-over-product type. Instances of this type correspond to
    tuples (products) over one of the `variant_rows` in the sum type, tagged by
    the index of the row.
    """

    variant_rows: list[TypeRow]

    def _to_serial(self) -> stys.GeneralSum:
        return stys.GeneralSum(rows=[ser_it(row) for row in self.variant_rows])

    def as_tuple(self) -> Tuple:
        assert (
            len(self.variant_rows) == 1
        ), "Sum type must have exactly one row to be converted to a Tuple"
        return Tuple(*self.variant_rows[0])

    def __repr__(self) -> str:
        if self == Bool:
            return "Bool"
        elif self == Unit:
            return "Unit"
        elif all(len(row) == 0 for row in self.variant_rows):
            return f"UnitSum({len(self.variant_rows)})"
        elif len(self.variant_rows) == 1:
            return f"Tuple{tuple(self.variant_rows[0])}"
        elif len(self.variant_rows) == 2 and len(self.variant_rows[0]) == 0:
            return f"Option({comma_sep_repr(self.variant_rows[1])})"
        elif len(self.variant_rows) == 2:
            left, right = self.variant_rows
            return f"Either(left={left}, right={right})"
        else:
            return f"Sum({self.variant_rows})"

    def __str__(self) -> str:
        if self == Bool:
            return "Bool"
        elif self == Unit:
            return "Unit"
        elif all(len(row) == 0 for row in self.variant_rows):
            return f"UnitSum({len(self.variant_rows)})"
        elif len(self.variant_rows) == 1:
            return f"Tuple{tuple(self.variant_rows[0])}"
        elif len(self.variant_rows) == 2 and len(self.variant_rows[0]) == 0:
            return f"Option({comma_sep_str(self.variant_rows[1])})"
        elif len(self.variant_rows) == 2:
            left, right = self.variant_rows
            return f"Either({comma_sep_str_paren(left)}, {comma_sep_str_paren(right)})"
        else:
            return f"Sum({self.variant_rows})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sum) and self.variant_rows == other.variant_rows

    def type_bound(self) -> TypeBound:
        return TypeBound.join(*(t.type_bound() for r in self.variant_rows for t in r))

    def resolve(self, registry: ext.ExtensionRegistry) -> Sum:
        """Resolve types in the sum type using the given registry."""
        return Sum([[ty.resolve(registry) for ty in row] for row in self.variant_rows])

    def to_model(self) -> model.Term:
        variants = model.List(
            [model.List([typ.to_model() for typ in row]) for row in self.variant_rows]
        )
        return model.Apply("core.adt", [variants])


@dataclass(eq=False, repr=False)
class UnitSum(Sum):
    """Simple :class:`Sum` type with `size` variants of empty rows."""

    size: int = field(compare=False)

    def __init__(self, size: int):
        self.size = size
        super().__init__(variant_rows=[[]] * size)

    def _to_serial(self) -> stys.UnitSum:  # type: ignore[override]
        return stys.UnitSum(size=self.size)

    def resolve(self, registry: ext.ExtensionRegistry) -> UnitSum:
        return self

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(eq=False, repr=False)
class Tuple(Sum):
    """Product type with `tys` elements. Instances of this type correspond to
    :class:`Sum` with a single variant.
    """

    def __init__(self, *tys: Type):
        self.variant_rows = [list(tys)]


@dataclass(eq=False, repr=False)
class Option(Sum):
    """Optional tuple of elements.

    Instances of this type correspond to :class:`Sum` with two variants.
    The second variant is the tuple of elements, the first is empty.
    """

    def __init__(self, *tys: Type):
        self.variant_rows = [[], list(tys)]


@dataclass(eq=False, repr=False)
class Either(Sum):
    """Two-variant tuple of elements.

    Instances of this type correspond to :class:`Sum` with a Left and a Right variant.

    In fallible contexts, the Right variant is used to represent success, and the
    Left variant is used to represent failure.
    """

    def __init__(self, left: Iterable[Type], right: Iterable[Type]):
        self.variant_rows = [list(left), list(right)]


@dataclass(frozen=True)
class Variable(Type):
    """A type variable with a given bound, identified by index."""

    idx: int
    bound: TypeBound

    def _to_serial(self) -> stys.Variable:
        return stys.Variable(i=self.idx, b=self.bound)

    def type_bound(self) -> TypeBound:
        return self.bound

    def __repr__(self) -> str:
        return f"${self.idx}"

    def to_model(self) -> model.Term:
        return model.Var(str(self.idx))


@dataclass(frozen=True)
class RowVariable(Type):
    """A type variable standing in for a row of types, identified by index."""

    idx: int
    bound: TypeBound

    def _to_serial(self) -> stys.RowVar:
        return stys.RowVar(i=self.idx, b=self.bound)

    def type_bound(self) -> TypeBound:
        return self.bound

    def __repr__(self) -> str:
        return f"${self.idx}"

    def to_model(self):
        return model.Splice(model.Var(str(self.idx)))


@dataclass(frozen=True)
class USize(Type):
    """The Prelude unsigned size type."""

    def _to_serial(self) -> stys.USize:
        return stys.USize()

    def type_bound(self) -> TypeBound:
        return TypeBound.Copyable

    def __repr__(self) -> str:
        return "USize"

    def to_model(self) -> model.Term:
        return model.Apply("prelude.usize")


@dataclass(frozen=True)
class Alias(Type):
    """Type alias."""

    name: str
    bound: TypeBound

    def _to_serial(self) -> stys.Alias:
        return stys.Alias(name=self.name, bound=self.bound)

    def type_bound(self) -> TypeBound:
        return self.bound

    def __repr__(self) -> str:
        return self.name

    def to_model(self) -> model.Term:
        return model.Apply(self.name)


@dataclass(frozen=True)
class FunctionType(Type):
    """A function type, defined by input types,
    output types and extension requirements.
    """

    input: TypeRow
    output: TypeRow

    def type_bound(self) -> TypeBound:
        return TypeBound.Copyable

    def _to_serial(self) -> stys.FunctionType:
        return stys.FunctionType(
            input=ser_it(self.input),
            output=ser_it(self.output),
        )

    @classmethod
    def empty(cls) -> FunctionType:
        """Generate an empty function type.

        Example:
            >>> FunctionType.empty()
            FunctionType([], [])
        """
        return cls(input=[], output=[])

    @classmethod
    def endo(cls, tys: TypeRow) -> FunctionType:
        """Function type with the same input and output types.

        Example:
            >>> FunctionType.endo([Qubit])
            FunctionType([Qubit], [Qubit])
        """
        return cls(input=tys, output=tys)

    def flip(self) -> FunctionType:
        """Return a new function type with input and output types swapped.

        Example:
            >>> FunctionType([Qubit], [Bool]).flip()
            FunctionType([Bool], [Qubit])
        """
        return FunctionType(input=list(self.output), output=list(self.input))

    def __repr__(self) -> str:
        return f"FunctionType({self.input}, {self.output})"

    def resolve(self, registry: ext.ExtensionRegistry) -> FunctionType:
        """Resolve types in the function type using the given registry."""
        return FunctionType(
            input=[ty.resolve(registry) for ty in self.input],
            output=[ty.resolve(registry) for ty in self.output],
        )

    def __str__(self) -> str:
        return f"{comma_sep_str(self.input)} -> {comma_sep_str(self.output)}"

    def to_model(self) -> model.Term:
        inputs = model.List([input.to_model() for input in self.input])
        outputs = model.List([output.to_model() for output in self.output])
        return model.Apply("core.fn", [inputs, outputs])


@dataclass(frozen=True)
class PolyFuncType(Type):
    """Polymorphic function type or type scheme. Defined by a list of type
    parameters that may appear in the :class:`FunctionType` body.
    """

    params: list[TypeParam]
    body: FunctionType

    def type_bound(self) -> TypeBound:
        return TypeBound.Copyable

    def _to_serial(self) -> stys.PolyFuncType:
        return stys.PolyFuncType(
            params=[p._to_serial_root() for p in self.params],
            body=self.body._to_serial(),
        )

    def resolve(self, registry: ext.ExtensionRegistry) -> PolyFuncType:
        """Resolve types in the polymorphic function type using the given registry."""
        return PolyFuncType(
            params=self.params,
            body=self.body.resolve(registry),
        )

    def __str__(self) -> str:
        return f"∀ {comma_sep_str(self.params)}. {self.body!s}"

    @classmethod
    def empty(cls) -> PolyFuncType:
        """Generate an empty polymorphic function type.

        Example:
            >>> PolyFuncType.empty()
            PolyFuncType(params=[], body=FunctionType([], []))
        """
        return PolyFuncType(params=[], body=FunctionType.empty())

    def to_model(self) -> model.Term:
        # A `PolyFuncType` should not be a `Type`.
        error = "PolyFuncType used as a Type"
        raise TypeError(error)


@dataclass
class ExtType(Type):
    """Extension type, defined by a type definition and type arguments."""

    type_def: ext.TypeDef
    args: list[TypeArg] = field(default_factory=list)

    def type_bound(self) -> TypeBound:
        from hugr.ext import ExplicitBound, FromParamsBound

        match self.type_def.bound:
            case ExplicitBound(exp_bound):
                return exp_bound
            case FromParamsBound(indices):
                bounds: list[TypeBound] = []
                for idx in indices:
                    arg = self.args[idx]
                    if isinstance(arg, TypeTypeArg):
                        bounds.append(arg.ty.type_bound())
                return TypeBound.join(*bounds)

    def _to_serial(self) -> stys.Opaque:
        return self._to_opaque()._to_serial()

    def _to_opaque(self) -> Opaque:
        assert self.type_def._extension is not None, "Extension must be initialised."

        return Opaque(
            extension=self.type_def._extension.name,
            id=self.type_def.name,
            args=self.args,
            bound=self.type_bound(),
        )

    def __str__(self) -> str:
        return _type_str(self.type_def.name, self.args)

    def __eq__(self, value):
        # Ignore extra attributes on subclasses
        if isinstance(value, ExtType):
            return self.type_def == value.type_def and self.args == value.args
        return super().__eq__(value)

    def to_model(self) -> model.Term:
        return self._to_opaque().to_model()


def _type_str(name: str, args: Sequence[TypeArg]) -> str:
    if len(args) == 0:
        return name
    return f"{name}<{comma_sep_str(args)}>"


@dataclass
class Opaque(Type):
    """Opaque type, identified by `id` and with optional type arguments and bound."""

    id: str
    bound: TypeBound
    args: list[TypeArg] = field(default_factory=list)
    extension: ExtensionId = ""

    def _to_serial(self) -> stys.Opaque:
        return stys.Opaque(
            extension=self.extension,
            id=self.id,
            args=[arg._to_serial_root() for arg in self.args],
            bound=self.bound,
        )

    def type_bound(self) -> TypeBound:
        return self.bound

    def resolve(self, registry: ext.ExtensionRegistry) -> Type:
        """Resolve the opaque type to an :class:`ExtType` using the given registry.

        If the extension or type is not found, return the original type.
        """
        from hugr.ext import ExtensionRegistry, Extension  # noqa: I001 # no circular import

        try:
            type_def = registry.get_extension(self.extension).get_type(self.id)
        except (ExtensionRegistry.ExtensionNotFound, Extension.TypeNotFound):
            return self

        return ExtType(type_def, self.args)

    def __str__(self) -> str:
        return _type_str(self.id, self.args)

    def to_model(self) -> model.Term:
        # This cast is only necessary because `Type` can both be an
        # actual type or a row variable.
        args = [cast(model.Term, arg.to_model()) for arg in self.args]

        return model.Apply(f"{self.extension}.{self.id}", args)


@dataclass
class _QubitDef(Type):
    def type_bound(self) -> TypeBound:
        return TypeBound.Linear

    def _to_serial(self) -> stys.Qubit:
        return stys.Qubit()

    def __repr__(self) -> str:
        return "Qubit"

    def to_model(self) -> model.Term:
        return model.Apply("prelude.qubit", [])


#: Qubit type.
Qubit = _QubitDef()
#: Boolean type (:class:`UnitSum` of size 2).
Bool = UnitSum(size=2)
#: Unit type (:class:`UnitSum` of size 1).
Unit = UnitSum(size=1)


@dataclass(frozen=True)
class ValueKind:
    """Dataflow value edges."""

    #: Type of the value.
    ty: Type

    def __repr__(self) -> str:
        return f"ValueKind({self.ty})"


@dataclass(frozen=True)
class ConstKind:
    """Static constant value edges."""

    #: Type of the constant.
    ty: Type

    def __repr__(self) -> str:
        return f"ConstKind({self.ty})"


@dataclass(frozen=True)
class FunctionKind:
    """Statically defined function edges."""

    #: Type of the function.
    ty: PolyFuncType

    def __repr__(self) -> str:
        return f"FunctionKind({self.ty})"


@dataclass(frozen=True)
class CFKind:
    """Control flow edges."""


@dataclass(frozen=True)
class OrderKind:
    """State order edges."""


#: The kind of a HUGR graph edge.
Kind = ValueKind | ConstKind | FunctionKind | CFKind | OrderKind


def get_first_sum(types: TypeRow) -> tuple[Sum, TypeRow]:
    """Check the first type in a row of types is a :class:`Sum`, returning it
    and the rest.

    Args:
        types: row of types.

    Raises:
        AssertionError: if the first type is not a :class:`Sum`.

    Example:
        >>> get_first_sum([UnitSum(3), Qubit])
        (UnitSum(3), [Qubit])
        >>> get_first_sum([Qubit, UnitSum(3)]) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        AssertionError: Expected Sum, got Qubit
    """
    (sum_, *other) = types
    assert isinstance(sum_, Sum), f"Expected Sum, got {sum_}"
    return sum_, other
