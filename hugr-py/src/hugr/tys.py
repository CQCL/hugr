"""HUGR edge kinds, types, type parameters and type arguments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import hugr.serialization.tys as stys
from hugr.utils import ser_it

ExtensionId = stys.ExtensionId
ExtensionSet = stys.ExtensionSet
TypeBound = stys.TypeBound


class TypeParam(Protocol):
    """A HUGR type parameter."""

    def to_serial(self) -> stys.BaseTypeParam:
        """Convert to serializable model."""
        ...  # pragma: no cover

    def to_serial_root(self) -> stys.TypeParam:
        return stys.TypeParam(root=self.to_serial())  # type: ignore[arg-type]


class TypeArg(Protocol):
    """A HUGR type argument, which can be bound to a :class:TypeParam."""

    def to_serial(self) -> stys.BaseTypeArg:
        """Convert to serializable model."""
        ...  # pragma: no cover

    def to_serial_root(self) -> stys.TypeArg:
        return stys.TypeArg(root=self.to_serial())  # type: ignore[arg-type]


@runtime_checkable
class Type(Protocol):
    """A HUGR type."""

    def to_serial(self) -> stys.BaseType:
        """Convert to serializable model."""
        ...  # pragma: no cover

    def to_serial_root(self) -> stys.Type:
        return stys.Type(root=self.to_serial())  # type: ignore[arg-type]

    def type_arg(self) -> TypeTypeArg:
        """The :class:`TypeTypeArg` for this type.

        Example:
            >>> Qubit.type_arg()
            TypeTypeArg(ty=Qubit)
        """
        return TypeTypeArg(self)


#: Row of types.
TypeRow = list[Type]

# --------------------------------------------
# --------------- TypeParam ------------------
# --------------------------------------------


@dataclass(frozen=True)
class TypeTypeParam(TypeParam):
    """A type parameter indicating a type with a given boumd."""

    bound: TypeBound

    def to_serial(self) -> stys.TypeTypeParam:
        return stys.TypeTypeParam(b=self.bound)


@dataclass(frozen=True)
class BoundedNatParam(TypeParam):
    """A type parameter indicating a natural number with an optional upper bound."""

    upper_bound: int | None

    def to_serial(self) -> stys.BoundedNatParam:
        return stys.BoundedNatParam(bound=self.upper_bound)


@dataclass(frozen=True)
class OpaqueParam(TypeParam):
    """Opaque type parameter."""

    ty: Opaque

    def to_serial(self) -> stys.OpaqueParam:
        return stys.OpaqueParam(ty=self.ty.to_serial())


@dataclass(frozen=True)
class ListParam(TypeParam):
    """Type parameter which requires a list of type arguments."""

    param: TypeParam

    def to_serial(self) -> stys.ListParam:
        return stys.ListParam(param=self.param.to_serial_root())


@dataclass(frozen=True)
class TupleParam(TypeParam):
    """Type parameter which requires a tuple of type arguments."""

    params: list[TypeParam]

    def to_serial(self) -> stys.TupleParam:
        return stys.TupleParam(params=ser_it(self.params))


@dataclass(frozen=True)
class ExtensionsParam(TypeParam):
    """An extension set parameter."""

    def to_serial(self) -> stys.ExtensionsParam:
        return stys.ExtensionsParam()


# ------------------------------------------
# --------------- TypeArg ------------------
# ------------------------------------------


@dataclass(frozen=True)
class TypeTypeArg(TypeArg):
    """A type argument for a :class:`TypeTypeParam`."""

    ty: Type

    def to_serial(self) -> stys.TypeTypeArg:
        return stys.TypeTypeArg(ty=self.ty.to_serial_root())


@dataclass(frozen=True)
class BoundedNatArg(TypeArg):
    """A type argument for a :class:`BoundedNatParam`."""

    n: int

    def to_serial(self) -> stys.BoundedNatArg:
        return stys.BoundedNatArg(n=self.n)


@dataclass(frozen=True)
class OpaqueArg(TypeArg):
    """An opaque type argument for a :class:`OpaqueParam`."""

    ty: Opaque
    value: Any

    def to_serial(self) -> stys.OpaqueArg:
        return stys.OpaqueArg(typ=self.ty.to_serial(), value=self.value)


@dataclass(frozen=True)
class SequenceArg(TypeArg):
    """Sequence of type arguments, for a :class:`ListParam` or :class:`TupleParam`."""

    elems: list[TypeArg]

    def to_serial(self) -> stys.SequenceArg:
        return stys.SequenceArg(elems=ser_it(self.elems))


@dataclass(frozen=True)
class ExtensionsArg(TypeArg):
    """Type argument for an :class:`ExtensionsParam`."""

    extensions: ExtensionSet

    def to_serial(self) -> stys.ExtensionsArg:
        return stys.ExtensionsArg(es=self.extensions)


@dataclass(frozen=True)
class VariableArg(TypeArg):
    """A type argument variable."""

    idx: int
    param: TypeParam

    def to_serial(self) -> stys.VariableArg:
        return stys.VariableArg(idx=self.idx, cached_decl=self.param.to_serial_root())


# ----------------------------------------------
# --------------- Type -------------------------
# ----------------------------------------------


@dataclass(frozen=True)
class Array(Type):
    """Prelude fixed `size` array of `ty` elements."""

    ty: Type
    size: int

    def to_serial(self) -> stys.Array:
        return stys.Array(ty=self.ty.to_serial_root(), len=self.size)


@dataclass()
class Sum(Type):
    """Algebraic sum-over-product type. Instances of this type correspond to
    tuples (products) over one of the `variant_rows` in the sum type, tagged by
    the index of the row.
    """

    variant_rows: list[TypeRow]

    def to_serial(self) -> stys.GeneralSum:
        return stys.GeneralSum(rows=[ser_it(row) for row in self.variant_rows])

    def as_tuple(self) -> Tuple:
        assert (
            len(self.variant_rows) == 1
        ), "Sum type must have exactly one row to be converted to a Tuple"
        return Tuple(*self.variant_rows[0])

    def __repr__(self) -> str:
        return f"Sum({self.variant_rows})"


@dataclass()
class UnitSum(Sum):
    """Simple :class:`Sum` type with `size` variants of empty rows."""

    size: int

    def __init__(self, size: int):
        self.size = size
        super().__init__(variant_rows=[[]] * size)

    def to_serial(self) -> stys.UnitSum:  # type: ignore[override]
        return stys.UnitSum(size=self.size)

    def __repr__(self) -> str:
        if self == Bool:
            return "Bool"
        elif self == Unit:
            return "Unit"
        return f"UnitSum({self.size})"


@dataclass()
class Tuple(Sum):
    """Product type with `tys` elements. Instances of this type correspond to
    :class:`Sum` with a single variant.
    """

    def __init__(self, *tys: Type):
        self.variant_rows = [list(tys)]

    def __repr__(self) -> str:
        return f"Tuple{tuple(self.variant_rows[0])}"


@dataclass(frozen=True)
class Variable(Type):
    """A type variable with a given bound, identified by index."""

    idx: int
    bound: TypeBound

    def to_serial(self) -> stys.Variable:
        return stys.Variable(i=self.idx, b=self.bound)


@dataclass(frozen=True)
class RowVariable(Type):
    """A type variable standing in for a row of types, identified by index."""

    idx: int
    bound: TypeBound

    def to_serial(self) -> stys.RowVar:
        return stys.RowVar(i=self.idx, b=self.bound)


@dataclass(frozen=True)
class USize(Type):
    """The Prelude unsigned size type."""

    def to_serial(self) -> stys.USize:
        return stys.USize()


@dataclass(frozen=True)
class Alias(Type):
    """Type alias."""

    name: str
    bound: TypeBound

    def to_serial(self) -> stys.Alias:
        return stys.Alias(name=self.name, bound=self.bound)


@dataclass(frozen=True)
class FunctionType(Type):
    """A function type, defined by input types,
    output types and extension requirements.
    """

    input: TypeRow
    output: TypeRow
    extension_reqs: ExtensionSet = field(default_factory=ExtensionSet)

    def to_serial(self) -> stys.FunctionType:
        return stys.FunctionType(
            input=ser_it(self.input),
            output=ser_it(self.output),
            extension_reqs=self.extension_reqs,
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


@dataclass(frozen=True)
class PolyFuncType(Type):
    """Polymorphic function type or type scheme. Defined by a list of type
    parameters that may appear in the :class:`FunctionType` body.
    """

    params: list[TypeParam]
    body: FunctionType

    def to_serial(self) -> stys.PolyFuncType:
        return stys.PolyFuncType(
            params=[p.to_serial_root() for p in self.params], body=self.body.to_serial()
        )


@dataclass
class Opaque(Type):
    """Opaque type, identified by `id` and with optional type arguments and bound."""

    id: str
    bound: TypeBound
    args: list[TypeArg] = field(default_factory=list)
    extension: ExtensionId = ""

    def to_serial(self) -> stys.Opaque:
        return stys.Opaque(
            extension=self.extension,
            id=self.id,
            args=[arg.to_serial_root() for arg in self.args],
            bound=self.bound,
        )


@dataclass
class _QubitDef(Type):
    def to_serial(self) -> stys.Qubit:
        return stys.Qubit()

    def __repr__(self) -> str:
        return "Qubit"


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
