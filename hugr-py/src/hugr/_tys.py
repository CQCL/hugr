from __future__ import annotations
from dataclasses import dataclass, field
import hugr.serialization.tys as stys
from hugr.utils import ser_it
from typing import Any, Protocol, runtime_checkable

ExtensionId = stys.ExtensionId
ExtensionSet = stys.ExtensionSet
TypeBound = stys.TypeBound


class TypeParam(Protocol):
    """A type parameter."""

    def to_serial(self) -> stys.BaseTypeParam: ...

    def to_serial_root(self) -> stys.TypeParam:
        return stys.TypeParam(root=self.to_serial())  # type: ignore[arg-type]


class TypeArg(Protocol):
    """A type argument."""

    def to_serial(self) -> stys.BaseTypeArg: ...

    def to_serial_root(self) -> stys.TypeArg:
        return stys.TypeArg(root=self.to_serial())  # type: ignore[arg-type]


@runtime_checkable
class Type(Protocol):
    """A type."""

    def to_serial(self) -> stys.BaseType: ...

    def to_serial_root(self) -> stys.Type:
        return stys.Type(root=self.to_serial())  # type: ignore[arg-type]


TypeRow = list[Type]

# --------------------------------------------
# --------------- TypeParam ------------------
# --------------------------------------------


@dataclass(frozen=True)
class TypeTypeParam(TypeParam):
    bound: TypeBound

    def to_serial(self) -> stys.TypeTypeParam:
        return stys.TypeTypeParam(b=self.bound)


@dataclass(frozen=True)
class BoundedNatParam(TypeParam):
    upper_bound: int | None

    def to_serial(self) -> stys.BoundedNatParam:
        return stys.BoundedNatParam(bound=self.upper_bound)


@dataclass(frozen=True)
class OpaqueParam(TypeParam):
    ty: Opaque

    def to_serial(self) -> stys.OpaqueParam:
        return stys.OpaqueParam(ty=self.ty.to_serial())


@dataclass(frozen=True)
class ListParam(TypeParam):
    param: TypeParam

    def to_serial(self) -> stys.ListParam:
        return stys.ListParam(param=self.param.to_serial_root())


@dataclass(frozen=True)
class TupleParam(TypeParam):
    params: list[TypeParam]

    def to_serial(self) -> stys.TupleParam:
        return stys.TupleParam(params=ser_it(self.params))


@dataclass(frozen=True)
class ExtensionsParam(TypeParam):
    def to_serial(self) -> stys.ExtensionsParam:
        return stys.ExtensionsParam()


# ------------------------------------------
# --------------- TypeArg ------------------
# ------------------------------------------


@dataclass(frozen=True)
class TypeTypeArg(TypeArg):
    ty: Type

    def to_serial(self) -> stys.TypeTypeArg:
        return stys.TypeTypeArg(ty=self.ty.to_serial_root())


@dataclass(frozen=True)
class BoundedNatArg(TypeArg):
    n: int

    def to_serial(self) -> stys.BoundedNatArg:
        return stys.BoundedNatArg(n=self.n)


@dataclass(frozen=True)
class OpaqueArg(TypeArg):
    ty: Opaque
    value: Any

    def to_serial(self) -> stys.OpaqueArg:
        return stys.OpaqueArg(typ=self.ty.to_serial(), value=self.value)


@dataclass(frozen=True)
class SequenceArg(TypeArg):
    elems: list[TypeArg]

    def to_serial(self) -> stys.SequenceArg:
        return stys.SequenceArg(elems=ser_it(self.elems))


@dataclass(frozen=True)
class ExtensionsArg(TypeArg):
    extensions: ExtensionSet

    def to_serial(self) -> stys.ExtensionsArg:
        return stys.ExtensionsArg(es=self.extensions)


@dataclass(frozen=True)
class VariableArg(TypeArg):
    idx: int
    param: TypeParam

    def to_serial(self) -> stys.VariableArg:
        return stys.VariableArg(idx=self.idx, cached_decl=self.param.to_serial_root())


# ----------------------------------------------
# --------------- Type -------------------------
# ----------------------------------------------


@dataclass(frozen=True)
class Array(Type):
    ty: Type
    size: int

    def to_serial(self) -> stys.Array:
        return stys.Array(ty=self.ty.to_serial_root(), len=self.size)


@dataclass()
class Sum(Type):
    variant_rows: list[TypeRow]

    def to_serial(self) -> stys.GeneralSum:
        return stys.GeneralSum(rows=[ser_it(row) for row in self.variant_rows])

    def as_tuple(self) -> Tuple:
        assert (
            len(self.variant_rows) == 1
        ), "Sum type must have exactly one row to be converted to a Tuple"
        return Tuple(*self.variant_rows[0])


@dataclass()
class UnitSum(Sum):
    size: int

    def __init__(self, size: int):
        self.size = size
        super().__init__(variant_rows=[[]] * size)

    def to_serial(self) -> stys.UnitSum:  # type: ignore[override]
        return stys.UnitSum(size=self.size)


@dataclass()
class Tuple(Sum):
    def __init__(self, *tys: Type):
        self.variant_rows = [list(tys)]


@dataclass(frozen=True)
class Variable(Type):
    idx: int
    bound: TypeBound

    def to_serial(self) -> stys.Variable:
        return stys.Variable(i=self.idx, b=self.bound)


@dataclass(frozen=True)
class RowVariable(Type):
    idx: int
    bound: TypeBound

    def to_serial(self) -> stys.RowVar:
        return stys.RowVar(i=self.idx, b=self.bound)


@dataclass(frozen=True)
class USize(Type):
    def to_serial(self) -> stys.USize:
        return stys.USize()


@dataclass(frozen=True)
class Alias(Type):
    name: str
    bound: TypeBound

    def to_serial(self) -> stys.Alias:
        return stys.Alias(name=self.name, bound=self.bound)


@dataclass(frozen=True)
class FunctionType(Type):
    input: list[Type]
    output: list[Type]
    extension_reqs: ExtensionSet = field(default_factory=ExtensionSet)

    def to_serial(self) -> stys.FunctionType:
        return stys.FunctionType(input=ser_it(self.input), output=ser_it(self.output))

    @classmethod
    def empty(cls) -> FunctionType:
        return cls(input=[], output=[])

    def flip(self) -> FunctionType:
        return FunctionType(input=self.output, output=self.input)


@dataclass(frozen=True)
class PolyFuncType(Type):
    params: list[TypeParam]
    body: FunctionType

    def to_serial(self) -> stys.PolyFuncType:
        return stys.PolyFuncType(
            params=[p.to_serial_root() for p in self.params], body=self.body.to_serial()
        )


@dataclass
class Opaque(Type):
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
class QubitDef(Type):
    def to_serial(self) -> stys.Qubit:
        return stys.Qubit()


Qubit = QubitDef()
Bool = UnitSum(size=2)
Unit = UnitSum(size=1)


@dataclass(frozen=True)
class ValueKind:
    ty: Type


@dataclass(frozen=True)
class ConstKind:
    ty: Type


@dataclass(frozen=True)
class FunctionKind:
    ty: PolyFuncType


@dataclass(frozen=True)
class CFKind: ...


@dataclass(frozen=True)
class OrderKind: ...


Kind = ValueKind | ConstKind | FunctionKind | CFKind | OrderKind
