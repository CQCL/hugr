from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
import hugr.serialization.ops as sops
import hugr.serialization.tys as stys
import hugr._tys as tys
from hugr.utils import ser_it

if TYPE_CHECKING:
    from hugr._hugr import Hugr


@runtime_checkable
class Value(Protocol):
    def to_serial(self) -> sops.BaseValue: ...
    def to_serial_root(self) -> sops.Value:
        return sops.Value(root=self.to_serial())  # type: ignore[arg-type]

    def type_(self) -> tys.Type: ...


@dataclass
class Sum(Value):
    tag: int
    typ: tys.Sum
    vals: list[Value]

    def type_(self) -> tys.Sum:
        return self.typ

    def to_serial(self) -> sops.SumValue:
        return sops.SumValue(
            tag=self.tag,
            typ=stys.SumType(root=self.type_().to_serial()),
            vs=ser_it(self.vals),
        )


def bool_value(b: bool) -> Sum:
    return Sum(
        tag=int(b),
        typ=tys.Bool,
        vals=[],
    )


Unit = Sum(0, tys.Unit, [])
TRUE = bool_value(True)
FALSE = bool_value(False)


@dataclass
class Tuple(Value):
    vals: list[Value]

    def type_(self) -> tys.Tuple:
        return tys.Tuple(*(v.type_() for v in self.vals))

    def to_serial(self) -> sops.TupleValue:
        return sops.TupleValue(
            vs=ser_it(self.vals),
        )


@dataclass
class Function(Value):
    body: Hugr

    def type_(self) -> tys.FunctionType:
        return self.body.root_op().inner_signature()

    def to_serial(self) -> sops.FunctionValue:
        return sops.FunctionValue(
            hugr=self.body.to_serial(),
        )


@dataclass
class Extension(Value):
    name: str
    typ: tys.Type
    val: Any
    extensions: tys.ExtensionSet = field(default_factory=tys.ExtensionSet)

    def type_(self) -> tys.Type:
        return self.typ

    def to_serial(self) -> sops.ExtensionValue:
        return sops.ExtensionValue(
            typ=self.typ.to_serial_root(),
            value=sops.CustomConst(c=self.name, v=self.val),
            extensions=self.extensions,
        )


class ExtensionValue(Value, Protocol):
    def to_value(self) -> Extension: ...

    def type_(self) -> tys.Type:
        return self.to_value().type_()

    def to_serial(self) -> sops.ExtensionValue:
        return self.to_value().to_serial()
