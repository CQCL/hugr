"""Statically sized immutable array type and its operations."""

from __future__ import annotations

from dataclasses import dataclass

from hugr import tys, val
from hugr.std import _load_extension
from hugr.utils import comma_sep_str

EXTENSION = _load_extension("collections.static_array")


@dataclass(eq=False)
class StaticArray(tys.ExtType):
    """Fixed size immutable array of `ty` elements."""

    def __init__(self, ty: tys.Type) -> None:
        self.type_def = EXTENSION.types["static_array"]
        if (
            tys.TypeBound.join(ty.type_bound(), tys.TypeBound.Copyable)
            != tys.TypeBound.Copyable
        ):
            msg = "Static array elements must be copyable"
            raise ValueError(msg)
        self.args = [tys.TypeTypeArg(ty)]

    @property
    def ty(self) -> tys.Type:
        assert isinstance(
            self.args[0], tys.TypeTypeArg
        ), "Array elements must have a valid type"
        return self.args[0].ty

    def type_bound(self) -> tys.TypeBound:
        return self.ty.type_bound()


@dataclass
class StaticArrayVal(val.ExtensionValue):
    """Constant value for a statically sized immutable array of elements."""

    v: list[val.Value]
    ty: StaticArray
    name: str

    def __init__(self, v: list[val.Value], elem_ty: tys.Type, name: str) -> None:
        self.v = v
        self.ty = StaticArray(elem_ty)
        self.name = name

    def to_value(self) -> val.Extension:
        serial_val = {
            "value": {
                "values": [v._to_serial_root() for v in self.v],
                "typ": self.ty.ty._to_serial_root(),
            },
            "name": self.name,
        }
        return val.Extension("StaticArrayValue", typ=self.ty, val=serial_val)

    def __str__(self) -> str:
        return f"static_array({comma_sep_str(self.v)})"
