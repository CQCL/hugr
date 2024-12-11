"""Collection types and operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import hugr.tys as tys
from hugr import val
from hugr.std import _load_extension
from hugr.utils import comma_sep_str

EXTENSION = _load_extension("collections.list")


@dataclass(eq=False)
class List(tys.ExtType):
    """List type with a fixed element type."""

    ty: tys.Type = field(default_factory=lambda: tys.Unit)

    def __init__(self, ty: tys.Type | tys.TypeTypeArg) -> None:
        if isinstance(ty, tys.Type):
            ty = tys.TypeTypeArg(ty)

        self.type_def = EXTENSION.types["List"]
        self.args = [ty]
        self.ty = ty.ty

    def type_bound(self) -> tys.TypeBound:
        return self.ty.type_bound()


@dataclass
class ListVal(val.ExtensionValue):
    """Constant value for a list of elements."""

    v: list[val.Value]
    ty: tys.Type

    def __init__(self, v: list[val.Value], elem_ty: tys.Type) -> None:
        self.v = v
        self.ty = List(elem_ty)

    def to_value(self) -> val.Extension:
        name = "ListValue"
        # The value list must be serialized at this point, otherwise the
        # `Extension` value would not be serializable.
        vs = [v._to_serial_root() for v in self.v]
        return val.Extension(name, typ=self.ty, val=vs, extensions=[EXTENSION.name])

    def __str__(self) -> str:
        return f"[{comma_sep_str(self.v)}]"
