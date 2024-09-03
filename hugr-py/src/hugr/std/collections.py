"""Collection types and operations."""

from __future__ import annotations

from dataclasses import dataclass

import hugr.tys as tys
from hugr import val
from hugr.std import _load_extension
from hugr.utils import comma_sep_str

EXTENSION = _load_extension("collections")


def list_type(ty: tys.Type) -> tys.ExtType:
    """Returns a list type with a fixed element type."""
    arg = tys.TypeTypeArg(ty)
    return EXTENSION.types["List"].instantiate([arg])


@dataclass
class ListVal(val.ExtensionValue):
    """Constant value for a list of elements."""

    v: list[val.Value]
    ty: tys.Type

    def __init__(self, v: list[val.Value], elem_ty: tys.Type) -> None:
        self.v = v
        self.ty = list_type(elem_ty)

    def to_value(self) -> val.Extension:
        name = "ListValue"
        # The value list must be serialized at this point, otherwise the
        # `Extension` value would not be serializable.
        vs = [v._to_serial_root() for v in self.v]
        return val.Extension(name, typ=self.ty, val=vs, extensions=[EXTENSION.name])

    def __str__(self) -> str:
        return f"[{comma_sep_str(self.v)}]"
