"""Static array types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from hugr import tys, val
from hugr.std import _load_extension
from hugr.utils import comma_sep_str

EXTENSION = _load_extension("collections.static_array")


@dataclass(eq=False)
class StaticArray(tys.ExtType):
    """Static array of `ty` elements."""

    def __init__(self, ty: tys.Type) -> None:
        ty_arg = tys.TypeTypeArg(ty)

        if (
            tys.TypeBound.join(ty.type_bound(), tys.TypeBound.Copyable)
            != tys.TypeBound.Copyable
        ):
            msg = f"Static array elements must be copyable, not {ty}"
            raise ValueError(msg)

        self.type_def = EXTENSION.types["static_array"]
        self.args = [ty_arg]

    @property
    def ty(self) -> tys.Type:
        assert isinstance(
            self.args[0], tys.TypeTypeArg
        ), "Static array elements must have a valid type"
        return self.args[0].ty

    def type_bound(self) -> tys.TypeBound:
        return tys.TypeBound.Copyable


@dataclass
class StaticArrayVal(val.ExtensionValue):
    """Constant value for a static array."""

    v: list[val.Value]
    ty: StaticArray
    name: str

    def __init__(self, v: list[val.Value], elem_ty: tys.Type, name: str) -> None:
        self.v = v
        self.ty = StaticArray(elem_ty)
        self.name = name

    def to_value(self) -> val.Extension:
        name = "StaticArrayValue"
        # The value list must be serialized at this point, otherwise the
        # `Extension` value would not be serializable.
        vs = [v._to_serial_root() for v in self.v]
        element_ty = self.ty.ty._to_serial_root()
        serial_val = {"values": vs, "typ": element_ty, "name": self.name}
        return val.Extension(
            name, typ=self.ty, val=serial_val, extensions=[EXTENSION.name]
        )

    def __str__(self) -> str:
        return f"static_array({comma_sep_str(self.v)})"
