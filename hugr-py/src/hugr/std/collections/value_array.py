"""Value array types and operations."""

from __future__ import annotations

from dataclasses import dataclass

from hugr import tys, val
from hugr.std import _load_extension
from hugr.utils import comma_sep_str

EXTENSION = _load_extension("collections.value_array")


@dataclass(eq=False)
class ValueArray(tys.ExtType):
    """Fixed `size` array of `ty` elements that is copyable if its elements are."""

    def __init__(self, ty: tys.Type, size: int | tys.TypeArg) -> None:
        if isinstance(size, int):
            size = tys.BoundedNatArg(size)

        err_msg = (
            f"ValueArray size must be a bounded natural or a nat variable, not {size}"
        )
        match size:
            case tys.BoundedNatArg(_n):
                pass
            case tys.VariableArg(_idx, param):
                if not isinstance(param, tys.BoundedNatParam):
                    raise ValueError(err_msg)  # noqa: TRY004
            case _:
                raise ValueError(err_msg)

        ty_arg = tys.TypeTypeArg(ty)

        self.type_def = EXTENSION.types["value_array"]
        self.args = [size, ty_arg]

    @property
    def ty(self) -> tys.Type:
        assert isinstance(
            self.args[1], tys.TypeTypeArg
        ), "ValueArray elements must have a valid type"
        return self.args[1].ty

    @property
    def size(self) -> int | None:
        """If the value array has a concrete size, return it.

        Otherwise, return None.
        """
        if isinstance(self.args[0], tys.BoundedNatArg):
            return self.args[0].n
        return None

    def type_bound(self) -> tys.TypeBound:
        return self.ty.type_bound()


@dataclass
class ValueArrayVal(val.ExtensionValue):
    """Constant value for a statically sized value array of elements."""

    v: list[val.Value]
    ty: ValueArray

    def __init__(self, v: list[val.Value], elem_ty: tys.Type) -> None:
        self.v = v
        self.ty = ValueArray(elem_ty, len(v))

    def to_value(self) -> val.Extension:
        name = "VArrayValue"
        # The value list must be serialized at this point, otherwise the
        # `Extension` value would not be serializable.
        vs = [v._to_serial_root() for v in self.v]
        element_ty = self.ty.ty._to_serial_root()
        serial_val = {"values": vs, "typ": element_ty}
        return val.Extension(name, typ=self.ty, val=serial_val)

    def __str__(self) -> str:
        return f"value_array({comma_sep_str(self.v)})"
