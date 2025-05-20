"""Collection types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import hugr.model as model
from hugr import tys, val
from hugr.std import _load_extension
from hugr.utils import comma_sep_str

EXTENSION = _load_extension("collections.array")


@dataclass(eq=False)
class Array(tys.ExtType):
    """Fixed `size` array of `ty` elements."""

    def __init__(self, ty: tys.Type, size: int | tys.TypeArg) -> None:
        if isinstance(size, int):
            size = tys.BoundedNatArg(size)

        err_msg = f"Array size must be a bounded natural or a nat variable, not {size}"
        match size:
            case tys.BoundedNatArg(_n):
                pass
            case tys.VariableArg(_idx, param):
                if not isinstance(param, tys.BoundedNatParam):
                    raise ValueError(err_msg)  # noqa: TRY004
            case _:
                raise ValueError(err_msg)

        ty_arg = tys.TypeTypeArg(ty)

        self.type_def = EXTENSION.types["array"]
        self.args = [size, ty_arg]

    @property
    def ty(self) -> tys.Type:
        assert isinstance(
            self.args[1], tys.TypeTypeArg
        ), "Array elements must have a valid type"
        return self.args[1].ty

    @property
    def size(self) -> int | None:
        """If the array has a concrete size, return it.

        Otherwise, return None.
        """
        if isinstance(self.args[0], tys.BoundedNatArg):
            return self.args[0].n
        return None

    def type_bound(self) -> tys.TypeBound:
        return tys.TypeBound.Any


@dataclass
class ArrayVal(val.ExtensionValue):
    """Constant value for a statically sized array of elements."""

    v: list[val.Value]
    ty: Array

    def __init__(self, v: list[val.Value], elem_ty: tys.Type) -> None:
        self.v = v
        self.ty = Array(elem_ty, len(v))

    def to_value(self) -> val.Extension:
        name = "ArrayValue"
        # The value list must be serialized at this point, otherwise the
        # `Extension` value would not be serializable.
        vs = [v._to_serial_root() for v in self.v]
        element_ty = self.ty.ty._to_serial_root()
        serial_val = {"values": vs, "typ": element_ty}
        return val.Extension(name, typ=self.ty, val=serial_val)

    def __str__(self) -> str:
        return f"array({comma_sep_str(self.v)})"

    def to_model(self) -> model.Term:
        return model.Apply(
            "collections.array.const",
            [
                model.Literal(len(self.v)),
                cast(model.Term, self.ty.ty.to_model()),
                model.List([value.to_model() for value in self.v]),
            ],
        )
