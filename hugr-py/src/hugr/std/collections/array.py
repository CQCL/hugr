"""Collection types and operations."""

from __future__ import annotations

from dataclasses import dataclass

import hugr.tys as tys
from hugr.std import _load_extension

EXTENSION = _load_extension("collections.array")


@dataclass(eq=False)
class Array(tys.ExtType):
    """Fixed `size` array of `ty` elements."""

    def __init__(self, ty: tys.Type, size: int | tys.BoundedNatArg) -> None:
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
        return self.ty.type_bound()
