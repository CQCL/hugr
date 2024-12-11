"""Collection types and operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import hugr.tys as tys
from hugr.std import _load_extension

EXTENSION = _load_extension("collections.array")


@dataclass(eq=False)
class Array(tys.ExtType):
    """Fixed `size` array of `ty` elements."""

    ty: tys.Type = field(default_factory=lambda: tys.Unit)
    size: int = field(default=0)

    def __init__(
        self, ty: tys.Type | tys.TypeTypeArg, size: int | tys.BoundedNatArg
    ) -> None:
        if isinstance(size, int):
            size = tys.BoundedNatArg(size)
        if isinstance(ty, tys.Type):
            ty = tys.TypeTypeArg(ty)

        self.type_def = EXTENSION.types["array"]
        self.args = [size, ty]
        self.size = size.n
        self.ty = ty.ty

    def type_bound(self) -> tys.TypeBound:
        return self.ty.type_bound()
