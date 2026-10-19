"""HUGR pointer operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import tys
from hugr.std._util import _load_extension

if TYPE_CHECKING:
    from hugr.ext import ExtensionRegistry, UsedExtensionResolver

EXTENSION = _load_extension("ptr")

PTR_T_DEF = EXTENSION.types["ptr"]


@dataclass(eq=False)
class Ptr(tys.ExtType):
    """Linear handle to a shared mutable cell with a fixed element type."""

    def __init__(self, ty: tys.Type) -> None:
        ty_arg = tys.TypeTypeArg(ty)

        self.type_def = EXTENSION.types["ptr"]
        self.args = [ty_arg]

    @property
    def ty(self) -> tys.Type:
        """Returns the type of the pointer."""
        assert isinstance(
            self.args[0], tys.TypeTypeArg
        ), "Pointer elements must have a valid type"
        return self.args[0].ty

    def _resolve_used_extensions(
        self, resolver: UsedExtensionResolver, registry: ExtensionRegistry | None = None
    ) -> Ptr:
        ext_type = super()._resolve_used_extensions(resolver, registry)

        assert isinstance(
            ext_type, tys.ExtType
        ), "HUGR internal error, expected resolved type to be extension type."
        assert (
            ext_type.type_def == EXTENSION.types["ptr"]
        ), "HUGR internal error, expected resolved type to be pointer."

        ptr = Ptr(tys.Unit)
        ptr.args = ext_type.args
        return ptr
