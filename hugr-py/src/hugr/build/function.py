"""Builder classes for defining functions and modules in HUGR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import ops
from hugr.build.dfg import DefinitionBuilder, Function
from hugr.hugr import Hugr

if TYPE_CHECKING:
    from hugr.hugr.node_port import Node
    from hugr.tys import PolyFuncType, TypeBound, TypeRow

__all__ = ["Function", "Module"]


@dataclass
class Module(DefinitionBuilder[ops.Module]):
    """Build a top-level HUGR module.

    Examples:
        >>> m = Module()
        >>> m.hugr.entrypoint_op()
        Module()
    """

    hugr: Hugr[ops.Module]

    def __init__(self) -> None:
        self.hugr = Hugr(ops.Module())

    def define_main(self, input_types: TypeRow) -> Function:
        """Define the 'main' function in the module. See :meth:`define_function`."""
        return self.define_function("main", input_types)

    def declare_function(self, name: str, signature: PolyFuncType) -> Node:
        """Add a function declaration to the module.

        Args:
            name: The name of the function.
            signature: The (polymorphic) signature of the function.

        Returns:
            The node representing the function declaration.

        Examples:
            >>> from hugr.function import Module
            >>> m = Module()
            >>> sig = tys.PolyFuncType([], tys.FunctionType.empty())
            >>> m.declare_function("f", sig)
            Node(1)
        """
        return self.hugr.add_node(ops.FuncDecl(name, signature), self.hugr.entrypoint)

    def add_alias_decl(self, name: str, bound: TypeBound) -> Node:
        """Add a type alias declaration."""
        return self.hugr.add_node(ops.AliasDecl(name, bound), self.hugr.entrypoint)

    @property
    def metadata(self) -> dict[str, object]:
        """Metadata associated with this module."""
        return self.hugr.entrypoint.metadata
