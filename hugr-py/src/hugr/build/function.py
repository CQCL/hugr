"""Builder classes for defining functions and modules in HUGR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import ops
from hugr.build.dfg import DefinitionBuilder, Function
from hugr.hugr import Hugr

if TYPE_CHECKING:
    from hugr.hugr.node_port import Node
    from hugr.tys import PolyFuncType, Type, TypeBound, TypeParam, TypeRow

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

    def __init__(self, hugr: Hugr | None = None) -> None:
        self.hugr = Hugr(ops.Module()) if hugr is None else hugr

    def define_main(self, input_types: TypeRow) -> Function:
        """Define the 'main' function in the module. See :meth:`define_function`."""
        return self.define_function("main", input_types)

    def define_function(
        self,
        name: str,
        input_types: TypeRow,
        output_types: TypeRow | None = None,
        type_params: list[TypeParam] | None = None,
    ) -> Function:
        """Start building a function definition in the graph.

        Args:
            name: The name of the function.
            input_types: The input types for the function.
            output_types: The output types for the function.
                If not provided, it will be inferred after the function is built.
            type_params: The type parameters for the function, if polymorphic.
            parent: The parent node of the constant. Defaults to the entrypoint node.

        Returns:
            The new function builder.
        """
        parent_op = ops.FuncDefn(name, input_types, type_params or [])
        func = Function.new_nested(parent_op, self.hugr, self.hugr.module_root)
        if output_types is not None:
            func.declare_outputs(output_types)
        return func

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
        return self.hugr.add_node(
            ops.FuncDecl(name, signature), self.hugr.entrypoint, num_outs=1
        )

    def add_alias_defn(self, name: str, ty: Type) -> Node:
        """Add a type alias definition."""
        return self.hugr.add_node(ops.AliasDefn(name, ty), self.hugr.module_root)

    def add_alias_decl(self, name: str, bound: TypeBound) -> Node:
        """Add a type alias declaration."""
        return self.hugr.add_node(ops.AliasDecl(name, bound), self.hugr.module_root)

    @property
    def metadata(self) -> dict[str, object]:
        """Metadata associated with this module."""
        return self.hugr.entrypoint.metadata
