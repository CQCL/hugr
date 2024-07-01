"""Builder classes for defining functions and modules in HUGR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import ops, val

from .dfg import _DfBase
from .hugr import Hugr

if TYPE_CHECKING:
    from hugr.node_port import Node

    from .tys import PolyFuncType, Type, TypeBound, TypeParam, TypeRow


@dataclass
class Function(_DfBase[ops.FuncDefn]):
    """Build a function definition as a HUGR dataflow graph.

    Args:
        name: The name of the function.
        input_types: The input types for the function (output types are
        computed by propagating types from input node through the graph).
        type_params: The type parameters for the function, if polymorphic.

    Examples:
        >>> f = Function("f", [tys.Bool])
        >>> f.parent_op
        FuncDefn(name='f', inputs=[Bool], params=[])
    """

    def __init__(
        self,
        name: str,
        input_types: TypeRow,
        type_params: list[TypeParam] | None = None,
    ) -> None:
        root_op = ops.FuncDefn(name, input_types, type_params or [])
        super().__init__(root_op)


@dataclass
class Module:
    """Build a top-level HUGR module.

    Examples:
        >>> m = Module()
        >>> m.hugr.root_op()
        Module()
    """

    hugr: Hugr[ops.Module]

    def __init__(self) -> None:
        self.hugr = Hugr(ops.Module())

    def define_function(
        self,
        name: str,
        input_types: TypeRow,
        type_params: list[TypeParam] | None = None,
    ) -> Function:
        """Start building a function definition in the module.

        Args:
            name: The name of the function.
            input_types: The input types for the function.
            type_params: The type parameters for the function, if polymorphic.

        Returns:
            The new function builder.
        """
        parent_op = ops.FuncDefn(name, input_types, type_params or [])
        return Function.new_nested(parent_op, self.hugr)

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
            >>> m = Module()
            >>> sig = tys.PolyFuncType([], tys.FunctionType.empty())
            >>> m.declare_function("f", sig)
            Node(1)
        """
        return self.hugr.add_node(ops.FuncDecl(name, signature), self.hugr.root)

    def add_const(self, value: val.Value) -> Node:
        """Add a static constant to the module.

        Args:
            value: The constant value to add.

        Returns:
            The node holding the constant.

        Examples:
            >>> m = Module()
            >>> m.add_const(val.FALSE)
            Node(1)
        """
        return self.hugr.add_node(ops.Const(value), self.hugr.root)

    def add_alias_defn(self, name: str, ty: Type) -> Node:
        """Add a type alias definition."""
        return self.hugr.add_node(ops.AliasDefn(name, ty), self.hugr.root)

    def add_alias_decl(self, name: str, bound: TypeBound) -> Node:
        """Add a type alias declaration."""
        return self.hugr.add_node(ops.AliasDecl(name, bound), self.hugr.root)
