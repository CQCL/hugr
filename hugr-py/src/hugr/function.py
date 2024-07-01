from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hugr.ops as ops
import hugr.val as val

from .dfg import _DfBase
from .hugr import Hugr

if TYPE_CHECKING:
    from hugr.node_port import Node

    from .tys import PolyFuncType, Type, TypeBound, TypeParam, TypeRow


@dataclass
class Function(_DfBase[ops.FuncDefn]):
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
    hugr: Hugr

    def __init__(self) -> None:
        self.hugr = Hugr(ops.Module())

    def define_function(
        self,
        name: str,
        input_types: TypeRow,
        type_params: list[TypeParam] | None = None,
    ) -> Function:
        parent_op = ops.FuncDefn(name, input_types, type_params or [])
        return Function.new_nested(parent_op, self.hugr)

    def define_main(self, input_types: TypeRow) -> Function:
        return self.define_function("main", input_types)

    def declare_function(self, name: str, signature: PolyFuncType) -> Node:
        return self.hugr.add_node(ops.FuncDecl(name, signature), self.hugr.root)

    def add_const(self, value: val.Value) -> Node:
        return self.hugr.add_node(ops.Const(value), self.hugr.root)

    def add_alias_defn(self, name: str, ty: Type) -> Node:
        return self.hugr.add_node(ops.AliasDefn(name, ty), self.hugr.root)

    def add_alias_decl(self, name: str, bound: TypeBound) -> Node:
        return self.hugr.add_node(ops.AliasDecl(name, bound), self.hugr.root)
