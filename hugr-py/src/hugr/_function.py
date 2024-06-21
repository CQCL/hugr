from __future__ import annotations

from dataclasses import dataclass

import hugr._ops as ops
import hugr._val as val

from ._dfg import _DfBase
from hugr._node_port import Node
from ._hugr import Hugr
from ._tys import TypeRow, TypeParam, PolyFuncType


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
