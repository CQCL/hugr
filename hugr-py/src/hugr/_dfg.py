from __future__ import annotations

from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Iterable,
    TypeVar,
)

from typing_extensions import Self

import hugr._ops as ops
import hugr._val as val
from hugr._tys import Type, TypeRow

from ._exceptions import NoSiblingAncestor
from ._hugr import Hugr, Node, OutPort, ParentBuilder, ToNode, Wire

if TYPE_CHECKING:
    from ._cfg import Cfg


DP = TypeVar("DP", bound=ops.DfParentOp)


@dataclass()
class _DfBase(ParentBuilder[DP]):
    hugr: Hugr
    parent_node: Node
    input_node: Node
    output_node: Node

    def __init__(self, parent_op: DP) -> None:
        self.hugr = Hugr(parent_op)
        self.parent_node = self.hugr.root
        self._init_io_nodes(parent_op)

    def _init_io_nodes(self, parent_op: DP):
        inputs = parent_op._inputs()

        self.input_node = self.hugr.add_node(
            ops.Input(inputs), self.parent_node, len(inputs)
        )
        self.output_node = self.hugr.add_node(ops.Output(), self.parent_node)

    @classmethod
    def new_nested(
        cls, parent_op: DP, hugr: Hugr, parent: ToNode | None = None
    ) -> Self:
        new = cls.__new__(cls)

        new.hugr = hugr
        new.parent_node = hugr.add_node(parent_op, parent or hugr.root)
        new._init_io_nodes(parent_op)
        return new

    def _input_op(self) -> ops.Input:
        return self.hugr._get_typed_op(self.input_node, ops.Input)

    def _output_op(self) -> ops.Output:
        return self.hugr._get_typed_op(self.output_node, ops.Output)

    def inputs(self) -> list[OutPort]:
        return [self.input_node.out(i) for i in range(len(self._input_op().types))]

    def add_op(self, op: ops.DataflowOp, /, *args: Wire) -> Node:
        new_n = self.hugr.add_node(op, self.parent_node)
        self._wire_up(new_n, args)

        return replace(new_n, _num_out_ports=op.num_out)

    def add(self, com: ops.Command) -> Node:
        return self.add_op(com.op, *com.incoming)

    def insert_nested(self, dfg: Dfg, *args: Wire) -> Node:
        mapping = self.hugr.insert_hugr(dfg.hugr, self.parent_node)
        self._wire_up(mapping[dfg.parent_node], args)
        return mapping[dfg.parent_node]

    def add_nested(
        self,
        *args: Wire,
    ) -> Dfg:
        from ._dfg import Dfg

        input_types = [self._get_dataflow_type(w) for w in args]

        parent_op = ops.DFG(list(input_types))
        dfg = Dfg.new_nested(parent_op, self.hugr, self.parent_node)
        self._wire_up(dfg.parent_node, args)
        return dfg

    def add_cfg(
        self,
        *args: Wire,
    ) -> Cfg:
        from ._cfg import Cfg

        input_types = [self._get_dataflow_type(w) for w in args]

        cfg = Cfg.new_nested(input_types, self.hugr, self.parent_node)
        self._wire_up(cfg.parent_node, args)
        return cfg

    def insert_cfg(self, cfg: Cfg, *args: Wire) -> Node:
        mapping = self.hugr.insert_hugr(cfg.hugr, self.parent_node)
        self._wire_up(mapping[cfg.parent_node], args)
        return mapping[cfg.parent_node]

    def set_outputs(self, *args: Wire) -> None:
        self._wire_up(self.output_node, args)
        self.parent_op._set_out_types(self._output_op().types)

    def add_state_order(self, src: Node, dst: Node) -> None:
        # adds edge to the right of all existing edges
        self.hugr.add_link(src.out(-1), dst.inp(-1))

    def add_const(self, val: val.Value) -> Node:
        return self.hugr.add_const(val, self.parent_node)

    def load_const(self, const_node: ToNode) -> Node:
        const_op = self.hugr._get_typed_op(const_node, ops.Const)
        load_op = ops.LoadConst(const_op.val.type_())

        load = self.add(load_op())
        self.hugr.add_link(const_node.out_port(), load.inp(0))

        return load

    def add_load_const(self, val: val.Value) -> Node:
        return self.load_const(self.add_const(val))

    def _wire_up(self, node: Node, ports: Iterable[Wire]) -> TypeRow:
        tys = [self._wire_up_port(node, i, p) for i, p in enumerate(ports)]
        if isinstance(op := self.hugr[node].op, ops.PartialOp):
            op.set_in_types(tys)
        return tys

    def _get_dataflow_type(self, wire: Wire) -> Type:
        port = wire.out_port()
        ty = self.hugr.port_type(port)
        if ty is None:
            raise ValueError(f"Port {port} is not a dataflow port.")
        return ty

    def _wire_up_port(self, node: Node, offset: int, p: Wire):
        src = p.out_port()
        node_ancestor = _ancestral_sibling(self.hugr, src.node, node)
        if node_ancestor is None:
            raise NoSiblingAncestor(src.node.idx, node.idx)
        if node_ancestor != node:
            self.add_state_order(src.node, node_ancestor)
        self.hugr.add_link(src, node.inp(offset))
        return self._get_dataflow_type(src)


class Dfg(_DfBase[ops.DFG]):
    def __init__(self, *input_types: Type) -> None:
        parent_op = ops.DFG(list(input_types))
        super().__init__(parent_op)


def _ancestral_sibling(h: Hugr, src: Node, tgt: Node) -> Node | None:
    src_parent = h[src].parent

    while (tgt_parent := h[tgt].parent) is not None:
        if tgt_parent == src_parent:
            return tgt
        tgt = tgt_parent

    return None
