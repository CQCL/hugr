from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Iterable, TypeVar, cast
from typing_extensions import Self
import hugr._ops as ops
from hugr._tys import FunctionType, TypeRow

from ._exceptions import NoSiblingAncestor
from ._hugr import Hugr, Node, OutPort, ParentBuilder, Wire, ToNode

if TYPE_CHECKING:
    from ._cfg import Cfg


DP = TypeVar("DP", bound=ops.DfParentOp)


@dataclass()
class _DfBase(ParentBuilder, Generic[DP]):
    hugr: Hugr
    root: Node
    input_node: Node
    output_node: Node

    def __init__(self, root_op: DP) -> None:
        self.hugr = Hugr(root_op)
        self.root = self.hugr.root
        self._init_io_nodes(root_op)

    def _init_io_nodes(self, root_op: DP):
        inner_sig = root_op.inner_signature()

        self.input_node = self.hugr.add_node(
            ops.Input(inner_sig.input), self.root, len(inner_sig.input)
        )
        self.output_node = self.hugr.add_node(ops.Output(inner_sig.output), self.root)

    @classmethod
    def new_nested(cls, root_op: DP, hugr: Hugr, parent: ToNode | None = None) -> Self:
        new = cls.__new__(cls)

        new.hugr = hugr
        new.root = hugr.add_node(root_op, parent or hugr.root)
        new._init_io_nodes(root_op)
        return new

    def _input_op(self) -> ops.Input:
        dop = self.hugr[self.input_node].op
        assert isinstance(dop, ops.Input)
        return dop

    def _output_op(self) -> ops.Output:
        dop = self.hugr[self.output_node].op
        assert isinstance(dop, ops.Output)
        return dop

    def root_op(self) -> DP:
        return cast(DP, self.hugr[self.root].op)

    def inputs(self) -> list[OutPort]:
        return [self.input_node.out(i) for i in range(len(self._input_op().types))]

    def add_op(
        self, op: ops.DataflowOp, /, *args: Wire, num_outs: int | None = None
    ) -> Node:
        new_n = self.hugr.add_node(op, self.root, num_outs=num_outs)
        self._wire_up(new_n, args)
        return new_n

    def add(self, com: ops.Command) -> Node:
        return self.add_op(com.op, *com.incoming, num_outs=com.op.num_out)

    def insert_nested(self, dfg: Dfg, *args: Wire) -> Node:
        mapping = self.hugr.insert_hugr(dfg.hugr, self.root)
        self._wire_up(mapping[dfg.root], args)
        return mapping[dfg.root]

    def add_nested(
        self,
        input_types: TypeRow,
        output_types: TypeRow,
        *args: Wire,
    ) -> Dfg:
        from ._dfg import Dfg

        root_op = ops.DFG(FunctionType(input=input_types, output=output_types))
        dfg = Dfg.new_nested(root_op, self.hugr, self.root)
        self._wire_up(dfg.root, args)
        return dfg

    def add_cfg(
        self,
        input_types: TypeRow,
        output_types: TypeRow,
        *args: Wire,
    ) -> Cfg:
        from ._cfg import Cfg

        cfg = Cfg.new_nested(input_types, output_types, self.hugr, self.root)
        self._wire_up(cfg.root, args)
        return cfg

    def insert_cfg(self, cfg: Cfg, *args: Wire) -> Node:
        mapping = self.hugr.insert_hugr(cfg.hugr, self.root)
        self._wire_up(mapping[cfg.root], args)
        return mapping[cfg.root]

    def set_outputs(self, *args: Wire) -> None:
        self._wire_up(self.output_node, args)

    def add_state_order(self, src: Node, dst: Node) -> None:
        # adds edge to the right of all existing edges
        self.hugr.add_link(src.out(-1), dst.inp(-1))

    def _wire_up(self, node: Node, ports: Iterable[Wire]):
        for i, p in enumerate(ports):
            self._wire_up_port(node, i, p)

    def _wire_up_port(self, node: Node, offset: int, p: Wire):
        src = p.out_port()
        node_ancestor = _ancestral_sibling(self.hugr, src.node, node)
        if node_ancestor is None:
            raise NoSiblingAncestor(src.node.idx, node.idx)
        if node_ancestor != node:
            self.add_state_order(src.node, node_ancestor)
        self.hugr.add_link(src, node.inp(offset))


class Dfg(_DfBase[ops.DFG]):
    def __init__(self, input_types: TypeRow, output_types: TypeRow) -> None:
        root_op = ops.DFG(FunctionType(input=input_types, output=output_types))
        super().__init__(root_op)

    @classmethod
    def endo(cls, types: TypeRow) -> Dfg:
        return cls(types, types)


def _ancestral_sibling(h: Hugr, src: Node, tgt: Node) -> Node | None:
    src_parent = h[src].parent

    while (tgt_parent := h[tgt].parent) is not None:
        if tgt_parent == src_parent:
            return tgt
        tgt = tgt_parent

    return None
