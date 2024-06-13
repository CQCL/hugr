from __future__ import annotations
from dataclasses import dataclass, replace
from typing import (
    Iterator,
    Iterable,
    TYPE_CHECKING,
    Generic,
    TypeVar,
    cast,
)
from ._hugr import Hugr, Node, Wire, OutPort, ParentBuilder

from typing_extensions import Self
import hugr._ops as ops
from hugr._tys import FunctionType, TypeRow

from ._exceptions import NoSiblingAncestor
from ._hugr import ToNode
from hugr._tys import Type

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
        return self.hugr._get_typed_op(self.input_node, ops.Input)

    def _output_op(self) -> ops.Output:
        return self.hugr._get_typed_op(self.output_node, ops.Output)

    def root_op(self) -> DP:
        return cast(DP, self.hugr[self.root].op)

    def inputs(self) -> list[OutPort]:
        return [self.input_node.out(i) for i in range(len(self._input_op().types))]

    def add_op(self, op: ops.DataflowOp, /, *args: Wire) -> Node:
        new_n = self.hugr.add_node(op, self.root)
        self._wire_up(new_n, args)

        return replace(new_n, _num_out_ports=op.num_out)

    def add(self, com: ops.Command) -> Node:
        return self.add_op(com.op, *com.incoming)

    def insert_nested(self, dfg: Dfg, *args: Wire) -> Node:
        mapping = self.hugr.insert_hugr(dfg.hugr, self.root)
        self._wire_up(mapping[dfg.root], args)
        return mapping[dfg.root]

    def add_nested(
        self,
        *args: Wire,
    ) -> Dfg:
        from ._dfg import Dfg

        _, input_types = zip(*self._get_dataflow_types(args)) if args else ([], [])

        root_op = ops.DFG(FunctionType(input=list(input_types), output=[]))
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
        self.root_op()._set_out_types(self._output_op().types)

    def add_state_order(self, src: Node, dst: Node) -> None:
        # adds edge to the right of all existing edges
        self.hugr.add_link(src.out(-1), dst.inp(-1))

    def _wire_up(self, node: Node, ports: Iterable[Wire]):
        tys = []
        for i, (p, ty) in enumerate(self._get_dataflow_types(ports)):
            tys.append(ty)
            self._wire_up_port(node, i, p)
        if isinstance(op := self.hugr[node].op, ops.DataflowOp):
            op._set_in_types(tys)

    def _get_dataflow_types(self, wires: Iterable[Wire]) -> Iterator[tuple[Wire, Type]]:
        for w in wires:
            port = w.out_port()
            ty = self.hugr.port_type(port)
            if ty is None:
                raise ValueError(f"Port {port} is not a dataflow port.")
            yield w, ty

    def _wire_up_port(self, node: Node, offset: int, p: Wire):
        src = p.out_port()
        node_ancestor = _ancestral_sibling(self.hugr, src.node, node)
        if node_ancestor is None:
            raise NoSiblingAncestor(src.node.idx, node.idx)
        if node_ancestor != node:
            self.add_state_order(src.node, node_ancestor)
        self.hugr.add_link(src, node.inp(offset))


class Dfg(_DfBase[ops.DFG]):
    def __init__(self, *input_types: Type) -> None:
        root_op = ops.DFG(FunctionType(input=list(input_types), output=[]))
        super().__init__(root_op)


def _ancestral_sibling(h: Hugr, src: Node, tgt: Node) -> Node | None:
    src_parent = h[src].parent

    while (tgt_parent := h[tgt].parent) is not None:
        if tgt_parent == src_parent:
            return tgt
        tgt = tgt_parent

    return None
