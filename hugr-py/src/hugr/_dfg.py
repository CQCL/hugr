from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Iterable
from ._hugr import Hugr, Node, Wire, OutPort

from ._ops import Op, Command, Input, Output, DFG
from hugr.serialization.tys import FunctionType, Type


@dataclass()
class Dfg:
    hugr: Hugr
    root: Node
    input_node: Node
    output_node: Node

    def __init__(
        self, input_types: Sequence[Type], output_types: Sequence[Type]
    ) -> None:
        input_types = list(input_types)
        output_types = list(output_types)
        root_op = DFG(FunctionType(input=input_types, output=output_types))
        self.hugr = Hugr(root_op)
        self.root = self.hugr.root
        self.input_node = self.hugr.add_node(
            Input(input_types), self.root, len(input_types)
        )
        self.output_node = self.hugr.add_node(Output(output_types), self.root)

    @classmethod
    def endo(cls, types: Sequence[Type]) -> Dfg:
        return Dfg(types, types)

    def _input_op(self) -> Input:
        dop = self.hugr[self.input_node].op
        assert isinstance(dop, Input)
        return dop

    def inputs(self) -> list[OutPort]:
        return [self.input_node.out(i) for i in range(len(self._input_op().types))]

    def add_op(self, op: Op, /, *args: Wire, num_outs: int | None = None) -> Node:
        new_n = self.hugr.add_node(op, self.root, num_outs=num_outs)
        self._wire_up(new_n, args)
        return new_n

    def add(self, com: Command) -> Node:
        return self.add_op(com.op, *com.incoming, num_outs=com.op.num_out)

    def insert_nested(self, dfg: Dfg, *args: Wire) -> Node:
        mapping = self.hugr.insert_hugr(dfg.hugr, self.root)
        self._wire_up(mapping[dfg.root], args)
        return mapping[dfg.root]

    def add_nested(
        self,
        input_types: Sequence[Type],
        output_types: Sequence[Type],
        *args: Wire,
    ) -> Dfg:
        dfg = self.hugr.add_dfg(input_types, output_types)
        self._wire_up(dfg.root, args)
        return dfg

    def set_outputs(self, *args: Wire) -> None:
        self._wire_up(self.output_node, args)

    def add_state_order(self, src: Node, dst: Node) -> None:
        # adds edge to the right of all existing edges
        # breaks if further edges are added
        self.hugr.add_link(
            src.out(self.hugr.num_outgoing(src)), dst.inp(self.hugr.num_incoming(dst))
        )

    def _wire_up(self, node: Node, ports: Iterable[Wire]):
        for i, p in enumerate(ports):
            src = p.out_port()
            self.hugr.add_link(src, node.inp(i))
