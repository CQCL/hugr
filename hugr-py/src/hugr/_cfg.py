from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import hugr._ops as ops

from ._dfg import DfBase, _from_base
from ._exceptions import NoSiblingAncestor, NotInSameCfg
from ._hugr import Hugr, Node, ParentBuilder, ToNode, Wire
from ._tys import FunctionType, Sum, TypeRow


class Block(DfBase[ops.DataflowBlock]):
    def set_block_outputs(self, branching: Wire, *other_outputs: Wire) -> None:
        self.set_outputs(branching, *other_outputs)

    def set_single_successor_outputs(self, *outputs: Wire) -> None:
        # TODO requires constants
        raise NotImplementedError

    def _wire_up(self, node: Node, ports: Iterable[Wire]):
        for i, p in enumerate(ports):
            src = p.out_port()
            cfg_node = self.hugr[self.root].parent
            assert cfg_node is not None
            src_parent = self.hugr[src.node].parent
            try:
                self._wire_up_port(node, i, p)
            except NoSiblingAncestor:
                # note this just checks if there is a common CFG ancestor
                # it does not check for valid dominance between basic blocks
                # that is deferred to full HUGR validation.
                while cfg_node != src_parent:
                    if src_parent is None or src_parent == self.hugr.root:
                        raise NotInSameCfg(src.node.idx, node.idx)
                    src_parent = self.hugr[src_parent].parent

                self.hugr.add_link(src, node.inp(i))


@dataclass
class Cfg(ParentBuilder):
    hugr: Hugr
    root: Node
    _entry_block: Block
    exit: Node

    def __init__(self, input_types: TypeRow, output_types: TypeRow) -> None:
        root_op = ops.CFG(FunctionType(input=input_types, output=output_types))
        self.hugr = Hugr(root_op)
        self.root = self.hugr.root
        # to ensure entry is first child, add a dummy entry at the start
        self._entry_block = _from_base(
            Block, self.hugr.add_dfg(ops.DataflowBlock(input_types, []))
        )

        self.exit = self.hugr.add_node(ops.ExitBlock(output_types), self.root)

    @property
    def entry(self) -> Node:
        return self._entry_block.root

    def _entry_op(self) -> ops.DataflowBlock:
        dop = self.hugr[self.entry].op
        assert isinstance(dop, ops.DataflowBlock)
        return dop

    def add_entry(self, sum_rows: Sequence[TypeRow], other_outputs: TypeRow) -> Block:
        # update entry block types
        self._entry_op().sum_rows = list(sum_rows)
        self._entry_op().other_outputs = other_outputs
        self._entry_block._output_op().types = [Sum(list(sum_rows)), *other_outputs]
        return self._entry_block

    def simple_entry(self, n_branches: int, other_outputs: TypeRow) -> Block:
        return self.add_entry([[]] * n_branches, other_outputs)

    def add_block(
        self, input_types: TypeRow, sum_rows: Sequence[TypeRow], other_outputs: TypeRow
    ) -> Block:
        new_block = self.hugr.add_dfg(
            ops.DataflowBlock(input_types, list(sum_rows), other_outputs)
        )
        return _from_base(Block, new_block)

    def simple_block(
        self, input_types: TypeRow, n_branches: int, other_outputs: TypeRow
    ) -> Block:
        return self.add_block(input_types, [[]] * n_branches, other_outputs)

    def branch(self, src: Wire, dst: ToNode) -> None:
        self.hugr.add_link(src.out_port(), dst.inp(0))

    def _replace_hugr(self, mapping: Mapping[Node, Node], new_hugr: Hugr) -> None:
        self.hugr = new_hugr
        self.root = mapping[self.root]

        self.exit = mapping[self.exit]
        self._entry_block._replace_hugr(mapping, new_hugr)
