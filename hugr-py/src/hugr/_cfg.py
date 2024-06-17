from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from ._hugr import Hugr, Node, Wire
from ._dfg import DfBase, _from_base
from ._tys import FunctionType, TypeRow, Sum
import hugr._ops as ops


class Block(DfBase[ops.DataflowBlock]):
    def block_outputs(self, branching: Wire, *other_outputs: Wire) -> None:
        self.set_outputs(branching, *other_outputs)

    def single_successor_outputs(self, *outputs: Wire) -> None:
        # TODO requires constants
        raise NotImplementedError


@dataclass
class Cfg:
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

    def branch(self, src: Wire, dst: Node) -> None:
        self.hugr.add_link(src.out_port(), dst.inp(0))
