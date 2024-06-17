from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import hugr._ops as ops

from ._dfg import _DfBase
from ._exceptions import NoSiblingAncestor, NotInSameCfg
from ._hugr import Hugr, Node, ParentBuilder, ToNode, Wire
from ._tys import FunctionType, Sum, TypeRow, Type


class Block(_DfBase[ops.DataflowBlock]):
    def set_block_outputs(self, branching: Wire, *other_outputs: Wire) -> None:
        self.set_outputs(branching, *other_outputs)

    def set_single_successor_outputs(self, *outputs: Wire) -> None:
        # TODO requires constants
        raise NotImplementedError

    def _wire_up_port(self, node: Node, offset: int, p: Wire) -> Type:
        src = p.out_port()
        cfg_node = self.hugr[self.parent_node].parent
        assert cfg_node is not None
        src_parent = self.hugr[src.node].parent
        try:
            super()._wire_up_port(node, offset, p)
        except NoSiblingAncestor:
            # note this just checks if there is a common CFG ancestor
            # it does not check for valid dominance between basic blocks
            # that is deferred to full HUGR validation.
            while cfg_node != src_parent:
                if src_parent is None or src_parent == self.hugr.root:
                    raise NotInSameCfg(src.node.idx, node.idx)
                src_parent = self.hugr[src_parent].parent

            self.hugr.add_link(src, node.inp(offset))
        return self._get_dataflow_type(src)


@dataclass
class Cfg(ParentBuilder[ops.CFG]):
    hugr: Hugr
    parent_node: Node
    _entry_block: Block
    exit: Node

    def __init__(self, input_types: TypeRow, output_types: TypeRow) -> None:
        root_op = ops.CFG(FunctionType(input=input_types, output=output_types))
        hugr = Hugr(root_op)
        self._init_impl(hugr, hugr.root, input_types, output_types)

    def _init_impl(
        self: Cfg, hugr: Hugr, root: Node, input_types: TypeRow, output_types: TypeRow
    ) -> None:
        self.hugr = hugr
        self.parent_node = root
        # to ensure entry is first child, add a dummy entry at the start
        self._entry_block = Block.new_nested(
            ops.DataflowBlock(input_types, []), hugr, root
        )

        self.exit = self.hugr.add_node(ops.ExitBlock(output_types), self.parent_node)

    @classmethod
    def new_nested(
        cls,
        input_types: TypeRow,
        output_types: TypeRow,
        hugr: Hugr,
        parent: ToNode | None = None,
    ) -> Cfg:
        new = cls.__new__(cls)
        root = hugr.add_node(
            ops.CFG(FunctionType(input=input_types, output=output_types)),
            parent or hugr.root,
        )
        new._init_impl(hugr, root, input_types, output_types)
        return new

    @property
    def entry(self) -> Node:
        return self._entry_block.parent_node

    def _entry_op(self) -> ops.DataflowBlock:
        return self.hugr._get_typed_op(self.entry, ops.DataflowBlock)

    def _exit_op(self) -> ops.ExitBlock:
        return self.hugr._get_typed_op(self.exit, ops.ExitBlock)

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
        new_block = Block.new_nested(
            ops.DataflowBlock(input_types, list(sum_rows), other_outputs),
            self.hugr,
            self.parent_node,
        )
        return new_block

    def simple_block(
        self, input_types: TypeRow, n_branches: int, other_outputs: TypeRow
    ) -> Block:
        return self.add_block(input_types, [[]] * n_branches, other_outputs)

    def branch(self, src: Wire, dst: ToNode) -> None:
        self.hugr.add_link(src.out_port(), dst.inp(0))
