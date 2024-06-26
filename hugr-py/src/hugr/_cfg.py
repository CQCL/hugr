from __future__ import annotations

from dataclasses import dataclass

import hugr._ops as ops

from ._dfg import _DfBase
from ._exceptions import NoSiblingAncestor, NotInSameCfg, MismatchedExit
from ._hugr import Hugr, ParentBuilder
from ._node_port import Node, Wire, ToNode
from ._tys import TypeRow, Type
import hugr._val as val


class Block(_DfBase[ops.DataflowBlock]):
    def set_block_outputs(self, branching: Wire, *other_outputs: Wire) -> None:
        self.set_outputs(branching, *other_outputs)

    def set_single_succ_outputs(self, *outputs: Wire) -> None:
        u = self.load(val.Unit)
        self.set_outputs(u, *outputs)

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

    def __init__(self, input_types: TypeRow) -> None:
        root_op = ops.CFG(inputs=input_types)
        hugr = Hugr(root_op)
        self._init_impl(hugr, hugr.root, input_types)

    def _init_impl(self: Cfg, hugr: Hugr, root: Node, input_types: TypeRow) -> None:
        self.hugr = hugr
        self.parent_node = root
        # to ensure entry is first child, add a dummy entry at the start
        self._entry_block = Block.new_nested(ops.DataflowBlock(input_types), hugr, root)

        self.exit = self.hugr.add_node(ops.ExitBlock(), self.parent_node)

    @classmethod
    def new_nested(
        cls,
        input_types: TypeRow,
        hugr: Hugr,
        parent: ToNode | None = None,
    ) -> Cfg:
        new = cls.__new__(cls)
        root = hugr.add_node(
            ops.CFG(inputs=input_types),
            parent or hugr.root,
        )
        new._init_impl(hugr, root, input_types)
        return new

    @property
    def entry(self) -> Node:
        return self._entry_block.parent_node

    @property
    def _entry_op(self) -> ops.DataflowBlock:
        return self.hugr._get_typed_op(self.entry, ops.DataflowBlock)

    @property
    def _exit_op(self) -> ops.ExitBlock:
        return self.hugr._get_typed_op(self.exit, ops.ExitBlock)

    def add_entry(self) -> Block:
        return self._entry_block

    def add_block(self, input_types: TypeRow) -> Block:
        new_block = Block.new_nested(
            ops.DataflowBlock(input_types),
            self.hugr,
            self.parent_node,
        )
        return new_block

    # TODO insert_block

    def add_successor(self, pred: Wire) -> Block:
        b = self.add_block(self._nth_outputs(pred))

        self.branch(pred, b)
        return b

    def _nth_outputs(self, wire: Wire) -> TypeRow:
        port = wire.out_port()
        block = self.hugr._get_typed_op(port.node, ops.DataflowBlock)
        return block.nth_outputs(port.offset)

    def branch(self, src: Wire, dst: ToNode) -> None:
        # TODO check for existing link/type compatibility
        if dst.to_node() == self.exit:
            return self.branch_exit(src)
        src = src.out_port()
        self.hugr.add_link(src, dst.inp(0))

    def branch_exit(self, src: Wire) -> None:
        src = src.out_port()
        self.hugr.add_link(src, self.exit.inp(0))

        out_types = self._nth_outputs(src)
        if self._exit_op._cfg_outputs is not None:
            if self._exit_op._cfg_outputs != out_types:
                raise MismatchedExit(src.node.idx)
        else:
            self._exit_op._cfg_outputs = out_types
            self.parent_op._outputs = out_types
