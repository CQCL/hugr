"""Builder classes for HUGR control flow graphs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from typing_extensions import Self

from hugr import ops, tys, val
from hugr.build.base import ParentBuilder
from hugr.build.dfg import DfBase
from hugr.exceptions import MismatchedExit, NoSiblingAncestor, NotInSameCfg
from hugr.hugr import Hugr

if TYPE_CHECKING:
    from hugr.hugr.node_port import Node, PortOffset, ToNode, Wire
    from hugr.tys import Type, TypeRow


class Block(DfBase[ops.DataflowBlock]):
    """Builder class for a basic block in a HUGR control flow graph."""

    def set_outputs(self, *outputs: Wire) -> None:
        assert len(outputs) > 0
        branching = outputs[0]
        branch_type = self.hugr.port_type(branching.out_port())
        assert isinstance(branch_type, tys.Sum)
        self._set_parent_output_count(len(branch_type.variant_rows))

        super().set_outputs(*outputs)

    def set_block_outputs(self, branching: Wire, *other_outputs: Wire) -> None:
        self.set_outputs(branching, *other_outputs)

    def set_single_succ_outputs(self, *outputs: Wire) -> None:
        u = self.load(val.Unit)
        self.set_outputs(u, *outputs)

    def _wire_up_port(self, node: Node, offset: PortOffset, p: Wire) -> Type:
        src = p.out_port()
        cfg_node = self.hugr[self.parent_node].parent
        assert cfg_node is not None
        src_parent = self.hugr[src.node].parent
        try:
            super()._wire_up_port(node, offset, p)
        except NoSiblingAncestor as e:
            # note this just checks if there is a common CFG ancestor
            # it does not check for valid dominance between basic blocks
            # that is deferred to full HUGR validation.
            while cfg_node != src_parent:
                if src_parent is None or src_parent == self.hugr.module_root:
                    raise NotInSameCfg(src.node.idx, node.idx) from e
                src_parent = self.hugr[src_parent].parent

            self.hugr.add_link(src, node.inp(offset))
        return self._get_dataflow_type(src)


@dataclass
class Cfg(ParentBuilder[ops.CFG], AbstractContextManager):
    """Builder class for a HUGR control flow graph, with the HUGR entrypoint node
    being a :class:`CFG <hugr.ops.CFG>`.

    Args:
        input_types: The input types for the CFG. Outputs are computed
        by propagating types through the control flow graph to the exit block.

    Examples:
        >>> cfg = Cfg(tys.Bool, tys.Unit)
        >>> cfg.parent_op
        CFG(inputs=[Bool, Unit])
    """

    #: The HUGR instance this CFG is part of.
    hugr: Hugr
    #: The parent node of the CFG.
    parent_node: Node
    _entry_block: Block
    #: The node holding the root of the exit block.
    exit: Node

    def __init__(self, *input_types: Type) -> None:
        input_typs = list(input_types)
        root_op = ops.CFG(inputs=input_typs)
        hugr = Hugr(root_op)
        self._init_impl(hugr, hugr.entrypoint, input_typs)

    def _init_impl(
        self: Cfg, hugr: Hugr, entrypoint: Node, input_types: TypeRow
    ) -> None:
        self.hugr = hugr
        self.parent_node = entrypoint
        # to ensure entry is first child, add a dummy entry at the start
        self._entry_block = Block.new_nested(
            ops.DataflowBlock(input_types), hugr, entrypoint
        )

        self.exit = self.hugr.add_node(ops.ExitBlock(), self.parent_node)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        return None

    @classmethod
    def new_nested(
        cls,
        input_types: TypeRow,
        hugr: Hugr,
        parent: ToNode | None = None,
    ) -> Cfg:
        """Start building a CFG nested inside an existing HUGR graph.

        Args:
            input_types: The input types for the CFG.
            hugr: The HUGR instance this CFG is part of.
            parent: The parent node for the CFG: defaults to the root of the HUGR
                instance.

        Returns:
            The new CFG builder.

        Examples:
            >>> hugr = Hugr()
            >>> cfg = Cfg.new_nested([tys.Bool], hugr)
            >>> cfg.parent_op
            CFG(inputs=[Bool])
        """
        new = cls.__new__(cls)
        root = hugr.add_node(
            ops.CFG(inputs=input_types),
            parent or hugr.entrypoint,
        )
        new._init_impl(hugr, root, input_types)
        return new

    @property
    def entry(self) -> Node:
        """Node for entry block of the CFG.

        Examples:
            >>> cfg = Cfg(tys.Bool)
            >>> cfg.entry
            Node(1)
        """
        return self._entry_block.parent_node

    @property
    def _entry_op(self) -> ops.DataflowBlock:
        return self.hugr._get_typed_op(self.entry, ops.DataflowBlock)

    @property
    def _exit_op(self) -> ops.ExitBlock:
        return self.hugr._get_typed_op(self.exit, ops.ExitBlock)

    def add_entry(self) -> Block:
        """Start building the entry block of the CFG.

        Returns:
            The entry block builder.

        Examples:
            >>> cfg = Cfg(tys.Bool)
            >>> entry = cfg.add_entry()
            >>> entry.set_outputs(*entry.inputs())
        """
        return self._entry_block

    def add_block(self, *input_types: Type) -> Block:
        """Add a new block to the CFG and start building it.

        Args:
            input_types: The input types for the block.

        Returns:
            The block builder.

        Examples:
            >>> cfg = Cfg(tys.Bool)
            >>> with cfg.add_block(tys.Unit) as b:\
                    b.set_single_succ_outputs(*b.inputs())
        """
        new_block = Block.new_nested(
            ops.DataflowBlock(list(input_types)),
            self.hugr,
            self.parent_node,
        )
        return new_block

    # TODO insert_block

    def add_successor(self, pred: Wire) -> Block:
        """Start building a block that succeeds an existing block.

        Args:
            pred: The wire from the predecessor block to the new block. The
            port of the wire determines the branching index of the new block.


        Returns:
            The new block builder.

        Examples:
            >>> cfg = Cfg(tys.Bool)
            >>> with cfg.add_entry() as entry:\
                    entry.set_single_succ_outputs()
            >>> with cfg.add_successor(entry[0]) as b:\
                    b.set_single_succ_outputs(*b.inputs())
        """
        b = self.add_block(*self._nth_outputs(pred))

        self.branch(pred, b)
        return b

    def _nth_outputs(self, wire: Wire) -> TypeRow:
        port = wire.out_port()
        block = self.hugr._get_typed_op(port.node, ops.DataflowBlock)
        return block.nth_outputs(port.offset)

    def branch(self, src: Wire, dst: ToNode) -> None:
        """Add a branching control flow link between blocks.

        Args:
            src: The wire from the predecessor block.
            dst: The destination block.

        Examples:
            >>> cfg = Cfg(tys.Bool)
            >>> with cfg.add_entry() as entry:\
                    entry.set_single_succ_outputs()
            >>> b = cfg.add_block(tys.Unit)
            >>> cfg.branch(entry[0], b)
        """
        # TODO check for existing link/type compatibility
        if dst.to_node() == self.exit:
            return self.branch_exit(src)
        src = src.out_port()
        self.hugr.add_link(src, dst.inp(0))

    def branch_exit(self, src: Wire) -> None:
        """Branch from a block to the exit block.

        Args:
            src: The wire from the predecessor block.

        Examples:
            >>> cfg = Cfg(tys.Bool)
            >>> with cfg.add_entry() as entry:\
                    entry.set_single_succ_outputs()
            >>> cfg.branch_exit(entry[0])
        """
        src = src.out_port()
        self.hugr.add_link(src, self.exit.inp(0))

        out_types = self._nth_outputs(src)
        if self._exit_op._cfg_outputs is not None:
            if self._exit_op._cfg_outputs != out_types:
                raise MismatchedExit(src.node.idx)
        else:
            self._exit_op._cfg_outputs = out_types
            self.parent_op._outputs = out_types
            self.parent_node = self.hugr._update_node_outs(
                self.parent_node, len(out_types)
            )
            if (
                self.parent_op._entrypoint_requires_wiring
                and self.hugr.entrypoint == self.parent_node
            ):
                self.hugr._connect_df_entrypoint_outputs()
