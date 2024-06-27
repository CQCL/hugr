from __future__ import annotations

from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Iterable,
    Sequence,
    TypeVar,
)

from typing_extensions import Self

import hugr.ops as ops
import hugr.val as val
from hugr.tys import (
    Type,
    TypeRow,
    get_first_sum,
    FunctionType,
    TypeArg,
    FunctionKind,
    PolyFuncType,
    ExtensionSet,
)

from .exceptions import NoSiblingAncestor
from .hugr import Hugr, ParentBuilder
from .node_port import Node, OutPort, Wire, ToNode

if TYPE_CHECKING:
    from .cfg import Cfg
    from .cond_loop import Conditional, If, TailLoop


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

    def _insert_nested_impl(self, builder: ParentBuilder, *args: Wire) -> Node:
        mapping = self.hugr.insert_hugr(builder.hugr, self.parent_node)
        self._wire_up(mapping[builder.parent_node], args)
        return mapping[builder.parent_node]

    def insert_nested(self, dfg: Dfg, *args: Wire) -> Node:
        return self._insert_nested_impl(dfg, *args)

    def add_nested(
        self,
        *args: Wire,
    ) -> Dfg:
        from .dfg import Dfg

        parent_op = ops.DFG(self._wire_types(args))
        dfg = Dfg.new_nested(parent_op, self.hugr, self.parent_node)
        self._wire_up(dfg.parent_node, args)
        return dfg

    def _wire_types(self, args: Iterable[Wire]) -> TypeRow:
        return [self._get_dataflow_type(w) for w in args]

    def add_cfg(
        self,
        *args: Wire,
    ) -> Cfg:
        from .cfg import Cfg

        cfg = Cfg.new_nested(self._wire_types(args), self.hugr, self.parent_node)
        self._wire_up(cfg.parent_node, args)
        return cfg

    def insert_cfg(self, cfg: Cfg, *args: Wire) -> Node:
        return self._insert_nested_impl(cfg, *args)

    def add_conditional(self, cond: Wire, *args: Wire) -> Conditional:
        from .cond_loop import Conditional

        args = (cond, *args)
        (sum_, other_inputs) = get_first_sum(self._wire_types(args))
        cond = Conditional.new_nested(sum_, other_inputs, self.hugr, self.parent_node)
        self._wire_up(cond.parent_node, args)
        return cond

    def insert_conditional(self, cond: Conditional, *args: Wire) -> Node:
        return self._insert_nested_impl(cond, *args)

    def add_if(self, cond: Wire, *args: Wire) -> If:
        from .cond_loop import If

        conditional = self.add_conditional(cond, *args)
        return If(conditional.add_case(1))

    def add_tail_loop(
        self, just_inputs: Sequence[Wire], rest: Sequence[Wire]
    ) -> TailLoop:
        from .cond_loop import TailLoop

        just_input_types = self._wire_types(just_inputs)
        rest_types = self._wire_types(rest)
        parent_op = ops.TailLoop(just_input_types, rest_types)
        tl = TailLoop.new_nested(parent_op, self.hugr, self.parent_node)
        self._wire_up(tl.parent_node, (*just_inputs, *rest))
        return tl

    def insert_tail_loop(self, tl: TailLoop, *args: Wire) -> Node:
        return self._insert_nested_impl(tl, *args)

    def set_outputs(self, *args: Wire) -> None:
        self._wire_up(self.output_node, args)
        self.parent_op._set_out_types(self._output_op().types)

    def add_state_order(self, src: Node, dst: Node) -> None:
        # adds edge to the right of all existing edges
        self.hugr.add_link(src.out(-1), dst.inp(-1))

    def add_const(self, val: val.Value) -> Node:
        return self.hugr.add_const(val, self.parent_node)

    def load(self, const: ToNode | val.Value) -> Node:
        if isinstance(const, val.Value):
            const = self.add_const(const)
        const_op = self.hugr._get_typed_op(const, ops.Const)
        load_op = ops.LoadConst(const_op.val.type_())

        load = self.add(load_op())
        self.hugr.add_link(const.out_port(), load.inp(0))

        return load

    def call(
        self,
        func: ToNode,
        *args: Wire,
        instantiation: FunctionType | None = None,
        type_args: Sequence[TypeArg] | None = None,
    ) -> Node:
        signature = self._fn_sig(func)
        call_op = ops.Call(signature, instantiation, type_args)
        call_n = self.hugr.add_node(call_op, self.parent_node, call_op.num_out)
        self.hugr.add_link(func.out(0), call_n.inp(call_op.function_port_offset()))

        self._wire_up(call_n, args)

        return call_n

    def load_function(
        self,
        func: ToNode,
        instantiation: FunctionType | None = None,
        type_args: Sequence[TypeArg] | None = None,
    ) -> Node:
        signature = self._fn_sig(func)
        load_op = ops.LoadFunc(signature, instantiation, type_args)
        load_n = self.hugr.add_node(load_op, self.parent_node)
        self.hugr.add_link(func.out(0), load_n.inp(0))

        return load_n

    def _fn_sig(self, func: ToNode) -> PolyFuncType:
        f_op = self.hugr[func]
        f_kind = f_op.op.port_kind(func.out(0))
        match f_kind:
            case FunctionKind(sig):
                signature = sig
            case _:
                raise ValueError("Expected 'func' to be a function")
        return signature

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

    def _wire_up_port(self, node: Node, offset: int, p: Wire) -> Type:
        src = p.out_port()
        node_ancestor = _ancestral_sibling(self.hugr, src.node, node)
        if node_ancestor is None:
            raise NoSiblingAncestor(src.node.idx, node.idx)
        if node_ancestor != node:
            self.add_state_order(src.node, node_ancestor)
        self.hugr.add_link(src, node.inp(offset))
        return self._get_dataflow_type(src)


class Dfg(_DfBase[ops.DFG]):
    def __init__(
        self, *input_types: Type, extension_delta: ExtensionSet | None = None
    ) -> None:
        parent_op = ops.DFG(list(input_types), None, extension_delta or [])
        super().__init__(parent_op)


def _ancestral_sibling(h: Hugr, src: Node, tgt: Node) -> Node | None:
    src_parent = h[src].parent

    while (tgt_parent := h[tgt].parent) is not None:
        if tgt_parent == src_parent:
            return tgt
        tgt = tgt_parent

    return None
