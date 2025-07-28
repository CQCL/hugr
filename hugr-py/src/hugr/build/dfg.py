"""Builder for HUGR datflow graphs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

from typing_extensions import Self

from hugr import ops, tys, val
from hugr.build.base import ParentBuilder
from hugr.exceptions import NoSiblingAncestor
from hugr.hugr import Hugr

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from hugr.hugr.node_port import Node, OutPort, PortOffset, ToNode, Wire
    from hugr.tys import Type, TypeParam, TypeRow

    from .cfg import Cfg
    from .cond_loop import Conditional, If, TailLoop

OpVar = TypeVar("OpVar", bound=ops.Op)


class DataflowError(Exception):
    """Error building a :class:`DfBase` dataflow graph."""


@dataclass()
class DefinitionBuilder(Generic[OpVar]):
    """Base class for builders that can define functions, constants, and aliases.

    As this class may be a root node, it does not extend `ParentBuilder`.
    """

    hugr: Hugr[OpVar]

    def define_function(
        self,
        name: str,
        input_types: TypeRow,
        output_types: TypeRow | None = None,
        type_params: list[TypeParam] | None = None,
        parent: ToNode | None = None,
    ) -> Function:
        """Start building a function definition in the graph.

        Args:
            name: The name of the function.
            input_types: The input types for the function.
            output_types: The output types for the function.
                If not provided, it will be inferred after the function is built.
            type_params: The type parameters for the function, if polymorphic.
            parent: The parent node of the constant. Defaults to the entrypoint node.

        Returns:
            The new function builder.
        """
        parent_node = parent or self.hugr.entrypoint
        parent_op = ops.FuncDefn(name, input_types, type_params or [])
        func = Function.new_nested(parent_op, self.hugr, parent_node)
        if output_types is not None:
            func.declare_outputs(output_types)
        return func

    def add_const(self, value: val.Value, parent: ToNode | None = None) -> Node:
        """Add a static constant to the graph.

        Args:
            value: The constant value to add.
            parent: The parent node of the constant. Defaults to the entrypoint node.

        Returns:
            The node holding the :class:`Const <hugr.ops.Const>` operation.

        Example:
            >>> dfg = Dfg()
            >>> const_n = dfg.add_const(val.TRUE)
            >>> dfg.hugr[const_n].op
            Const(TRUE)
        """
        parent_node = parent or self.hugr.entrypoint
        return self.hugr.add_node(ops.Const(value), parent_node)

    def add_alias_defn(self, name: str, ty: Type, parent: ToNode | None = None) -> Node:
        """Add a type alias definition."""
        parent_node = parent or self.hugr.entrypoint
        return self.hugr.add_node(ops.AliasDefn(name, ty), parent_node)


DP = TypeVar("DP", bound=ops.DfParentOp)


@dataclass()
class DfBase(ParentBuilder[DP], DefinitionBuilder, AbstractContextManager):
    """Base class for dataflow graph builders.

    Args:
        parent_op: The parent operation of the dataflow graph.
    """

    #: The Hugr instance that the builder is using.
    hugr: Hugr = field(repr=False)
    #: The parent node of the dataflow graph.
    parent_node: Node
    #: The input node of the dataflow graph.
    input_node: Node = field(repr=False)
    #: The output node of the dataflow graph.
    output_node: Node = field(repr=False)

    def __init__(self, parent_op: DP) -> None:
        self.hugr = Hugr(parent_op)
        self.parent_node = self.hugr.entrypoint
        self._init_io_nodes(parent_op)

    def _init_io_nodes(self, parent_op: DP):
        inputs = parent_op._inputs()

        self.input_node = self.hugr.add_node(
            ops.Input(inputs), self.parent_node, len(inputs)
        )
        self.output_node = self.hugr.add_node(ops.Output(), self.parent_node)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        return None

    @classmethod
    def new_nested(
        cls, parent_op: DP, hugr: Hugr, parent: ToNode | None = None
    ) -> Self:
        """Start building a dataflow graph nested inside a larger HUGR.

        Args:
            parent_op: The parent operation of the new dataflow graph.
            hugr: The host HUGR instance to build the dataflow graph in.
            parent: Parent of new dataflow graph's root node: defaults to the
            host HUGR entrypoint.

        Example:
            >>> hugr = Hugr()
            >>> dfg = Dfg.new_nested(ops.DFG([]), hugr)
            >>> dfg.parent_node
            Node(1)
        """
        new = cls.__new__(cls)

        new.hugr = hugr
        new.parent_node = hugr.add_node(parent_op, parent or hugr.entrypoint)
        new._init_io_nodes(parent_op)
        return new

    @classmethod
    def _new_existing(cls, hugr: Hugr, root: ToNode | None = None) -> Self:
        """Start a dataflow graph builder for an existing node.

        Args:
            hugr: The host HUGR instance to build the dataflow graph in.
            root: The dataflow graph's root node.
                Defaults to the host HUGR's entrypoint.

        Example:
            >>> hugr = Hugr(ops.DFG([]))
            >>> dfg = Dfg._new_existing(hugr)
            >>> dfg.parent_node
            Node(4)

        Raises:
            :class:`DataflowError` if the `root` operation is not a dataflow
            parent.
        """
        root = root or hugr.entrypoint

        if not ops.is_df_parent_op(hugr[root].op):
            msg = f"{hugr[root].op} is not a dataflow parent"
            raise DataflowError(msg)

        new = cls.__new__(cls)
        new.hugr = hugr
        new.parent_node = root.to_node()
        [inp, out] = hugr.children(root)[:2]
        new.input_node = inp
        new.output_node = out

        return new

    def _input_op(self) -> ops.Input:
        return self.hugr._get_typed_op(self.input_node, ops.Input)

    def _output_op(self) -> ops.Output:
        return self.hugr._get_typed_op(self.output_node, ops.Output)

    def inputs(self) -> list[OutPort]:
        """List all incoming wires (output ports of the input node).

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> dfg.inputs()
            [OutPort(Node(1), 0)]
        """
        return [self.input_node.out(i) for i in range(len(self._input_op().types))]

    def add_op(
        self, op: ops.DataflowOp, /, *args: Wire, metadata: dict[str, Any] | None = None
    ) -> Node:
        """Add a dataflow operation to the graph, wiring in input ports.

        Args:
            op: The operation to add.
            args: The input wires to the operation.
            metadata: Metadata to attach to the function definition. Defaults to None.

        Returns:
            The node holding the new operation.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> dfg.add_op(ops.Noop(), dfg.inputs()[0])
            Node(3)
        """
        new_n = self.hugr.add_node(op, self.parent_node, metadata=metadata)
        self._wire_up(new_n, args)
        new_n._num_out_ports = op.num_out
        return new_n

    def add(self, com: ops.Command, *, metadata: dict[str, Any] | None = None) -> Node:
        """Add a command (holding a dataflow operation and the incoming wires)
        to the graph.

        Args:
            com: The command to add.
            metadata: Metadata to attach to the function definition. Defaults to None.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> (i,) = dfg.inputs()
            >>> dfg.add(ops.Noop()(i))
            Node(3)

        """

        def raise_no_ints():
            error_message = "Command used with Dfg must hold Wire, not integer indices."
            raise ValueError(error_message)

        wires = (
            (w if not isinstance(w, int) else raise_no_ints()) for w in com.incoming
        )
        return self.add_op(com.op, *wires, metadata=metadata)

    def extend(self, *coms: ops.Command) -> list[Node]:
        """Add a series of commands to the DFG.

        Shorthand for calling :meth:`add` on each command in `coms`.

        Args:
            coms: Commands to add.

        Returns:
            List of the new nodes in the same order as the commands.

        Raises:
            IndexError: If any input index is not a tracked wire.

        Examples:
            >>> dfg = Dfg(tys.Bool, tys.Unit)
            >>> (b, u) = dfg.inputs()
            >>> dfg.extend(ops.Noop()(b), ops.Noop()(u))
            [Node(3), Node(4)]
        """
        return [self.add(com) for com in coms]

    def _insert_nested_impl(self, builder: ParentBuilder, *args: Wire) -> Node:
        mapping = self.hugr.insert_hugr(builder.hugr, self.parent_node)
        self._wire_up(mapping[builder.parent_node], args)
        return mapping[builder.parent_node]

    def insert_nested(self, dfg: Dfg, *args: Wire) -> Node:
        """Insert a nested dataflow graph into the current graph, wiring in the
        inputs.

        Args:
            dfg: The dataflow graph to insert.
            args: The input wires to the graph.

        Returns:
            The entrypoint node of the inserted graph.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> dfg2 = Dfg(tys.Bool)
            >>> dfg.insert_nested(dfg2, dfg.inputs()[0])
            Node(3)
        """
        return self._insert_nested_impl(dfg, *args)

    def add_nested(
        self,
        *args: Wire,
    ) -> Dfg:
        """Start building a nested dataflow graph.

        Args:
            args: The input wires to the nested DFG.

        Returns:
            Builder for new nested dataflow graph.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> with dfg.add_nested(dfg.inputs()[0]) as dfg2:\
                   dfg2.parent_node
            Node(3)
        """
        from .dfg import Dfg

        parent_op = ops.DFG(self._wire_types(args))
        dfg = Dfg.new_nested(parent_op, self.hugr, self.parent_node)
        self._wire_up(dfg.parent_node, args)
        return dfg

    def _wire_types(self, args: Iterable[Wire]) -> tys.TypeRow:
        return [self._get_dataflow_type(w) for w in args]

    def add_cfg(
        self,
        *args: Wire,
    ) -> Cfg:
        """Start building a new CFG nested inside the current dataflow graph.

        Args:
            args: The input wires to the new CFG.

        Returns:
            Builder for new nested CFG.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> with dfg.add_cfg(dfg.inputs()[0]) as cfg:\
                    cfg.parent_op
            CFG(inputs=[Bool])
        """
        from .cfg import Cfg

        cfg = Cfg.new_nested(self._wire_types(args), self.hugr, self.parent_node)
        self._wire_up(cfg.parent_node, args)
        return cfg

    def insert_cfg(self, cfg: Cfg, *args: Wire) -> Node:
        """Insert a CFG into the current dataflow graph, wiring in the inputs.

        Args:
            cfg: The CFG to insert.
            args: The input wires to the CFG.

        Returns:
            The root node of the inserted CFG.

        Example:
            >>> from hugr.cfg import Cfg
            >>> dfg = Dfg(tys.Bool)
            >>> cfg = Cfg(tys.Bool)
            >>> dfg.insert_cfg(cfg, dfg.inputs()[0])
            Node(3)
        """
        return self._insert_nested_impl(cfg, *args)

    def add_conditional(self, cond_wire: Wire, *args: Wire) -> Conditional:
        """Start building a new conditional nested inside the current dataflow
        graph.

        Args:
            cond_wire: The wire holding the value (of Sum type) to branch the
            conditional on.
            args: Remaining input wires to the conditional.

        Returns:
            Builder for new nested conditional.

        Example:
            >>> dfg = Dfg(tys.Bool, tys.Unit)
            >>> (cond, unit) = dfg.inputs()
            >>> cond = dfg.add_conditional(cond, unit)
            >>> cond.parent_node
            Node(3)
        """
        from .cond_loop import Conditional

        args = (cond_wire, *args)
        (sum_, other_inputs) = tys.get_first_sum(self._wire_types(args))
        cond_wire = Conditional.new_nested(
            sum_, other_inputs, self.hugr, self.parent_node
        )
        self._wire_up(cond_wire.parent_node, args)
        return cond_wire

    def insert_conditional(
        self, cond: Conditional, cond_wire: Wire, *args: Wire
    ) -> Node:
        """Insert a conditional into the current dataflow graph, wiring in the
        inputs.

        Args:
            cond: The conditional to insert.
            cond_wire: The wire holding the value (of Sum type)
              to branch the Conditional on.
            args: Remaining input wires to the conditional.

        Returns:
            The root node of the inserted conditional.

        Example:
            >>> from hugr.cond_loop import Conditional
            >>> cond = Conditional(tys.Bool, [])
            >>> dfg = Dfg(tys.Bool)
            >>> cond_n = dfg.insert_conditional(cond, dfg.inputs()[0])
            >>> dfg.hugr[cond_n].op
            Conditional(sum_ty=Bool, other_inputs=[])
        """
        return self._insert_nested_impl(cond, *(cond_wire, *args))

    def add_if(self, cond_wire: Wire, *args: Wire) -> If:
        """Start building a new if block nested inside the current dataflow
        graph.

        Args:
            cond_wire: The wire holding the Bool value to branch the If on.
            args: Remaining input wires to the If (and subsequent Else).

        Returns:
            Builder for new nested If.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> (cond,) = dfg.inputs()
            >>> if_ = dfg.add_if(cond, cond)
            >>> if_.parent_op
            Case(inputs=[Bool])
        """
        from .cond_loop import If

        conditional = self.add_conditional(cond_wire, *args)
        return If(conditional.add_case(1))

    def add_tail_loop(
        self, just_inputs: Sequence[Wire], rest: Sequence[Wire]
    ) -> TailLoop:
        """Start building a new tail loop nested inside the current dataflow
        graph.

        Args:
            just_inputs: input wires for types that are only inputs to the loop body.
            rest: input wires for types that are inputs and outputs of the loop
            body.

        Returns:
            Builder for new nested TailLoop.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> (cond,) = dfg.inputs()
            >>> tl = dfg.add_tail_loop([cond], [cond])
            >>> tl.parent_op
            TailLoop(just_inputs=[Bool], rest=[Bool])
        """
        from .cond_loop import TailLoop

        just_input_types = self._wire_types(just_inputs)
        rest_types = self._wire_types(rest)
        parent_op = ops.TailLoop(just_input_types, rest_types)
        tl = TailLoop.new_nested(parent_op, self.hugr, self.parent_node)
        self._wire_up(tl.parent_node, (*just_inputs, *rest))
        return tl

    def insert_tail_loop(
        self, tl: TailLoop, just_inputs: Sequence[Wire], rest: Sequence[Wire]
    ) -> Node:
        """Insert a tail loop into the current dataflow graph, wiring in the
        inputs.

        Args:
            tl: The tail loop to insert.
            just_inputs: input wires for types that are only inputs to the loop body.
            rest: input wires for types that are inputs and outputs of the loop
            body.

        Returns:
            The root node of the inserted tail loop.

        Example:
            >>> from hugr.cond_loop import TailLoop
            >>> tl = TailLoop([tys.Bool], [tys.Bool])
            >>> dfg = Dfg(tys.Bool)
            >>> (b,) = dfg.inputs()
            >>> tl_n = dfg.insert_tail_loop(tl, [b], [b])
            >>> dfg.hugr[tl_n].op
            TailLoop(just_inputs=[Bool], rest=[Bool])
        """
        return self._insert_nested_impl(tl, *(*just_inputs, *rest))

    def set_outputs(self, *args: Wire) -> None:
        """Set the outputs of the dataflow graph.
        Connects wires to the output node.

        Args:
            args: Wires to connect to the output node.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> dfg.set_outputs(dfg.inputs()[0]) # connect input to output
        """
        self._wire_up(self.output_node, args)
        out_types = self._output_op().types
        self.parent_op._set_out_types(out_types)
        if (
            isinstance(self.parent_op, ops.DataflowOp)
            and self.parent_op._entrypoint_requires_wiring
            and self.hugr.entrypoint == self.parent_node
        ):
            self.hugr._connect_df_entrypoint_outputs()

    def _set_parent_output_count(self, count: int) -> None:
        """Set the final number of output ports on the parent operation.

        Args:
            count: The number of output ports.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> dfg._set_parent_output_count(2)
        """
        self.parent_node = self.hugr._update_node_outs(self.parent_node, count)

    def add_state_order(self, src: Node, dst: Node) -> None:
        """Add a state order link between two nodes.

        Args:
            src: The source node.
            dst: The destination node.

        Examples:
            >>> df = dfg.Dfg()
            >>> df.add_state_order(df.input_node, df.output_node)
            >>> list(df.hugr.outgoing_order_links(df.input_node))
            [Node(2)]
        """
        # adds edge to the right of all existing edges
        self.hugr.add_order_link(src, dst)

    def load(
        self, const: ToNode | val.Value, const_parent: ToNode | None = None
    ) -> Node:
        """Load a constant into the graph as a dataflow value.

        Args:
            const: The constant to load, either a Value that will be added as a
                child Const node then loaded, or a node corresponding to an
                existing Const.
            const_parent: If `const` is a Value, the parent node for the new
                constant definition. Defaults to the current dataflow container.

        Returns:
            The node holding the :class:`LoadConst <hugr.ops.LoadConst>`
            operation.

        Example:
            >>> dfg = Dfg()
            >>> const_n = dfg.load(val.TRUE)
            >>> len(dfg.hugr) # parent, input, output, const, load
            5
            >>> dfg.hugr[const_n].op
            LoadConst(Bool)
        """
        if isinstance(const, val.Value):
            const_parent = const_parent or self.parent_node
            const = self.add_const(const, parent=const_parent)
        const_op = self.hugr._get_typed_op(const, ops.Const)
        load_op = ops.LoadConst(const_op.val.type_())

        load = self.add(load_op())
        self.hugr.add_link(const.out_port(), load.inp(0))

        return load

    def call(
        self,
        func: ToNode,
        *args: Wire,
        instantiation: tys.FunctionType | None = None,
        type_args: Sequence[tys.TypeArg] | None = None,
    ) -> Node:
        """Call a static function in the graph.
        See :class:`Call <hugr.ops.Call>` for more on how polymorphic functions
        are handled.

        Args:
            func: The node corresponding to the function definition/declaration to call.
            args: The input wires to the function call.
            instantiation: The concrete function type to call (needed if polymorphic).
            type_args: The type arguments for the function (needed if
            polymorphic).

        Returns:
            The node holding the :class:`Call <hugr.ops.Call>` operation.
        """
        signature = self._fn_sig(func)
        call_op = ops.Call(signature, instantiation, type_args)
        call_n = self.hugr.add_node(call_op, self.parent_node, call_op.num_out)
        self.hugr.add_link(func.out(0), call_n.inp(call_op._function_port_offset()))

        self._wire_up(call_n, args)

        return call_n

    def load_function(
        self,
        func: ToNode,
        instantiation: tys.FunctionType | None = None,
        type_args: Sequence[tys.TypeArg] | None = None,
    ) -> Node:
        """Load a static function into the graph as a higher-order value.

        Args:
            func: The node corresponding to the function definition/declaration to load.
            instantiation: The concrete function type to load (needed if polymorphic).
            type_args: The type arguments for the function (needed if
            polymorphic).

        Returns:
            The node holding the :class:`LoadFunc <hugr.ops.LoadFunc>` operation.
        """
        signature = self._fn_sig(func)
        load_op = ops.LoadFunc(signature, instantiation, type_args)
        load_n = self.hugr.add_node(load_op, self.parent_node)
        self.hugr.add_link(func.out(0), load_n.inp(0))

        return load_n

    def _fn_sig(self, func: ToNode) -> tys.PolyFuncType:
        f_op = self.hugr[func]
        f_kind = f_op.op.port_kind(func.out(0))
        match f_kind:
            case tys.FunctionKind(sig):
                signature = sig
            case _:
                msg = "Expected 'func' to be a function"
                raise ValueError(msg)
        return signature

    def _wire_up(self, node: Node, ports: Iterable[Wire]) -> tys.TypeRow:
        tys = [self._wire_up_port(node, i, p) for i, p in enumerate(ports)]
        if isinstance(op := self.hugr[node].op, ops._PartialOp):
            op._set_in_types(tys)
            if isinstance(op, ops.DataflowOp):
                # Update the node's input and output port count
                sig = op.outer_signature()
                self.hugr._update_port_count(
                    node, num_inps=len(sig.input), num_outs=len(sig.output)
                )
        return tys

    def _get_dataflow_type(self, wire: Wire) -> tys.Type:
        port = wire.out_port()
        ty = self.hugr.port_type(port)
        if ty is None:
            msg = f"Port {port} is not a dataflow port."
            raise ValueError(msg)
        return ty

    def _wire_up_port(self, node: Node, offset: PortOffset, p: Wire) -> tys.Type:
        src = p.out_port()
        node_ancestor = _ancestral_sibling(self.hugr, src.node, node)
        if node_ancestor is None:
            raise NoSiblingAncestor(src.node.idx, node.idx)
        if node_ancestor != node:
            self.add_state_order(src.node, node_ancestor)
        self.hugr.add_link(src, node.inp(offset))
        return self._get_dataflow_type(src)


class Dfg(DfBase[ops.DFG]):
    """Builder for a simple nested Dataflow graph, with entrypoint node of type
    :class:`DFG <hugr.ops.DFG>`.

    Args:
        input_types: The input types of the the dataflow graph. Output types are
        calculated by propagating types through the graph.

    Example:
        >>> dfg = Dfg(tys.Bool)
        >>> dfg.parent_op
        DFG(inputs=[Bool])
    """

    def __init__(self, *input_types: tys.Type) -> None:
        parent_op = ops.DFG(list(input_types), None)
        super().__init__(parent_op)

    def set_outputs(self, *outputs: Wire) -> None:
        self._set_parent_output_count(len(outputs))
        super().set_outputs(*outputs)


def _ancestral_sibling(h: Hugr, src: Node, tgt: Node) -> Node | None:
    """Find the ancestor of `tgt` that is a sibling of `src`, if one exists."""
    src_parent = h[src].parent

    while (tgt_parent := h[tgt].parent) is not None:
        if tgt_parent == src_parent:
            return tgt
        tgt = tgt_parent

    return None


@dataclass
class Function(DfBase[ops.FuncDefn]):
    """Build a function definition as a HUGR dataflow graph.

    Args:
        name: The name of the function.
        input_types: The input types for the function (output types are
        computed by propagating types from input node through the graph).
        type_params: The type parameters for the function, if polymorphic.

    Examples:
        >>> f = Function("f", [tys.Bool])
        >>> f.parent_op
        FuncDefn(f_name='f', inputs=[Bool], params=[])
    """

    def __init__(
        self,
        name: str,
        input_types: TypeRow,
        type_params: list[TypeParam] | None = None,
    ) -> None:
        root_op = ops.FuncDefn(name, input_types, type_params or [])
        super().__init__(root_op)

    def declare_outputs(self, output_types: TypeRow) -> None:
        """Declare the output types of the function.

        This is required when calling a function which hasn't been completely
        defined yet. The wires passed to :meth:`set_outputs` must match the
        declared output types.
        """
        self._set_parent_output_count(len(output_types))
        self.parent_op._set_out_types(output_types)

    def set_outputs(self, *args: Wire) -> None:
        """Set the outputs of the dataflow graph.
        Connects wires to the output node.

        If :meth:`declare_outputs` has been called, the wire types must match
        the declared output types.

        Args:
            args: Wires to connect to the output node.

        Example:
            >>> dfg = Dfg(tys.Bool)
            >>> dfg.set_outputs(dfg.inputs()[0]) # connect input to output
        """
        if self.parent_op._outputs is not None:
            arg_types = [self._get_dataflow_type(w) for w in args]
            if arg_types != self.parent_op._outputs:
                error_message = (
                    f"The function has fixed output type {self.parent_op._outputs}, "
                    f"but was given output wires with types {arg_types}."
                )
                raise ValueError(error_message)

        super().set_outputs(*args)
