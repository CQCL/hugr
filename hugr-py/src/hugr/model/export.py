"""Helpers to export hugr graphs from their python representation to hugr model."""

import json
from collections.abc import Sequence
from typing import Generic, TypeVar, cast

import hugr.model as model
from hugr.hugr.base import Hugr, Node
from hugr.hugr.node_port import InPort, OutPort
from hugr.ops import (
    CFG,
    DFG,
    AliasDecl,
    AliasDefn,
    AsExtOp,
    Call,
    CallIndirect,
    Conditional,
    Const,
    Custom,
    DataflowBlock,
    ExitBlock,
    FuncDecl,
    FuncDefn,
    Input,
    LoadConst,
    LoadFunc,
    Output,
    Tag,
    TailLoop,
)
from hugr.tys import ConstKind, FunctionKind, Type, TypeBound, TypeParam, TypeTypeParam


class ModelExport:
    """Helper to export a Hugr."""

    def __init__(self, hugr: Hugr):
        self.hugr = hugr
        self.link_ports: _UnionFind[InPort | OutPort] = _UnionFind()
        self.link_names: dict[InPort | OutPort, str] = {}

        # TODO: Store the hugr entrypoint

        for a, b in self.hugr.links():
            self.link_ports.union(a, b)

    def link_name(self, port: InPort | OutPort) -> str:
        """Return the name of the link that a given port is connected to."""
        root = self.link_ports[port]

        if root in self.link_names:
            return self.link_names[root]
        else:
            index = str(len(self.link_names))
            self.link_names[root] = index
            return index

    def export_node(self, node: Node) -> model.Node | None:
        """Export the node with the given node id."""
        node_data = self.hugr[node]

        inputs = [self.link_name(InPort(node, i)) for i in range(node_data._num_inps)]
        outputs = [self.link_name(OutPort(node, i)) for i in range(node_data._num_outs)]
        meta = []

        # Export JSON metadata
        for meta_name, meta_value in node_data.metadata.items():
            # TODO: Is this the correct way to convert the metadata as JSON?
            meta_json = json.dumps(meta_value)
            meta.append(
                model.Apply(
                    "compat.meta_json",
                    [model.Literal(meta_name), model.Literal(meta_json)],
                )
            )

        # Add an order hint key to the node if necessary
        if _needs_order_key(self.hugr, node):
            meta.append(model.Apply("core.order_hint.key", [model.Literal(node.idx)]))

        match node_data.op:
            case DFG() as op:
                region = self.export_region_dfg(node)

                return model.Node(
                    operation=model.Dfg(),
                    regions=[region],
                    signature=op.outer_signature().to_model(),
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case Custom() as op:
                name = f"{op.extension}.{op.op_name}"
                args = cast(list[model.Term], [arg.to_model() for arg in op.args])
                signature = op.signature.to_model()

                return model.Node(
                    operation=model.CustomOp(model.Apply(name, args)),
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case AsExtOp() as op:
                name = op.op_def().qualified_name()
                args = cast(
                    list[model.Term], [arg.to_model() for arg in op.type_args()]
                )
                signature = op.outer_signature().to_model()

                return model.Node(
                    operation=model.CustomOp(model.Apply(name, args)),
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case Conditional() as op:
                regions = [
                    self.export_region_dfg(child) for child in node_data.children
                ]

                signature = op.outer_signature().to_model()

                return model.Node(
                    operation=model.Conditional(),
                    regions=regions,
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case TailLoop() as op:
                region = self.export_region_dfg(node)
                signature = op.outer_signature().to_model()
                return model.Node(
                    operation=model.TailLoop(),
                    regions=[region],
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case FuncDefn() as op:
                name = _mangle_name(node, op.f_name)
                symbol = self.export_symbol(
                    name, op.signature.params, op.signature.body
                )
                region = self.export_region_dfg(node)

                return model.Node(
                    operation=model.DefineFunc(symbol), regions=[region], meta=meta
                )

            case FuncDecl() as op:
                name = _mangle_name(node, op.f_name)
                symbol = self.export_symbol(
                    name, op.signature.params, op.signature.body
                )
                return model.Node(operation=model.DeclareFunc(symbol), meta=meta)

            case AliasDecl() as op:
                symbol = model.Symbol(name=op.alias, signature=model.Apply("core.type"))

                return model.Node(operation=model.DeclareAlias(symbol), meta=meta)

            case AliasDefn() as op:
                symbol = model.Symbol(name=op.alias, signature=model.Apply("core.type"))

                alias_value = cast(model.Term, op.definition.to_model())

                return model.Node(
                    operation=model.DefineAlias(symbol, alias_value), meta=meta
                )

            case Call() as op:
                input_types = [type.to_model() for type in op.instantiation.input]
                output_types = [type.to_model() for type in op.instantiation.output]
                signature = op.instantiation.to_model()
                func_args = cast(
                    list[model.Term], [type.to_model() for type in op.type_args]
                )
                func_name = self.find_func_input(node)

                if func_name is None:
                    error = f"Call node {node} is not connected to a function."
                    raise ValueError(error)

                func = model.Apply(func_name, func_args)

                return model.Node(
                    operation=model.CustomOp(
                        model.Apply(
                            "core.call",
                            [
                                model.List(input_types),
                                model.List(output_types),
                                func,
                            ],
                        )
                    ),
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case LoadFunc() as op:
                signature = op.instantiation.to_model()
                func_args = cast(
                    list[model.Term], [type.to_model() for type in op.type_args]
                )
                func_name = self.find_func_input(node)

                if func_name is None:
                    error = f"LoadFunc node {node} is not connected to a function."
                    raise ValueError(error)

                func = model.Apply(func_name, func_args)

                return model.Node(
                    operation=model.CustomOp(
                        model.Apply("core.load_const", [signature, func])
                    ),
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case CallIndirect() as op:
                input_types = [type.to_model() for type in op.signature.input]
                output_types = [type.to_model() for type in op.signature.output]

                func = model.Apply(
                    "core.fn",
                    [model.List(input_types), model.List(output_types)],
                )

                signature = model.Apply(
                    "core.fn",
                    [
                        model.List([func, *input_types]),
                        model.List(output_types),
                    ],
                )

                return model.Node(
                    operation=model.CustomOp(
                        model.Apply(
                            "core.call_indirect",
                            [
                                model.List(input_types),
                                model.List(output_types),
                            ],
                        )
                    ),
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case LoadConst() as op:
                value = self.find_const_input(node)

                if value is None:
                    error = f"LoadConst node {node} is not connected to a constant."
                    raise ValueError(error)

                type = cast(model.Term, op.type_.to_model())
                signature = op.outer_signature().to_model()

                return model.Node(
                    operation=model.CustomOp(
                        model.Apply("core.load_const", [type, value])
                    ),
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    meta=meta,
                )

            case Const() as op:
                return None

            case CFG() as op:
                signature = op.outer_signature().to_model()
                region = self.export_region_cfg(node)

                return model.Node(
                    operation=model.Cfg(),
                    signature=signature,
                    inputs=inputs,
                    outputs=outputs,
                    regions=[region],
                    meta=meta,
                )

            case DataflowBlock() as op:
                region = self.export_region_dfg(node)

                input_types = [
                    model.Apply(
                        "core.ctrl",
                        [model.List([type.to_model() for type in op.inputs])],
                    )
                ]

                other_output_types = [type.to_model() for type in op.other_outputs]
                output_types = [
                    model.Apply(
                        "core.ctrl",
                        [
                            model.List(
                                [
                                    *[type.to_model() for type in row],
                                    *other_output_types,
                                ]
                            )
                        ],
                    )
                    for row in op.sum_ty.variant_rows
                ]

                signature = model.Apply(
                    "core.fn",
                    [model.List(input_types), model.List(output_types)],
                )

                return model.Node(
                    operation=model.Block(),
                    inputs=inputs,
                    outputs=outputs,
                    regions=[region],
                    signature=signature,
                    meta=meta,
                )

            case Tag() as op:
                variants = model.List(
                    [
                        model.List([type.to_model() for type in row])
                        for row in op.sum_ty.variant_rows
                    ]
                )

                types = model.List(
                    [type.to_model() for type in op.sum_ty.variant_rows[op.tag]]
                )

                tag = model.Literal(op.tag)
                signature = op.outer_signature().to_model()

                return model.Node(
                    operation=model.CustomOp(
                        model.Apply("core.make_adt", [variants, types, tag])
                    ),
                    inputs=inputs,
                    outputs=outputs,
                    signature=signature,
                    meta=meta,
                )

            case op:
                error = f"Unknown operation: {op}"
                raise ValueError(error)

    def export_region_module(self, node: Node) -> model.Region:
        """Export a module node as a module region."""
        node_data = self.hugr[node]
        children = []

        for child in node_data.children:
            child_node = self.export_node(child)

            if child_node is not None:
                children.append(child_node)

        return model.Region(kind=model.RegionKind.MODULE, children=children)

    def export_region_dfg(self, node: Node) -> model.Region:
        """Export the children of a node as a dataflow region."""
        node_data = self.hugr[node]
        children: list[model.Node] = []
        source_types: model.Term = model.Wildcard()
        target_types: model.Term = model.Wildcard()
        sources = []
        targets = []
        meta = []

        for child in node_data.children:
            child_data = self.hugr[child]

            match child_data.op:
                case Input() as op:
                    source_types = model.List([type.to_model() for type in op.types])
                    sources = [
                        self.link_name(OutPort(child, i))
                        for i in range(child_data._num_outs)
                    ]

                case Output() as op:
                    target_types = model.List([type.to_model() for type in op.types])
                    targets = [
                        self.link_name(InPort(child, i))
                        for i in range(child_data._num_inps)
                    ]

                case _:
                    child_node = self.export_node(child)

                    if child_node is None:
                        continue

                    children.append(child_node)

                    meta += [
                        model.Apply(
                            "core.order_hint.order",
                            [model.Literal(child.idx), model.Literal(successor.idx)],
                        )
                        for successor in self.hugr.outgoing_order_links(child)
                        if not isinstance(self.hugr[successor].op, Output)
                    ]

        signature = model.Apply("core.fn", [source_types, target_types])

        return model.Region(
            kind=model.RegionKind.DATA_FLOW,
            signature=signature,
            children=children,
            sources=sources,
            targets=targets,
        )

    def export_region_cfg(self, node: Node) -> model.Region:
        """Export the children of a node as a control flow region."""
        node_data = self.hugr[node]

        source = None
        targets = []
        source_types: model.Term = model.Wildcard()
        target_types: model.Term = model.Wildcard()
        children = []

        for child in node_data.children:
            child_data = self.hugr[child]

            match child_data.op:
                case ExitBlock() as op:
                    target_types = model.List(
                        [type.to_model() for type in op.cfg_outputs]
                    )
                    targets = [
                        self.link_name(InPort(child, i))
                        for i in range(child_data._num_inps)
                    ]
                case DataflowBlock() as op:
                    if source is None:
                        source_types = model.List(
                            [type.to_model() for type in op.inputs]
                        )
                        source = self.link_name(OutPort(child, 0))

                    child_node = self.export_node(child)

                    if child_node is not None:
                        children.append(child_node)
                case _:
                    error = f"Unexpected operation in CFG {node}"
                    raise ValueError(error)

        if source is None:
            error = f"CFG {node} has no entry block."
            raise ValueError(error)

        signature = model.Apply("core.fn", [source_types, target_types])

        return model.Region(
            kind=model.RegionKind.CONTROL_FLOW,
            targets=targets,
            sources=[source],
            signature=signature,
            children=children,
        )

    def export_symbol(
        self, name: str, param_types: Sequence[TypeParam], body: Type
    ) -> model.Symbol:
        """Export a symbol."""
        constraints = []
        params = []

        for i, param_type in enumerate(param_types):
            param_name = str(i)

            params.append(model.Param(name=param_name, type=param_type.to_model()))

            match param_type:
                case TypeTypeParam(bound=TypeBound.Copyable):
                    constraints.append(
                        model.Apply("core.nonlinear", [model.Var(param_name)])
                    )
                case _:
                    pass

        return model.Symbol(
            name=name,
            params=params,
            constraints=constraints,
            signature=cast(model.Term, body.to_model()),
        )

    def find_func_input(self, node: Node) -> str | None:
        """Find the name of the function that a node is connected to, if any."""
        try:
            func_node = next(
                out_port.node
                for (in_port, out_ports) in self.hugr.incoming_links(node)
                if isinstance(self.hugr.port_kind(in_port), FunctionKind)
                for out_port in out_ports
            )
        except StopIteration:
            return None

        match self.hugr[func_node].op:
            case FuncDecl() as func_op:
                name = func_op.f_name
            case FuncDefn() as func_op:
                name = func_op.f_name
            case _:
                return None

        return _mangle_name(node, name)

    def find_const_input(self, node: Node) -> model.Term | None:
        """Find and export the constant that a node is connected to, if any."""
        try:
            const_node = next(
                out_port.node
                for (in_port, out_ports) in self.hugr.incoming_links(node)
                if isinstance(self.hugr.port_kind(in_port), ConstKind)
                for out_port in out_ports
            )
        except StopIteration:
            return None

        match self.hugr[const_node].op:
            case Const() as op:
                return op.val.to_model()
            case op:
                return None


def _mangle_name(node: Node, name: str) -> str:
    # Until we come to an agreement on the uniqueness of names, we mangle the names
    # by adding the node id.
    return f"_{name}_{node.idx}"


T = TypeVar("T")


class _UnionFind(Generic[T]):
    def __init__(self) -> None:
        self.parents: dict[T, T] = {}
        self.sizes: dict[T, int] = {}

    def __getitem__(self, item: T) -> T:
        if item not in self.parents:
            self.parents[item] = item
            self.sizes[item] = 1
            return item

        # Path splitting
        while self.parents[item] != item:
            parent = self.parents[item]
            self.parents[item] = self.parents[parent]
            item = parent

        return item

    def union(self, a: T, b: T):
        a = self[a]
        b = self[b]

        if a == b:
            return

        if self.sizes[a] < self.sizes[b]:
            (a, b) = (b, a)

        self.parents[b] = a
        self.sizes[a] += self.sizes[b]


def _needs_order_key(hugr: Hugr, node: Node) -> bool:
    """Checks whether the node has any order links for the purposes of
    exporting order hint metadata. Order links to `Input` or `Output`
    operations are ignored, since they are not present in the model format.
    """
    for succ in hugr.outgoing_order_links(node):
        succ_op = hugr[succ].op
        if not isinstance(succ_op, Output):
            return True

    for pred in hugr.incoming_order_links(node):
        pred_op = hugr[pred].op
        if not isinstance(pred_op, Input):
            return True

    return False
