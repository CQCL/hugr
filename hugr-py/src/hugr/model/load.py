"""Helpers to import hugr graphs from hugr model to their python representation."""

import json
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import hugr.model as model
from hugr import val
from hugr.hugr import InPort, OutPort
from hugr.hugr.base import Hugr
from hugr.hugr.node_port import Node
from hugr.ops import (
    CFG,
    DFG,
    AliasDecl,
    AliasDefn,
    Call,
    CallIndirect,
    Case,
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
    MakeTuple,
    Op,
    Output,
    Tag,
    TailLoop,
    UnpackTuple,
)
from hugr.std.collections.array import ArrayVal
from hugr.std.float import FloatVal
from hugr.std.int import IntVal
from hugr.tys import (
    BoundedNatArg,
    BoundedNatParam,
    BytesArg,
    BytesParam,
    ConstParam,
    FloatArg,
    FloatParam,
    FunctionType,
    ListArg,
    ListConcatArg,
    ListParam,
    Opaque,
    PolyFuncType,
    RowVariable,
    StringArg,
    StringParam,
    Sum,
    Tuple,
    TupleArg,
    TupleConcatArg,
    TupleParam,
    Type,
    TypeArg,
    TypeBound,
    TypeParam,
    TypeTypeArg,
    TypeTypeParam,
    Variable,
    _QubitDef,
)

ImportContext = model.Term | model.Node | model.Region | str


class ModelImportError(Exception):
    """Exception raised when importing from the model representation fails."""

    def __init__(self, message: str, location: ImportContext | None = None):
        self.message = message
        self.location = location

        match location:
            case model.Term() as term:
                location_error = f"Error caused by term:\n```\n{term}\n```"
            case model.Region() as region:
                location_error = f"Error caused by region:\n```\n{region}\n```"
            case model.Node() as node:
                location_error = f"Error caused by node:\n```\n{node}\n```"
            case str() as other:
                location_error = other
            case None:
                location_error = "Error in unspecified location."

        super().__init__(f"{message}\n{location_error}")


def _collect_meta_json(node: model.Node) -> dict[str, Any]:
    """Collects the `core.meta_json` metadata on the given node."""
    metadata = {}

    for meta in node.meta:
        match meta:
            case model.Apply(
                "compat.meta_json",
                [model.Literal(str() as key), model.Literal(str() as value)],
            ):
                pass
            case _:
                continue

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as err:
            error = "Failed to decode JSON metadata."
            raise ModelImportError(error, node) from err

        metadata[key] = decoded

    return metadata


def _find_meta_title(node: model.Node) -> str | None:
    """Searches for `core.title` metadata on the given node."""
    for meta in node.meta:
        match meta:
            case model.Apply("core.title", [model.Literal(str() as title)]):
                return title
            case model.Apply("core.title"):
                error = "Invalid instance of `core.title` metadata."
                raise ModelImportError(error, meta)
            case _:
                pass

    return None


def _find_meta_order_region(region: model.Region) -> "RegionOrderHints":
    """Searches for order hint metadata on the given region."""
    data = RegionOrderHints()

    for meta in region.meta:
        match meta:
            case model.Apply(
                "core.order_hint.input_key", [model.Literal(int() as key)]
            ):
                data.input_keys.append(key)
            case model.Apply(
                "core.order_hint.output_key", [model.Literal(int() as key)]
            ):
                data.output_keys.append(key)
            case model.Apply(
                "core.order_hint.order",
                [model.Literal(int() as before), model.Literal(int() as after)],
            ):
                data.edges.append((before, after))
            case _:
                pass

    return data


def _collect_meta_order_keys(node: model.Node) -> list[int]:
    """Collects all order hint keys in the metadata of a node."""
    keys = []

    for meta in node.meta:
        match meta:
            case model.Apply("core.order_hint.key", [model.Literal(int() as key)]):
                keys.append(key)
            case _:
                pass

    return keys


class ModelImport:
    """Helper to import a Hugr."""

    local_vars: dict[str, "LocalVarData"]
    current_symbol: str | None
    link_prefix: int | None
    linked_ports: dict[str, tuple[list[InPort], list[OutPort]]]
    static_edges: list[tuple[Node, Node]]

    module: model.Module
    symbols: dict[str, model.Node]
    fn_nodes: dict[str, Node]
    fn_calls: list[tuple[str, Node]]
    hugr: Hugr

    def __init__(self, module: model.Module):
        self.local_vars = {}
        self.current_symbol = None
        self.module = module
        self.symbols = {}
        self.hugr = Hugr()
        self.link_prefix = None
        self.linked_ports = {}
        self.static_edges = []
        self.fn_nodes = {}
        self.fn_calls = []

        for node in module.root.children:
            symbol_name = node.operation.symbol_name()

            if symbol_name is None:
                continue

            if symbol_name in self.symbols:
                error = f"Duplicate symbol name `{symbol_name}`."
                raise ModelImportError(error, node)

            self.symbols[symbol_name] = node

    def add_node(
        self, node: model.Node, operation: Op, parent: Node, num_outs: int | None = None
    ) -> Node:
        """Add a model Node to the Hugr and record its in- and out-links."""
        node_id = self.hugr.add_node(
            op=operation,
            parent=parent,
            num_outs=num_outs,
            metadata=_collect_meta_json(node),
        )
        self.record_in_links(node_id, node.inputs)
        self.record_out_links(node_id, node.outputs)
        if model.Apply("core.entrypoint") in node.meta:
            self.hugr.entrypoint = node_id
        return node_id

    def record_in_links(self, node: Node, links: Iterable[str]):
        """Record a bunch of links entering the given Hugr Node with the given names."""
        for offset, link in enumerate(links):
            in_port = InPort(node=node, offset=offset)
            self.linked_ports.setdefault(f"{self.link_prefix}_{link}", ([], []))[
                0
            ].append(in_port)

    def record_out_links(self, node: Node, links: Iterable[str]):
        """Record a bunch of links exiting the given Hugr Node with the given names."""
        for offset, link in enumerate(links):
            out_port = OutPort(node=node, offset=offset)
            self.linked_ports.setdefault(f"{self.link_prefix}_{link}", ([], []))[
                1
            ].append(out_port)

    def link_ports(self):
        """Add links to the Hugr according to the recorded data."""
        for link, (in_ports, out_ports) in self.linked_ports.items():
            match in_ports, out_ports:
                case [[], []]:
                    raise AssertionError
                case _, [out_port]:
                    for in_port in in_ports:
                        self.hugr.add_link(out_port, in_port)
                case [[in_port], _]:
                    for out_port in out_ports:
                        self.hugr.add_link(out_port, in_port)
                case _, _:
                    error = f"Link `{link}` has multiple inputs and outputs."
                    raise ModelImportError(error)

    def link_static_ports(self):
        for symbol, callnode in self.fn_calls:
            self.static_edges.append((self.fn_nodes[symbol], callnode))
        for src, dst in self.static_edges:
            out_port_offset = self.hugr.num_out_ports(src) - 1
            out_port = OutPort(node=src, offset=out_port_offset)

            in_port_offset = self.hugr.num_in_ports(dst)
            in_port = InPort(node=dst, offset=in_port_offset)

            self.hugr.add_link(out_port, in_port)

    def add_module_metadata(self):
        self.hugr[self.hugr.module_root].metadata = _collect_meta_json(self.module.root)

    def import_dfg_region(self, region: model.Region, parent: Node):
        """Import an entire DFG region from the model into the Hugr."""
        signature = self.import_signature(region.signature)

        input_node = self.hugr.add_node(
            Input(signature.input), parent=parent, num_outs=len(signature.input)
        )
        self.record_out_links(input_node, region.sources)

        output_node = self.hugr.add_node(Output(signature.output), parent=parent)
        self.record_in_links(output_node, region.targets)

        order_data = _find_meta_order_region(region)
        order_data.add_node_keys(input_node, order_data.input_keys)
        order_data.add_node_keys(output_node, order_data.output_keys)

        for child in region.children:
            child_id = self.import_node_in_dfg(child, parent)
            child_order_keys = _collect_meta_order_keys(child)
            order_data.add_node_keys(child_id, child_order_keys)

        for src_key, tgt_key in order_data.edges:
            src_node = order_data.get_node_by_key(src_key)
            tgt_node = order_data.get_node_by_key(tgt_key)
            self.hugr.add_order_link(src_node, tgt_node)

    def import_block(self, block: model.Node, parent: Node):
        # 1. Add the DataFlowBlock node:
        match block.signature:
            case model.Apply("core.ctrl", [ctrl_inputs, ctrl_outputs]):
                pass
            case _:
                error = f"Invalid signature for {block}."
                raise ModelImportError(error)
        match list(ctrl_inputs.to_list_parts()):
            case [inputs]:
                pass
            case _:
                error = f"DFB inputs should be singleton list: {ctrl_inputs}."
                raise ModelImportError(error)
        assert isinstance(inputs, model.Term)
        block_node = self.add_node(
            block,
            # TODO The translation here seems to be underdetermined. It could be
            # DataflowBlock(
            #     self.import_type_row(inputs),
            #     Sum(ts),
            #     ss,
            # ),
            # where the ctrl_outputs have been expressed as:
            # [[*ts[0], *ss], [*ts[1], *ss], ...]
            # with ss some common suffix of the lists in ctrl_outputs. But how do we
            # decide on that common suffix? Below we take it to be empty.
            DataflowBlock(
                self.import_type_row(inputs),
                Sum(
                    [
                        self.import_type_row(cast(model.Term, output))
                        for output in ctrl_outputs.to_list_parts()
                    ]
                ),
                [],
            ),
            parent,
        )
        # 2. Import the dataflow region:
        [block_region] = block.regions
        self.import_dfg_region(block_region, block_node)

    def import_cfg_region(
        self, region: model.Region, signature: FunctionType, parent: Node
    ):
        """Import an entire CFG region from the model into the Hugr."""
        [entry_link] = region.sources
        entry_block_idx = None
        for i, child in enumerate(region.children):
            if entry_link in child.inputs:
                entry_block_idx = i
                break
        assert entry_block_idx is not None
        entry_block = region.children[entry_block_idx]

        # 1. Import the entry block:
        self.import_block(entry_block, parent)

        # 2. Create the exit node:
        exit_node = self.hugr.add_node(ExitBlock(signature.output), parent)
        self.record_in_links(exit_node, region.targets)

        # 3. Import the other blocks:
        for i, child in enumerate(region.children):
            if i != entry_block_idx:
                self.import_block(child, parent)

    def import_node_in_dfg(self, node: model.Node, parent: Node) -> Node:
        """Import a model Node within a DFG region.

        Returns the Hugr Node corresponding to the model Node. The correspondence is
        almost 1-1, but a LoadConst model Node requires two Hugr Nodes (Const and
        LoadConst); in this case the LoadConst is returned.
        """
        signature = self.import_signature(node.signature)

        def import_dfg_node() -> Node:
            match node.regions:
                case [body]:
                    pass
                case _:
                    error = "DFG node expects a dataflow region."
                    raise ModelImportError(error, node)
            node_id = self.add_node(
                node, DFG(signature.input, signature.output), parent
            )
            self.import_dfg_region(body, node_id)
            return node_id

        def import_tail_loop() -> Node:
            match node.regions:
                case [body]:
                    pass
                case _:
                    error = "Loop node expects a dataflow region."
                    raise ModelImportError(error, node)

            match body.signature:
                case model.Apply("core.fn", [_, body_outputs]):
                    pass
                case _:
                    error = "Tail loop body expects `(core.fn _ _)` signature."
                    raise ModelImportError(error, node)

            match list(_import_closed_list(body_outputs)):
                case [model.Apply("core.adt", [variants]), *rest]:
                    pass
                case _:
                    error = "TailLoop body expects `(core.adt _)` as first target type."
                    raise ModelImportError(error, node)

            match list(_import_closed_list(variants)):
                case [just_inputs, just_outputs]:
                    pass
                case _:
                    error = "TailLoop body expects sum type with two variants."
                    raise ModelImportError(error, node)

            node_id = self.add_node(
                node,
                TailLoop(
                    just_inputs=self.import_type_row(just_inputs),
                    rest=[self.import_type(t) for t in rest],
                    _just_outputs=self.import_type_row(just_outputs),
                ),
                parent,
                len(signature.output),
            )
            self.import_dfg_region(body, node_id)
            return node_id

        def import_custom_node(op: model.Term) -> Node:
            match op:
                case model.Apply(symbol, args):
                    pass
                case _:
                    error = "The operation of a custom node must be a symbol "
                    "application."
                    raise ModelImportError(error, node)

            match symbol:
                case "core.call":
                    _input_types, _output_types, func = args
                    match func:
                        case model.Apply(fn_symbol, fn_args):
                            pass
                        case _:
                            error = "The function of a Call node must be a symbol "
                            "application."
                            raise ModelImportError(error, node)
                    type_args = [self.import_type_arg(fn_arg) for fn_arg in fn_args]
                    callnode = self.add_node(
                        node,
                        Call(
                            # FIXME PolyFuncType needs list[TypeParam], not
                            # list[TypeArg]. How to get this?
                            signature=PolyFuncType(type_args, signature),  # type: ignore[arg-type]
                            instantiation=signature,
                            type_args=type_args,
                        ),
                        parent,
                        len(signature.output),
                    )
                    self.fn_calls.append((fn_symbol, callnode))
                    return callnode
                case "core.call_indirect":
                    [inputs, outputs] = args
                    sig = FunctionType(
                        self.import_type_row(inputs), self.import_type_row(outputs)
                    )
                    callindirectnode = self.add_node(
                        node, CallIndirect(sig), parent, len(signature.output)
                    )
                    return callindirectnode
                case "core.load_const":
                    value = args[-1]
                    [datatype] = signature.output
                    match datatype:
                        case FunctionType(_inputs, _outputs):
                            # Import as a LoadFunc operation.
                            match value:
                                case model.Apply(str() as fn_id, fn_args):
                                    pass
                                case _:
                                    error = "Unexpected arguments to core.load_const: "
                                    f"{args}"
                                    raise ModelImportError(error, node)
                            type_args = [
                                self.import_type_arg(fn_arg) for fn_arg in fn_args
                            ]
                            loadfunc_node = self.add_node(
                                node,
                                LoadFunc(
                                    # FIXME PolyFuncType needs list[TypeParam], not
                                    # list[TypeArg]. How to get this?
                                    PolyFuncType(type_args, datatype),  # type: ignore[arg-type]
                                    datatype,
                                    type_args,
                                ),
                                parent,
                                1,
                            )
                            self.fn_calls.append((fn_id, loadfunc_node))
                            return loadfunc_node
                        case _:
                            # Import as a Const and a LoadConst node.
                            v = self.import_value(value)
                            const_node = self.hugr.add_node(Const(v), parent, 1)
                            loadconst_node = self.add_node(
                                node, LoadConst(datatype), parent, 1
                            )
                            self.hugr.add_link(
                                OutPort(const_node, 0), InPort(loadconst_node, 0)
                            )
                            return loadconst_node
                case "core.make_adt":
                    tag = args[-1]
                    match tag:
                        case model.Literal(int() as tagval):
                            pass
                        case _:
                            error = f"Unexpected tag: {tag}"
                            raise ModelImportError(error)
                    [sigout] = signature.output
                    match sigout:
                        case Sum(_variant_rows) as output_sum:
                            pass
                        case _:
                            error = f"Invalid signature with {symbol}: {node.signature}"
                            raise ModelImportError(error)
                    return self.add_node(node, Tag(tagval, output_sum), parent, 1)
                case "prelude.MakeTuple":
                    [arglist] = args
                    return self.add_node(
                        node,
                        MakeTuple(self.import_type_row(arglist)),
                        parent,
                        1,
                    )
                case "prelude.UnpackTuple":
                    [arglist] = args
                    typerow = self.import_type_row(arglist)
                    return self.add_node(
                        node,
                        UnpackTuple(typerow),
                        parent,
                        len(typerow),
                    )
                # Others are imported as Custom nodes.
                case _:
                    extension, op_name = _split_extension_name(symbol)
                    return self.add_node(
                        node,
                        Custom(
                            op_name=op_name,
                            extension=extension,
                            signature=signature,
                            args=[self.import_type_arg(arg) for arg in args],
                        ),
                        parent,
                        len(signature.output),
                    )

        def import_cfg() -> Node:
            match node.regions:
                case [body]:
                    pass
                case _:
                    error = "CFG node expects a control-flow region."
                    raise ModelImportError(error, node)
            node_id = self.add_node(
                node, CFG(signature.input, signature.output), parent
            )
            self.import_cfg_region(body, signature, node_id)
            return node_id

        def import_conditional() -> Node:
            match node.signature:
                case model.Apply("core.fn", [inputs, outputs]):
                    pass
                case _:
                    error = "Conditional node expects `(core.fn _ _)` signature."
                    raise ModelImportError(error, node)

            match list(_import_closed_list(inputs)):
                case [model.Apply("core.adt", [variants]), *other_inputs]:
                    sum_ty = Sum(
                        [
                            self.import_type_row(variant)
                            for variant in _import_closed_list(variants)
                        ]
                    )
                case _:
                    error = (
                        "Conditional node expects `(core.adt _)` as first input type."
                    )
                    raise ModelImportError(
                        error,
                        node,
                    )

            node_id = self.add_node(
                node,
                Conditional(
                    sum_ty=sum_ty,
                    other_inputs=[self.import_type(t) for t in other_inputs],
                    _outputs=self.import_type_row(outputs),
                ),
                parent,
            )

            for case_body in node.regions:
                case_signature = self.import_signature(case_body.signature)
                case_id = self.hugr.add_node(
                    Case(inputs=case_signature.input, _outputs=case_signature.output),
                    node_id,
                )
                self.import_dfg_region(case_body, case_id)

            return node_id

        match node.operation:
            case model.InvalidOp():
                error = "Invalid operation can not be imported."
                raise ModelImportError(error, node)
            case model.Dfg():
                return import_dfg_node()
            case model.Cfg():
                return import_cfg()
            case model.Block():
                error = "Unexpected basic block."
                raise ModelImportError(error, node)
            case model.CustomOp(op):
                return import_custom_node(op)
            case model.TailLoop():
                return import_tail_loop()
            case model.Conditional():
                return import_conditional()
            case _:
                error = "Unexpected node in DFG region."
                raise ModelImportError(error, node)

    def import_node_in_module(self, node: model.Node, link_prefix: int) -> Node | None:
        """Import a model Node at the Hugr Module level."""
        self.link_prefix = link_prefix

        def import_declare_func(symbol: model.Symbol) -> Node:
            f_name = _find_meta_title(node)
            if f_name is None:
                f_name = symbol.name
            signature = self.enter_symbol(symbol)
            node_id = self.add_node(
                node,
                FuncDecl(
                    f_name=f_name, signature=signature, visibility=symbol.visibility
                ),
                self.hugr.module_root,
                1,
            )
            self.exit_symbol()
            self.fn_nodes[symbol.name] = node_id
            return node_id

        def import_define_func(symbol: model.Symbol) -> Node:
            f_name = _find_meta_title(node)
            if f_name is None:
                f_name = symbol.name
            signature = self.enter_symbol(symbol)
            node_id = self.add_node(
                node,
                FuncDefn(
                    f_name=f_name,
                    inputs=signature.body.input,
                    _outputs=signature.body.output,
                    params=signature.params,
                    visibility=symbol.visibility,
                ),
                self.hugr.module_root,
                1,
            )

            match node.regions:
                case [body]:
                    pass
                case _:
                    error = "Function definition expects a single region."
                    raise ModelImportError(error, node)

            self.import_dfg_region(body, node_id)
            self.exit_symbol()
            self.fn_nodes[symbol.name] = node_id
            return node_id

        def import_declare_alias(symbol: model.Symbol) -> Node:
            match symbol:
                case model.Symbol(
                    name=name,
                    visibility=_visibility,
                    signature=model.Apply("core.type", []),
                ):
                    pass
                case _:
                    error = f"Unexpected symbol in alias declaration: {symbol}"
                    raise ModelImportError(error)
            return self.add_node(
                node,
                AliasDecl(alias=name, bound=TypeBound.Copyable),  # TODO which bound?
                self.hugr.module_root,
            )

        def import_define_alias(symbol: model.Symbol, value: model.Term) -> Node:
            match symbol:
                case model.Symbol(
                    name=name,
                    visibility=_visibility,
                    signature=model.Apply("core.type", []),
                ):
                    pass
                case _:
                    error = f"Unexpected symbol in alias definition: {symbol}"
                    raise ModelImportError(error)
            return self.add_node(
                node,
                AliasDefn(alias=name, definition=self.import_type(value)),
                self.hugr.module_root,
            )

        imported_node = None
        match node.operation:
            case model.DeclareFunc(symbol):
                imported_node = import_declare_func(symbol)
            case model.DefineFunc(symbol):
                imported_node = import_define_func(symbol)
            case model.DeclareAlias(symbol):
                imported_node = import_declare_alias(symbol)
            case model.DefineAlias(symbol, value):
                imported_node = import_define_alias(symbol, value)
            case model.Import():
                pass
            case model.DeclareConstructor():
                pass
            case model.DeclareOperation():
                pass
            case _:
                error = "Unexpected node in module region."
                raise ModelImportError(error, node)
        self.link_prefix = None
        return imported_node

    def enter_symbol(self, symbol: model.Symbol) -> PolyFuncType:
        assert len(self.local_vars) == 0

        bounds: dict[str, TypeBound] = {}

        for constraint in symbol.constraints:
            match constraint:
                case model.Apply("core.nonlinear", [model.Var(name)]):
                    bounds[name] = TypeBound.Copyable
                case _:
                    error = "Constraint other than `core.nonlinear` on a variable."
                    raise ModelImportError(error, constraint)

        param_types: list[TypeParam] = []

        for index, param in enumerate(symbol.params):
            bound = bounds.get(param.name, TypeBound.Linear)
            type = self.import_type_param(param.type, bound=bound)
            self.local_vars[param.name] = LocalVarData(index, type)
            param_types.append(type)

        body = self.import_signature(symbol.signature)
        return PolyFuncType(param_types, body)

    def exit_symbol(self):
        self.local_vars = {}

    def import_signature(self, term: model.Term | None) -> FunctionType:
        match term:
            case None:
                error = "Signature required."
                raise ModelImportError(error)
            case model.Apply("core.fn", [inputs, outputs]):
                return FunctionType(
                    self.import_type_row(inputs), self.import_type_row(outputs)
                )
            case _:
                error = "Invalid signature."
                raise ModelImportError(error, term)

    def lookup_var(self, name: str) -> "LocalVarData":
        if name not in self.local_vars:
            error = f"Unknown variable `{name}`."
            raise ModelImportError(error)

        return self.local_vars[name]

    def import_type_param(
        self, term: model.Term, bound: TypeBound = TypeBound.Linear
    ) -> TypeParam:
        """Import a TypeParam from a model Term."""
        match term:
            case model.Apply("core.nat"):
                return BoundedNatParam()
            case model.Apply("core.str"):
                return StringParam()
            case model.Apply("core.float"):
                return FloatParam()
            case model.Apply("core.bytes"):
                return BytesParam()
            case model.Apply("core.type"):
                return TypeTypeParam(bound)
            case model.Apply("core.list", [item_type]):
                return ListParam(self.import_type_param(item_type))
            case model.Apply("core.tuple", [item_types]):
                return TupleParam(
                    [
                        self.import_type_param(item_type)
                        for item_type in _import_closed_list(item_types)
                    ]
                )
            case model.Apply("core.const", [runtime_type]):
                return ConstParam(self.import_type(runtime_type))
            case _:
                error = "Failed to import TypeParam."
                raise ModelImportError(error, term)

    def import_type_arg(self, term: model.Term) -> TypeArg:
        """Import a TypeArg from a model Term."""

        def import_list(term: model.Term) -> TypeArg:
            lists: list[TypeArg] = []

            for group in _group_seq_parts(term.to_list_parts()):
                if isinstance(group, list):
                    lists.append(
                        ListArg([self.import_type_arg(item) for item in group])
                    )
                else:
                    lists.append(self.import_type_arg(group))

            return ListConcatArg(lists).flatten()

        def import_tuple(term: model.Term) -> TypeArg:
            tuples: list[TypeArg] = []

            for group in _group_seq_parts(term.to_list_parts()):
                if isinstance(group, list):
                    tuples.append(
                        TupleArg([self.import_type_arg(item) for item in group])
                    )
                else:
                    tuples.append(self.import_type_arg(group))

            return TupleConcatArg(tuples).flatten()

        match term:
            case model.Literal(str() as value):
                return StringArg(value)
            case model.Literal(int() as value):
                return BoundedNatArg(value)
            case model.Literal(float() as value):
                return FloatArg(value)
            case model.Literal(bytes() as value):
                return BytesArg(value)
            case model.List():
                return import_list(term)
            case model.Tuple():
                return import_tuple(term)
            case _:
                # Assume it's a TypeTypeArg
                return TypeTypeArg(self.import_type(term))

    def import_type(self, term: model.Term) -> Type:
        """Import the type from a model Term."""
        match term:
            case model.Apply("core.fn", [inputs, outputs]):
                return FunctionType(
                    self.import_type_row(inputs), self.import_type_row(outputs)
                )
            case model.Apply("core.adt", [variants]):
                return Sum(
                    [
                        self.import_type_row(variant)
                        for variant in _import_closed_list(variants)
                    ]
                )
            case model.Apply("prelude.qubit", []):
                return _QubitDef()
            case model.Apply(symbol, args):
                extension, type_id = _split_extension_name(symbol)
                return Opaque(
                    id=type_id,
                    extension=extension,
                    # TODO How to determine the type bound (Copyable or Linear)?
                    bound=TypeBound.Copyable,
                    args=[self.import_type_arg(arg) for arg in args],
                )
            case model.Var(name):
                var_data = self.lookup_var(name)
                return Variable(idx=var_data.index, bound=var_data.bound)
            case _:
                error = "Failed to import Type."
                raise ModelImportError(error, term)

    def import_type_row(self, term: model.Term) -> list[Type]:
        def import_part(part: model.SeqPart) -> Type:
            if isinstance(part, model.Splice):
                if isinstance(part.seq, model.Var):
                    var_data = self.lookup_var(part.seq.name)
                    return RowVariable(var_data.index, var_data.bound)
                else:
                    error = "Can only import spliced variables."
                    raise ModelImportError(error, term)
            else:
                return self.import_type(part)

        return [import_part(part) for part in term.to_list_parts()]

    def import_value(self, term: model.Term) -> val.Value:
        match term:
            case model.Apply(
                "arithmetic.int.const",
                [
                    model.Literal(int() as int_logwidth),
                    model.Literal(int() as int_value),
                ],
            ):
                # Ensure value is in signed form for conversion to IntVal:
                width = 1 << int_logwidth
                if int_value >= 1 << (width - 1):
                    int_value -= 1 << width
                return IntVal(int_value, int_logwidth)
            case model.Apply(
                "arithmetic.float.const_f64", [model.Literal(float() as float_value)]
            ):
                return FloatVal(float_value)
            case model.Apply("collections.array.const", [_, array_type, array_values]):
                return ArrayVal(
                    [
                        self.import_value(cast(model.Term, v))
                        for v in array_values.to_list_parts()
                    ],
                    self.import_type(array_type),
                )
            case model.Apply(
                "compat.const_json", [typ, model.Literal(str() as json_str)]
            ):
                json_dict = json.loads(json_str)
                match typ:
                    case model.Apply(typename, args):
                        match typename:
                            case "core.adt":
                                [arg] = args
                                match list(arg.to_list_parts()):
                                    case [model.List() as ts]:
                                        pass
                                    case _:
                                        error = f"Unexpected term: {term}"
                                        raise ModelImportError(error)
                                match json_dict:
                                    case {"c": "ConstExternalSymbol", "v": value}:
                                        return val.Extension(
                                            name="ConstExternalSymbol",
                                            typ=Tuple(
                                                *[
                                                    self.import_type(
                                                        cast(model.Term, t)
                                                    )
                                                    for t in ts.to_list_parts()
                                                ]
                                            ),
                                            val=value,
                                        )
                                    case _:
                                        error = f"Unexpected term: {term}"
                                        raise ModelImportError(error)
                            case _:
                                extension, type_id = _split_extension_name(typename)
                                match json_dict:
                                    case {"c": name, "v": value}:
                                        # Determine appropriate TypeBound
                                        bound = TypeBound.Copyable
                                        if typename == "collections.list.List":
                                            [arg] = args
                                            datatype = self.import_type(arg)
                                            bound = datatype.type_bound()
                                        # TODO Determine type bound in other cases
                                        return val.Extension(
                                            name=name,
                                            typ=Opaque(
                                                id=type_id,
                                                bound=bound,
                                                args=[
                                                    self.import_type_arg(arg)
                                                    for arg in args
                                                ],
                                                extension=extension,
                                            ),
                                            val=value,
                                        )
                                    case _:
                                        error = f"Unexpected term: {term}"
                                        raise ModelImportError(error)
                    case _:
                        error = f"Unexpected compat.const_json type: {typ}"
                        raise ModelImportError(error)
            case model.Apply("core.const.adt", [variants, _types, tag, values]):
                match tag:
                    case model.Literal(int() as tagval):
                        pass
                    case _:
                        error = f"Unexpected tag: {tag}"
                        raise ModelImportError(error)
                return val.Sum(
                    tag=tagval,
                    typ=Sum(
                        variant_rows=[
                            [
                                self.import_type(cast(model.Term, t))
                                for t in cast(model.Term, variant).to_list_parts()
                            ]
                            for variant in variants.to_list_parts()
                        ]
                    ),
                    vals=[
                        self.import_value(cast(model.Term, v))
                        for v in values.to_tuple_parts()
                    ],
                )
            case _:
                error = "Unsupported constant value."
                raise ModelImportError(error, term)


@dataclass
class LocalVarData:
    """Data describing a local variable."""

    index: int
    type: TypeParam
    bound: TypeBound = field(default=TypeBound.Linear)


@dataclass
class RegionOrderHints:
    """Order hint metadata."""

    input_keys: list[int] = field(default_factory=list)
    output_keys: list[int] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    key_to_node: dict[int, Node] = field(default_factory=dict)

    def add_node_keys(self, node: Node, keys: Iterable[int]):
        for key in keys:
            if key in self.key_to_node:
                error = f"Duplicate order key `{key}`."
                raise ModelImportError(error)

            self.key_to_node[key] = node

    def get_node_by_key(self, key: int) -> Node:
        if key not in self.key_to_node:
            error = f"Unknown order key `{key}`."
            raise ModelImportError(error)

        return self.key_to_node[key]


def _group_seq_parts(
    parts: Iterable[model.SeqPart],
) -> Generator[model.Term | list[model.Term]]:
    group: list[model.Term] = []

    for part in parts:
        if isinstance(part, model.Splice):
            if len(group) > 0:
                yield group
                group = []
            yield part.seq
        else:
            group.append(part)

    if len(group) > 0:
        yield group


def _import_closed_list(term: model.Term) -> Generator[model.Term]:
    for part in term.to_list_parts():
        if isinstance(part, model.Splice):
            error = "Expected closed list."
            raise ModelImportError(error, term)
        else:
            yield part


def _import_closed_tuple(term: model.Term) -> Generator[model.Term]:
    for part in term.to_tuple_parts():
        if isinstance(part, model.Splice):
            error = "Expected closed tuple."
            raise ModelImportError(error, term)
        else:
            yield part


def _split_extension_name(name: str) -> tuple[str, str]:
    match name.rsplit(".", 1):
        case [extension, id]:
            return (extension, id)
        case [id]:
            return ("", id)
        case _:
            raise AssertionError
