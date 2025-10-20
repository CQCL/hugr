"""Helpers to import hugr graphs from hugr model to their python representation."""

import json
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import Any

import hugr.model as model
from hugr import val
from hugr.hugr import InPort, OutPort
from hugr.hugr.base import Hugr
from hugr.hugr.node_port import Node
from hugr.ops import (
    DFG,
    Call,
    Case,
    Conditional,
    Custom,
    FuncDecl,
    FuncDefn,
    Input,
    Op,
    Output,
    TailLoop,
)
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
    TupleArg,
    TupleConcatArg,
    TupleParam,
    Type,
    TypeArg,
    TypeBound,
    TypeParam,
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


class ModelImport:
    """Helper to import a Hugr."""

    local_vars: dict[str, "LocalVarData"]
    current_symbol: str | None
    linked_ports: tuple[dict[str, list[InPort]], dict[str, list[OutPort]]]
    static_edges: list[tuple[Node, Node]]

    module: model.Module
    symbols: dict[str, model.Node]
    fn_nodes: dict[str, Node]
    hugr: Hugr

    def __init__(self, module: model.Module):
        self.local_vars = {}
        self.current_symbol = None
        self.module = module
        self.symbols = {}
        self.hugr = Hugr()
        self.linked_ports = ({}, {})
        self.static_edges = []
        self.fn_nodes = {}

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
        link_ports_in = self.linked_ports[0]

        for offset, link in enumerate(links):
            in_port = InPort(node=node, offset=offset)
            link_ports_in.setdefault(link, []).append(in_port)

    def record_out_links(self, node: Node, links: Iterable[str]):
        link_ports_out = self.linked_ports[1]

        for offset, link in enumerate(links):
            out_port = OutPort(node=node, offset=offset)
            link_ports_out.setdefault(link, []).append(out_port)

    def link_ports(self):
        link_ports_in, link_ports_out = self.linked_ports

        links = link_ports_in.keys() | link_ports_out.keys()

        for link in links:
            in_ports = link_ports_in[link]
            out_ports = link_ports_out[link]

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
        for src, dst in self.static_edges:
            out_port_offset = self.hugr.num_out_ports(src) - 1
            out_port = OutPort(node=src, offset=out_port_offset)

            in_port_offset = self.hugr.num_in_ports(dst)
            in_port = InPort(node=dst, offset=in_port_offset)

            self.hugr.add_link(out_port, in_port)

    def import_dfg_region(self, region: model.Region, parent: Node):
        signature = self.import_signature(region.signature)

        input_node = self.hugr.add_node(Input(signature.input), parent=parent)
        self.record_out_links(input_node, region.sources)

        output_node = self.hugr.add_node(Output(signature.output), parent=parent)
        self.record_in_links(output_node, region.targets)

        order_data = self.import_meta_order_region(region)
        order_data.add_node_keys(input_node, order_data.input_keys)
        order_data.add_node_keys(output_node, order_data.output_keys)

        for child in region.children:
            child_id = self.import_node_in_dfg(child, parent)
            child_order_keys = self.import_meta_order_keys(child)
            order_data.add_node_keys(child_id, child_order_keys)

        for src_key, tgt_key in order_data.edges:
            src_node = order_data.get_node_by_key(src_key)
            tgt_node = order_data.get_node_by_key(tgt_key)
            self.hugr.add_order_link(src_node, tgt_node)

    def import_node_in_dfg(self, node: model.Node, parent: Node) -> Node:
        def import_dfg_node() -> Node:
            match node.regions:
                case [body]:
                    pass
                case _:
                    error = "DFG node expects a dataflow region."
                    raise ModelImportError(error, node)

            signature = self.import_signature(node.signature)
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
            )
            self.import_dfg_region(body, node_id)
            return node_id

        def import_custom_node(op: model.Term) -> Node:
            match op:
                case model.Apply(symbol, args):
                    extension, op_name = _split_extension_name(symbol)
                case _:
                    error = "The operation of a custom node must be a symbol "
                    "application."
                    raise ModelImportError(error, node)

            if symbol == "core.call":
                input_types, output_types, func = args
                match func:
                    case model.Apply(symbol, args):
                        sig = self.import_signature(node.signature)
                        callnode = self.add_node(
                            node,
                            Call(
                                signature=PolyFuncType([], sig),  # TODO params
                                instantiation=sig,
                                type_args=[self.import_type_arg(arg) for arg in args],
                            ),
                            parent,
                        )
                        self.static_edges.append((self.fn_nodes[symbol], callnode))
                        return callnode
                    case _:
                        error = "The function of a Call node must be a symbol "
                        "application."
                        raise ModelImportError(error, node)

            return self.add_node(
                node,
                Custom(
                    op_name=op_name,
                    extension=extension,
                    signature=self.import_signature(node.signature),
                    args=[self.import_type_arg(arg) for arg in args],
                ),
                parent,
            )

        def import_cfg() -> Node:
            # TODO
            return Node(0)

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

    def import_node_in_module(self, node: model.Node) -> Node | None:
        def import_declare_func(symbol: model.Symbol) -> Node:
            title = self.import_meta_title(node)
            f_name = symbol.name if title is None else title
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
            self.fn_nodes[f_name] = node_id
            return node_id

        def import_define_func(symbol: model.Symbol) -> Node:
            title = self.import_meta_title(node)
            f_name = symbol.name if title is None else title
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
            self.fn_nodes[f_name] = node_id
            return node_id

        match node.operation:
            case model.DeclareFunc(symbol):
                return import_declare_func(symbol)
            case model.DefineFunc(symbol):
                return import_define_func(symbol)
            case model.DeclareAlias():
                error = "Aliases unsupported for now."
                raise ModelImportError(error, node)
            case model.DefineAlias():
                error = "Aliases unsupported for now."
                raise ModelImportError(error, node)
            case model.Import():
                return None
            case model.DeclareConstructor():
                return None
            case model.DeclareOperation():
                return None
            case _:
                error = "Unexpected node in module region."
                raise ModelImportError(error, node)

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
        if name in self.local_vars:
            error = f"Unknown variable `{name}`."
            raise ImportError(error)

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

        # TODO: TypeTypeArg

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
                error = "Failed to import TypeArg."
                raise ModelImportError(error, term)

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
                extension, id = _split_extension_name(symbol)
                return Opaque(
                    id=id,
                    extension=extension,
                    bound=TypeBound.Linear,
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

    def import_meta_title(self, node: model.Node) -> str | None:
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

    def import_meta_order_region(self, region: model.Region) -> "RegionOrderHints":
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

    def import_meta_order_keys(self, node: model.Node) -> list[int]:
        """Collects all order hint keys in the metadata of a node."""
        keys = []

        for meta in node.meta:
            match meta:
                case model.Apply("core.order_hint.key", [model.Literal(int() as key)]):
                    keys.append(key)
                case _:
                    pass

        return keys

    def import_value(self, term: model.Term) -> val.Value:
        match term:
            case model.Apply(
                "arithmetic.int.const",
                [
                    model.Literal(int() as int_bitwidth),
                    model.Literal(int() as int_value),
                ],
            ):
                return IntVal(int_value, int_bitwidth)
            case model.Apply(
                "arithmetic.float.const_f64", [model.Literal(float() as float_value)]
            ):
                return FloatVal(float_value)
            case model.Apply(
                "collections.array.const", [_, _array_item_type, _array_items]
            ):
                # TODO
                error = "Import array constants"
                raise NotImplementedError(error)
            case model.Apply(
                "compat.const_json",
                [
                    model.Literal(str() as _json),
                ],
            ):
                # TODO
                error = "Import json encoded constants"
                raise NotImplementedError(error)
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
        if key in self.key_to_node:
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
