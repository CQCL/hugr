from hugr.ops import Case
from hugr.ops import Conditional
from hugr.ops import TailLoop
from hugr.tys import Opaque
from hugr.tys import TupleConcatArg
from hugr.tys import TupleArg
from hugr.tys import ListConcatArg
from hugr.tys import ListArg
from hugr.tys import ConstParam
from hugr.tys import ListParam
from hugr.tys import TupleParam
from hugr.tys import Sum
from hugr.tys import BytesArg
from hugr.tys import FloatArg
from hugr.tys import BoundedNatArg
from hugr.tys import StringArg
from hugr.tys import TypeBound
from hugr.tys import TypeTypeParam
from hugr.tys import BoundedNatParam
from hugr.tys import StringParam
from hugr.tys import FloatParam
from hugr.tys import BytesParam
from hugr.tys import TypeParam
from typing import Tuple
from hugr.tys import TypeArg
from hugr.ops import Custom
from hugr.ops import Op
from hugr.ops import DFG
from hugr.tys import FunctionType
from hugr.hugr.node_port import Node
from hugr.hugr.base import Hugr
from hugr.tys import Type
from hugr.tys import RowVariable
from typing import Optional
from contextlib import contextmanager
from typing import Dict
from typing import List
from typing import Callable
from typing import Iterable
from typing import Generator
from typing import Sequence
import hugr.model as model

ImportContext = model.Term | model.Node | model.Region | str

class ModelImportError(Exception):
    """Exception raised when importing from the model representation fails."""

    def __init__(self, message: str,
        location: ImportContext | None = None):
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


class ModelImport:
    local_vars: Dict[str, int]
    current_symbol: str | None
    module: model.Module
    symbols: Dict[str, model.Node]
    hugr: Hugr

    def __init__(self, module: model.Module):
        self.local_vars = {}
        self.current_symbol = None
        self.module = module
        self.symbols = {}
        self.hugr = Hugr()

        for node in module.root.children:
            symbol_name = node.operation.symbol_name()

            if symbol_name is None:
                continue

            if symbol_name in self.symbols:
                error = f"Duplicate symbol name `{symbol_name}`."
                raise ModelImportError(error, node)

            self.symbols[symbol_name] = node


    def add_node(
        self,
        node: model.Node,
        operation: Op,
        parent: Node
    ) -> Node:
        node_id = self.hugr.add_node(operation, parent)

        # TODO: Input and output links

        return node_id

    def import_dfg_region(self, region: model.Region, parent: Node):
        pass

    def import_node_in_dfg(self, node: model.Node, parent: Node) -> Node:
        def import_dfg_node() -> Node:
            match node.regions:
                case [body]:
                    pass
                case _:
                    raise ModelImportError("DFG node expects a dataflow region.", node)

            signature = self.import_signature(node.signature)
            node_id = self.add_node(node, DFG(signature.input, signature.output), parent)
            self.import_dfg_region(body, node_id)
            return node_id

        def import_tail_loop() -> Node:
            match node.regions:
                case [body]:
                    pass
                case _:
                    raise ModelImportError("Loop node expects a dataflow region.", node)

            signature = self.import_signature(node.signature)
            body_signature = self.import_signature(body.signature)

            match body_signature.output:
                case [model.Apply("core.adt", [variants]), *rest]:
                    pass
                case _:
                    raise ModelImportError("TailLoop body expects `(core.adt _)` as first target type.", node)

            match import_closed_list(variants):
                case [just_inputs, just_outputs]:
                    pass
                case _:
                    raise ModelImportError("TailLoop body expects sum type with two variants.", node)

            node_id = self.add_node(node, TailLoop(
                just_inputs = self.import_type_row(just_inputs),
                rest = rest,
                _just_outputs = self.import_type_row(just_outputs)
            ), parent)
            self.import_dfg_region(body, node_id)
            return node_id


        def import_custom_node(op: model.Term) -> Node:
            match op:
                case model.Apply(symbol, args):
                    extension, op_name = split_extension_name(symbol)
                case _:
                    raise ModelImportError("The operation of a custom node must be a symbol application.", node)

            return self.add_node(node, Custom(
                op_name = op_name,
                extension = extension,
                signature = self.import_signature(node.signature),
                args = [self.import_type_arg(arg) for arg in args]
            ), parent)

        def import_cfg() -> Node:
            ...

        def import_conditional() -> Node:
            signature = self.import_signature(node.signature)

            match signature.input:
                case [model.Apply("core.adt", [variants]), *other_inputs]:
                    sum_ty = Sum([
                        self.import_type_row(variant)
                        for variant in import_closed_list(variants)
                    ])
                case _:
                    raise ModelImportError("Conditional node expects `(core.adt _)` as first input type.", node)

            node_id = self.add_node(node, Conditional(
                sum_ty = sum_ty,
                other_inputs = other_inputs,
                _outputs = signature.output
            ), parent)

            for case_body in node.regions:
                case_signature = self.import_signature(case_body.signature)
                case_id = self.hugr.add_node(
                    Case(
                        inputs = case_signature.input,
                        _outputs = case_signature.output
                    ),
                    node_id
                )
                self.import_dfg_region(case_body, case_id)

            return node_id

        match node.operation:
            case model.InvalidOp():
                raise ModelImportError("Invalid operation can not be imported.", node)
            case model.Dfg():
                return import_dfg_node()
            case model.Cfg():
                return import_cfg()
            case model.Block():
                raise ModelImportError("Unexpected basic block.", node)
            case model.CustomOp(op):
                return import_custom_node(op)
            case model.TailLoop():
                return import_tail_loop()
            case model.Conditional():
                return import_conditional()
            case _:
                raise ModelImportError("Unexpected node in DFG region.", node)

    def import_node_in_module(self, node: model.Node, parent: Node) -> Optional[Node]:
        match node.operation:
            case model.DeclareFunc():
                ...
            case model.DefineFunc():
                ...
            case model.DeclareAlias():
                ...
            case model.DefineAlias():
                ...
            case model.Import():
                return None
            case model.DeclareConstructor():
                return None
            case model.DeclareOperation():
                return None
            case _:
                raise ModelImportError("Unexpected node in module region.", node)

    def import_signature(self, term: Optional[model.Term]) -> FunctionType:
        match term:
            case None:
                raise ModelImportError("Signature required.")
            case model.Apply("core.fn", [inputs, outputs]):
                return FunctionType(
                    self.import_type_row(inputs),
                    self.import_type_row(outputs)
                )
            case _:
                raise ModelImportError("Invalid signature.", term)

    def open_symbol(self, name: str):
        @contextmanager
        def context():
            if self.current_symbol is not None:
                raise ModelImportError("Symbols can not be nested.")
            self.current_symbol = name
            yield
            self.current_symbol = None
            self.local_vars = {}
        return context()

    def lookup_var(self, name: str) -> Optional[int]:
        self.local_vars[name]


    def import_type_param(self, term: model.Term) -> TypeParam:
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
                return TypeTypeParam(TypeBound.Copyable)
            case model.Apply("core.list", [item_type]):
                return ListParam(self.import_type_param(item_type))
            case model.Apply("core.tuple", [item_types]):
                return TupleParam([
                    self.import_type_param(item_type)
                    for item_type in import_closed_list(item_types)
                ])
            case model.Apply("core.const", [runtime_type]):
                return ConstParam(self.import_type(runtime_type))
            case _:
                raise ModelImportError("Failed to import TypeParam.", term)

    def import_type_arg(self, term: model.Term) -> TypeArg:
        """Import a TypeArg from a model Term."""

        def import_list(term: model.Term) -> TypeArg:
            lists = []

            for group in group_seq_parts(term.to_list_parts()):
                if isinstance(group, list):
                    lists.append(ListArg([self.import_type_arg(item) for item in group]))
                else:
                    lists.append(self.import_type_arg(group))

            return ListConcatArg(lists).flatten()

        def import_tuple(term: model.Term) -> TypeArg:
            tuples = []

            for group in group_seq_parts(term.to_list_parts()):
                if isinstance(group, list):
                    tuples.append(TupleArg([self.import_type_arg(item) for item in group]))
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
                raise ModelImportError("Failed to import TypeArg.", term)

    def import_type(self, term: model.Term) -> Type:
        """Import the type from a model Term."""

        match term:
            case model.Apply("core.fn", [inputs, outputs]):
                return FunctionType(
                    self.import_type_row(inputs),
                    self.import_type_row(outputs)
                )
            case model.Apply("core.adt", [variants]):
                return Sum([
                    self.import_type_row(variant)
                    for variant in import_closed_list(variants)
                ])
            case model.Apply(symbol, args):
                extension, id = split_extension_name(symbol)
                return Opaque(
                    id = id,
                    extension = extension,
                    bound = TypeBound.Linear,
                    args = [self.import_type_arg(arg) for arg in args]
                )
            case model.Var(name):
                raise NotImplementedError("TODO")
            case _:
                raise ModelImportError("Failed to import Type.", term)

    def import_type_row(self, term: model.Term) -> List[Type]:
        def import_part(part: model.SeqPart) -> Type:
            if isinstance(part, model.Splice):
                if isinstance(part.seq, model.Var):
                    idx = self.lookup_var(part.seq.name)

                    if idx is None:
                        error = f"Unknown variable `{part.seq}`."
                        raise ImportError(error)

                    # TODO: Type bound?
                    return RowVariable(idx, TypeBound.Copyable)
                else:
                    raise ImportError("Can only import spliced variables.", term)
            else:
                return self.import_type(term)

        return [import_part(part) for part in term.to_list_parts()]



def group_seq_parts(
    parts: Iterable[model.SeqPart],
) -> Generator[model.Term | List[model.Term]]:
    group = []

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


def import_closed_list(term: model.Term) -> Generator[model.Term]:
    for part in term.to_list_parts():
        if isinstance(part, model.Splice):
            raise ModelImportError("Expected closed list.", term)
        else:
            yield part

def import_closed_tuple(term: model.Term) -> Generator[model.Term]:
    for part in term.to_tuple_parts():
        if isinstance(part, model.Splice):
            raise ModelImportError("Expected closed tuple.", term)
        else:
            yield part

def split_extension_name(name: str) -> Tuple[str, str]:
    match name.rsplit(".", 1):
        case [extension, id]:
            return (extension, id)
        case [id]:
            return ("", id)
        case _:
            assert False
