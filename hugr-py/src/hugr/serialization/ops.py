from __future__ import annotations
import inspect
import sys
from abc import ABC
from typing import Any, Literal

from pydantic import Field, RootModel, ConfigDict

from . import tys
from .tys import (
    ExtensionId,
    ExtensionSet,
    FunctionType,
    PolyFuncType,
    Type,
    TypeRow,
    SumType,
    TypeBound,
    ConfiguredBaseModel,
    classes as tys_classes,
    model_rebuild as tys_model_rebuild,
)
from hugr.utils import deser_it

NodeID = int


class BaseOp(ABC, ConfiguredBaseModel):
    """Base class for ops that store their node's input/output types"""

    # Parent node index of node the op belongs to, used only at serialization time
    parent: NodeID

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        """Hook to insert type information from the input and output ports into the
        op"""

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        """Hook to insert type information from a child dataflow graph"""

    def display_name(self) -> str:
        """Name of the op for visualisation"""
        return self.__class__.__name__

    def deserialize(self) -> _ops.Op:
        """Deserializes the model into the corresponding Op."""
        return _ops.SerWrap(self)


# ----------------------------------------------------------
# --------------- Module level operations ------------------
# ----------------------------------------------------------


class Module(BaseOp):
    """The root of a module, parent of all other `ModuleOp`s."""

    op: Literal["Module"] = "Module"


class FuncDefn(BaseOp):
    """A function definition. Children nodes are the body of the definition."""

    op: Literal["FuncDefn"] = "FuncDefn"

    name: str
    signature: PolyFuncType


class FuncDecl(BaseOp):
    """External function declaration, linked at runtime."""

    op: Literal["FuncDecl"] = "FuncDecl"
    name: str
    signature: PolyFuncType


class CustomConst(ConfiguredBaseModel):
    c: str
    v: Any


class ExtensionValue(ConfiguredBaseModel):
    """An extension constant value, that can check it is of a given [CustomType]."""

    v: Literal["Extension"] = Field("Extension", title="ValueTag")
    extensions: ExtensionSet
    typ: Type
    value: CustomConst


class FunctionValue(ConfiguredBaseModel):
    """A higher-order function value."""

    v: Literal["Function"] = Field("Function", title="ValueTag")
    hugr: Any  # TODO


class TupleValue(ConfiguredBaseModel):
    """A constant tuple value."""

    v: Literal["Tuple"] = Field("Tuple", title="ValueTag")
    vs: list["Value"]


class SumValue(ConfiguredBaseModel):
    """A Sum variant

    For any Sum type where this value meets the type of the variant indicated by the tag
    """

    v: Literal["Sum"] = Field("Sum", title="ValueTag")
    tag: int
    typ: SumType
    vs: list["Value"]
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A Sum variant For any Sum type where this value meets the type "
                "of the variant indicated by the tag."
            ),
        }
    )


class Value(RootModel):
    """A constant Value."""

    root: ExtensionValue | FunctionValue | TupleValue | SumValue = Field(
        discriminator="v"
    )

    model_config = ConfigDict(json_schema_extra={"required": ["v"]})


class Const(BaseOp):
    """A Const operation definition."""

    op: Literal["Const"] = "Const"
    v: Value = Field()


# -----------------------------------------------
# --------------- BasicBlock types ------------------
# -----------------------------------------------


class DataflowBlock(BaseOp):
    """A CFG basic block node. The signature is that of the internal Dataflow
    graph."""

    op: Literal["DataflowBlock"] = "DataflowBlock"
    inputs: TypeRow = Field(default_factory=list)
    other_outputs: TypeRow = Field(default_factory=list)
    sum_rows: list[TypeRow]
    extension_delta: ExtensionSet = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        num_cases = len(out_types)
        self.sum_rows = [[] for _ in range(num_cases)]

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.inputs = inputs
        pred = outputs[0].root
        assert isinstance(pred, tys.SumType)
        if isinstance(pred.root, tys.UnitSum):
            self.sum_rows = [[] for _ in range(pred.root.size)]
        else:
            self.sum_rows = []
            for variant in pred.root.rows:
                self.sum_rows.append(variant)
        self.other_outputs = outputs[1:]

        # Needed to avoid random '\n's in the pydantic description

    model_config = ConfigDict(
        json_schema_extra={
            "description": "A CFG basic block node. The signature is that of the internal Dataflow graph.",
        }
    )


class ExitBlock(BaseOp):
    """The single exit node of the CFG, has no children, stores the types of
    the CFG node output."""

    op: Literal["ExitBlock"] = "ExitBlock"
    cfg_outputs: TypeRow

    model_config = ConfigDict(
        json_schema_extra={
            # Needed to avoid random '\n's in the pydantic description
            "description": "The single exit node of the CFG, has no children, stores the types of the CFG node output.",
        }
    )


# ---------------------------------------------
# --------------- DataflowOp ------------------
# ---------------------------------------------


class DataflowOp(BaseOp):
    pass


class Input(DataflowOp):
    """An input node. The outputs of this node are the inputs to the parent node."""

    op: Literal["Input"] = "Input"
    types: TypeRow = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert len(in_types) == 0
        self.types = list(out_types)

    def deserialize(self) -> _ops.Input:
        return _ops.Input(types=[t.deserialize() for t in self.types])


class Output(DataflowOp):
    """An output node. The inputs are the outputs of the function."""

    op: Literal["Output"] = "Output"
    types: TypeRow = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert len(out_types) == 0
        self.types = list(in_types)

    def deserialize(self) -> _ops.Output:
        return _ops.Output(deser_it(self.types))


class Call(DataflowOp):
    """
    Call a function directly.

    The first port is connected to the def/declare of the function being called
    directly, with a `ConstE<Graph>` edge. The signature of the remaining ports matches
    the function being called.
    """

    op: Literal["Call"] = "Call"
    func_sig: PolyFuncType
    type_args: list[tys.TypeArg]
    instantiation: FunctionType

    model_config = ConfigDict(
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra={
            "description": (
                "Operation to call a function directly. The first port is "
                "connected to the def/declare of the function being called directly, "
                "with a `Static<FunctionType>` edge. The signature of the remaining "
                "ports matches the function being called."
            )
        }
    )


class CallIndirect(DataflowOp):
    """Call a function indirectly.

    Like call, but the first input is a standard dataflow graph type.
    """

    op: Literal["CallIndirect"] = "CallIndirect"
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        fun_ty = in_types[0].root
        assert isinstance(fun_ty, FunctionType)
        assert len(fun_ty.input) == len(in_types) - 1
        assert len(fun_ty.output) == len(out_types)
        self.signature = fun_ty


class LoadConstant(DataflowOp):
    """An operation that loads a static constant in to the local dataflow graph."""

    op: Literal["LoadConstant"] = "LoadConstant"
    datatype: Type


class LoadFunction(DataflowOp):
    """Load a static function in to the local dataflow graph."""

    op: Literal["LoadFunction"] = "LoadFunction"
    func_sig: PolyFuncType
    type_args: list[tys.TypeArg]
    signature: FunctionType


class DFG(DataflowOp):
    """A simply nested dataflow graph."""

    op: Literal["DFG"] = "DFG"
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.signature = FunctionType(
            input=list(inputs), output=list(outputs), extension_reqs=ExtensionSet([])
        )

    def deserialize(self) -> _ops.DFG:
        return _ops.DFG(self.signature.deserialize())


# ------------------------------------------------
# --------------- ControlFlowOp ------------------
# ------------------------------------------------


class Conditional(DataflowOp):
    """Conditional operation, defined by child `Case` nodes for each branch."""

    op: Literal["Conditional"] = "Conditional"
    other_inputs: TypeRow = Field(default_factory=list)  # Remaining input types
    outputs: TypeRow = Field(default_factory=list)  # Output types
    sum_rows: list[TypeRow] = Field(
        description="The possible rows of the Sum input", default_factory=list
    )
    # Extensions used to produce the outputs
    extension_delta: ExtensionSet = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        # First port is a predicate, i.e. a sum of tuple types. We need to unpack
        # those into a list of type rows
        pred = in_types[0]
        assert isinstance(pred.root, tys.SumType)
        sum = pred.root.root
        if isinstance(sum, tys.UnitSum):
            self.sum_rows = [[] for _ in range(sum.size)]
        else:
            assert isinstance(sum, tys.GeneralSum)
            self.sum_rows = []
            for ty in sum.rows:
                self.sum_rows.append(ty)
        self.other_inputs = list(in_types[1:])
        self.outputs = list(out_types)


class Case(BaseOp):
    """Case ops - nodes valid inside Conditional nodes."""

    op: Literal["Case"] = "Case"
    # The signature of the contained dataflow graph.
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.signature = tys.FunctionType(
            input=list(inputs), output=list(outputs), extension_reqs=ExtensionSet([])
        )


class TailLoop(DataflowOp):
    """Tail-controlled loop."""

    op: Literal["TailLoop"] = "TailLoop"
    just_inputs: TypeRow = Field(default_factory=list)  # Types that are only input
    just_outputs: TypeRow = Field(default_factory=list)  # Types that are only output
    # Types that are appended to both input and output:
    rest: TypeRow = Field(default_factory=list)
    extension_delta: ExtensionSet = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert in_types == out_types
        # self.just_inputs = list(in_types)
        # self.just_outputs = list(out_types)
        self.rest = list(in_types)


class CFG(DataflowOp):
    """A dataflow node which is defined by a child CFG."""

    op: Literal["CFG"] = "CFG"
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_port_types(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.signature = FunctionType(
            input=list(inputs), output=list(outputs), extension_reqs=ExtensionSet([])
        )


ControlFlowOp = Conditional | TailLoop | CFG


class CustomOp(DataflowOp):
    """A user-defined operation that can be downcasted by the extensions that define
    it."""

    op: Literal["CustomOp"] = "CustomOp"
    extension: ExtensionId
    op_name: str
    signature: tys.FunctionType = Field(default_factory=tys.FunctionType.empty)
    description: str = ""
    args: list[tys.TypeArg] = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        self.signature = tys.FunctionType(input=list(in_types), output=list(out_types))

    def display_name(self) -> str:
        return self.op_name

    def deserialize(self) -> _ops.Custom:
        return _ops.Custom(
            extension=self.extension,
            op_name=self.op_name,
            signature=self.signature.deserialize(),
            args=deser_it(self.args),
        )

    model_config = ConfigDict(
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra={
            "description": (
                "A user-defined operation that can be downcasted by the extensions that "
                "define it."
            )
        }
    )


class Noop(DataflowOp):
    """A no-op operation."""

    op: Literal["Noop"] = "Noop"
    ty: Type

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert len(in_types) == 1
        assert len(out_types) == 1
        assert in_types[0] == out_types[0]
        self.ty = in_types[0]


class MakeTuple(DataflowOp):
    """An operation that packs all its inputs into a tuple."""

    op: Literal["MakeTuple"] = "MakeTuple"
    tys: TypeRow = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        # If we have a single order edge as input, this is a unit
        if in_types == [None]:
            in_types = []
        self.tys = list(in_types)

    def deserialize(self) -> _ops.MakeTupleDef:
        return _ops.MakeTupleDef(deser_it(self.tys))


class UnpackTuple(DataflowOp):
    """An operation that packs all its inputs into a tuple."""

    op: Literal["UnpackTuple"] = "UnpackTuple"
    tys: TypeRow = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        self.tys = list(out_types)

    def deserialize(self) -> _ops.UnpackTupleDef:
        return _ops.UnpackTupleDef(deser_it(self.tys))


class Tag(DataflowOp):
    """An operation that creates a tagged sum value from one of its variants."""

    op: Literal["Tag"] = "Tag"
    tag: int  # The variant to create.
    variants: list[TypeRow]  # The variants of the sum type.


class Lift(DataflowOp):
    """Fixes some TypeParams of a polymorphic type by providing TypeArgs."""

    op: Literal["Lift"] = "Lift"
    type_row: TypeRow
    new_extension: ExtensionId


class AliasDecl(BaseOp):
    op: Literal["AliasDecl"] = "AliasDecl"
    name: str
    bound: TypeBound


class AliasDefn(BaseOp):
    op: Literal["AliasDefn"] = "AliasDefn"
    name: str
    definition: Type


class OpType(RootModel):
    """A constant operation."""

    root: (
        Module
        | Case
        | FuncDefn
        | FuncDecl
        | Const
        | DataflowBlock
        | ExitBlock
        | Conditional
        | TailLoop
        | CFG
        | Input
        | Output
        | Call
        | CallIndirect
        | LoadConstant
        | LoadFunction
        | CustomOp
        | Noop
        | MakeTuple
        | UnpackTuple
        | Tag
        | Lift
        | DFG
        | AliasDecl
        | AliasDefn
    ) = Field(discriminator="op")

    model_config = ConfigDict(json_schema_extra={"required": ["parent", "op"]})


# --------------------------------------
# --------------- OpDef ----------------
# --------------------------------------


class FixedHugr(ConfiguredBaseModel):
    extensions: ExtensionSet
    hugr: Any


class OpDef(ConfiguredBaseModel, populate_by_name=True):
    """Serializable definition for dynamically loaded operations."""

    extension: ExtensionId
    name: str  # Unique identifier of the operation.
    description: str  # Human readable description of the operation.
    misc: dict[str, Any] | None = None
    signature: PolyFuncType | None = None
    lower_funcs: list[FixedHugr]


# Now that all classes are defined, we need to update the ForwardRefs in all type
# annotations. We use some inspect magic to find all classes defined in this file.
classes = (
    inspect.getmembers(
        sys.modules[__name__],
        lambda member: inspect.isclass(member) and member.__module__ == __name__,
    )
    + tys_classes
)

tys_model_rebuild(dict(classes))

from hugr import _ops  # noqa: E402  # needed to avoid circular imports
