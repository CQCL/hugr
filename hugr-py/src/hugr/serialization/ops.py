import inspect
import sys
from abc import ABC
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, RootModel

from . import tys
from .tys import (
    ExtensionId,
    ExtensionSet,
    FunctionType,
    PolyFuncType,
    Type,
    TypeRow,
)

NodeID = int


class BaseOp(ABC, BaseModel):
    """Base class for ops that store their node's input/output types"""

    # Parent node index of node the op belongs to, used only at serialization time
    parent: NodeID
    input_extensions: ExtensionSet = Field(default_factory=ExtensionSet)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        """Hook to insert type information from the input and output ports into the
        op"""

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        """Hook to insert type information from a child dataflow graph"""

    def display_name(self) -> str:
        """Name of the op for visualisation"""
        return self.__class__.__name__


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
    signature: PolyFuncType = Field(default_factory=PolyFuncType.empty)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert len(in_types) == 0
        assert len(out_types) == 1
        out = out_types[0]
        assert isinstance(out, PolyFuncType)
        self.signature = out  # TODO: Extensions


class FuncDecl(BaseOp):
    """External function declaration, linked at runtime."""

    op: Literal["FuncDecl"] = "FuncDecl"
    name: str
    signature: PolyFuncType = Field(default_factory=PolyFuncType.empty)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert len(in_types) == 0
        assert len(out_types) == 1
        out = out_types[0]
        assert isinstance(out, PolyFuncType)
        self.signature = out


class ConstBase(BaseOp):
    """A constant operation definition."""

    op: Literal["Const"] = "Const"


CustomConst = Any  # TODO


class ExtensionConst(ConstBase):
    """An extension constant value, that can check it is of a given [CustomType]."""

    c: Literal["Extension"] = Field("Extension", title="ConstTag")
    e: CustomConst = Field(title="CustomConst")

    class Config:
        json_schema_extra = {
            "required": ["parent", "op", "c", "e"],
        }


class FunctionConst(ConstBase):
    """A higher-order function value."""

    c: Literal["Function"] = Field("Function", title="ConstTag")
    hugr: Any  # TODO

    class Config:
        json_schema_extra = {
            "required": ["parent", "op", "c", "hugr"],
        }


class Tuple(ConstBase):
    """A constant tuple value."""

    c: Literal["Tuple"] = Field("Tuple", title="ConstTag")
    vs: list["Const"]

    class Config:
        json_schema_extra = {
            "required": ["parent", "op", "c", "vs"],
        }


class Sum(ConstBase):
    """A Sum variant

    For any Sum type where this value meets the type of the variant indicated by the tag
    """

    c: Literal["Sum"] = Field("Sum", title="ConstTag")
    tag: int
    typ: Type
    vs: list["Const"]

    class Config:
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra = {
            "description": (
                "A Sum variant For any Sum type where this value meets the type "
                "of the variant indicated by the tag."
            ),
            "required": ["parent", "op", "c", "tag", "typ", "vs"],
        }


class Const(RootModel):
    """A constant operation."""

    root: ExtensionConst | FunctionConst | Tuple | Sum = Field(discriminator="c")


# -----------------------------------------------
# --------------- BasicBlock types ------------------
# -----------------------------------------------


class DataflowBlock(BaseOp):
    """A CFG basic block node. The signature is that of the internal Dataflow
    graph."""

    op: Literal["DataflowBlock"] = "DataflowBlock"
    inputs: TypeRow = Field(default_factory=list)
    other_outputs: TypeRow = Field(default_factory=list)
    sum_rows: list[TypeRow] = Field(default_factory=list)
    extension_delta: ExtensionSet = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        num_cases = len(out_types)
        self.sum_rows = [[] for _ in range(num_cases)]

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.inputs = inputs
        pred = outputs[0]
        assert isinstance(pred, tys.UnitSum | tys.GeneralSum)
        if isinstance(pred, tys.UnitSum):
            self.sum_rows = [[] for _ in range(cast(tys.UnitSum, pred).size)]
        else:
            self.sum_rows = []
            for variant in pred.rows:
                self.sum_rows.append(variant)
        self.other_outputs = outputs[1:]

    class Config:
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra = {
            "required": [
                "parent",
                "op",
                "inputs",
                "other_outputs",
                "sum_rows",
                "extension_delta",
            ],
            "description": "A CFG basic block node. The signature is that of the internal Dataflow graph.",
        }


class ExitBlock(BaseOp):
    """The single exit node of the CFG, has no children, stores the types of
    the CFG node output."""

    op: Literal["ExitBlock"] = "ExitBlock"
    cfg_outputs: TypeRow

    class Config:
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra = {
            "description": "The single exit node of the CFG, has no children, stores the types of the CFG node output."
        }


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


class Output(DataflowOp):
    """An output node. The inputs are the outputs of the function."""

    op: Literal["Output"] = "Output"
    types: TypeRow = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert len(out_types) == 0
        self.types = list(in_types)


class Call(DataflowOp):
    """
    Call a function directly.

    The first port is connected to the def/declare of the function being called
    directly, with a `ConstE<Graph>` edge. The signature of the remaining ports matches
    the function being called.
    """

    op: Literal["Call"] = "Call"
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        # The constE edge comes after the value inputs
        fun_ty = in_types[-1]
        assert isinstance(fun_ty, PolyFuncType)
        poly_func = cast(PolyFuncType, fun_ty)
        assert len(poly_func.params) == 0
        self.signature = poly_func.body

    class Config:
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra = {
            "description": (
                "Operation to call a function directly. The first port is "
                "connected to the def/declare of the function being called directly, "
                "with a `Static<FunctionType>` edge. The signature of the remaining "
                "ports matches the function being called."
            )
        }


class CallIndirect(DataflowOp):
    """Call a function indirectly.

    Like call, but the first input is a standard dataflow graph type."""

    op: Literal["CallIndirect"] = "CallIndirect"
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        fun_ty = in_types[0]
        assert isinstance(fun_ty, PolyFuncType)
        poly_func = cast(PolyFuncType, fun_ty)
        assert len(poly_func.params) == 0
        assert len(poly_func.body.input) == len(in_types) - 1
        assert len(poly_func.body.output) == len(out_types)
        self.signature = poly_func.body


class LoadConstant(DataflowOp):
    """An operation that loads a static constant in to the local dataflow graph."""

    op: Literal["LoadConstant"] = "LoadConstant"
    datatype: Type


class DFG(DataflowOp):
    """A simply nested dataflow graph."""

    op: Literal["DFG"] = "DFG"
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.signature = FunctionType(
            input=list(inputs), output=list(outputs), extension_reqs=ExtensionSet([])
        )


# ------------------------------------------------
# --------------- ControlFlowOp ------------------
# ------------------------------------------------


class Conditional(DataflowOp):
    """Conditional operation, defined by child `Case` nodes for each branch."""

    op: Literal["Conditional"] = "Conditional"
    other_inputs: TypeRow = Field(default_factory=list)  # Remaining input types
    outputs: TypeRow = Field(default_factory=list)  # Output types
    sum_rows: list[TypeRow] = Field(description="The possible rows of the Sum input")
    # Extensions used to produce the outputs
    extension_delta: ExtensionSet = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        # First port is a predicate, i.e. a sum of tuple types. We need to unpack
        # those into a list of type rows
        pred = in_types[0]
        if isinstance(pred, tys.UnitSum):
            self.sum_rows = [[] for _ in range(cast(tys.UnitSum, pred).size)]
        else:
            assert isinstance(pred, tys.GeneralSum)
            self.sum_rows = []
            for ty in pred.rows:
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

    class Config:
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra = {
            "description": (
                "A user-defined operation that can be downcasted by the extensions that "
                "define it."
            )
        }


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


class UnpackTuple(DataflowOp):
    """An operation that packs all its inputs into a tuple."""

    op: Literal["UnpackTuple"] = "UnpackTuple"
    tys: TypeRow = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        self.tys = list(out_types)


class Tag(DataflowOp):
    """An operation that creates a tagged sum value from one of its variants."""

    op: Literal["Tag"] = "Tag"
    tag: int  # The variant to create.
    variants: TypeRow  # The variants of the sum type.


class Lift(DataflowOp):
    """Fixes some TypeParams of a polymorphic type by providing TypeArgs."""

    op: Literal["Lift"] = "Lift"
    type_row: TypeRow
    new_extension: ExtensionId


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
        | CustomOp
        | Noop
        | MakeTuple
        | UnpackTuple
        | Tag
        | Lift
        | DFG
    ) = Field(discriminator="op")


# --------------------------------------
# --------------- OpDef ----------------
# --------------------------------------


class OpDef(BaseOp, populate_by_name=True):
    """Serializable definition for dynamically loaded operations."""

    name: str  # Unique identifier of the operation.
    description: str  # Human readable description of the operation.
    inputs: list[tuple[str | None, Type]]
    outputs: list[tuple[str | None, Type]]
    misc: dict[str, Any]  # Miscellaneous data associated with the operation.
    def_: str | None = Field(
        ..., alias="def"
    )  # (YAML?)-encoded definition of the operation.
    extension_reqs: ExtensionSet  # Resources required to execute this operation.


# Now that all classes are defined, we need to update the ForwardRefs in all type
# annotations. We use some inspect magic to find all classes defined in this file.
classes = inspect.getmembers(
    sys.modules[__name__],
    lambda member: inspect.isclass(member) and member.__module__ == __name__,
)
for _, c in classes:
    if issubclass(c, BaseModel):
        c.model_rebuild()
