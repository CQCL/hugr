from __future__ import annotations

import inspect
import sys
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import ConfigDict, Field, RootModel

from hugr.hugr.node_port import (
    NodeIdx,  # noqa: TCH001 # pydantic needs this alias in scope
)
from hugr.utils import deser_it

from . import tys as stys
from .tys import (
    ConfiguredBaseModel,
    ExtensionId,
    FunctionType,
    PolyFuncType,
    SumType,
    Type,
    TypeBound,
    TypeRow,
)
from .tys import (
    classes as tys_classes,
)
from .tys import (
    model_rebuild as tys_model_rebuild,
)


class BaseOp(ABC, ConfiguredBaseModel):
    """Base class for ops that store their node's input/output types."""

    # Parent node index of node the op belongs to, used only at serialization time
    parent: NodeIdx

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        """Hook to insert type information from the input and output ports into the
        op.
        """

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        """Hook to insert type information from a child dataflow graph."""

    def display_name(self) -> str:
        """Name of the op for visualisation."""
        return self.__class__.__name__

    @abstractmethod
    def deserialize(self) -> ops.Op:
        """Deserializes the model into the corresponding Op."""


# ----------------------------------------------------------
# --------------- Module level operations ------------------
# ----------------------------------------------------------


class Module(BaseOp):
    """The root of a module, parent of all other `ModuleOp`s."""

    op: Literal["Module"] = "Module"

    def deserialize(self) -> ops.Module:
        return ops.Module()


class FuncDefn(BaseOp):
    """A function definition. Children nodes are the body of the definition."""

    op: Literal["FuncDefn"] = "FuncDefn"

    name: str
    signature: PolyFuncType

    def deserialize(self) -> ops.FuncDefn:
        poly_func = self.signature.deserialize()
        return ops.FuncDefn(
            self.name, inputs=poly_func.body.input, _outputs=poly_func.body.output
        )


class FuncDecl(BaseOp):
    """External function declaration, linked at runtime."""

    op: Literal["FuncDecl"] = "FuncDecl"
    name: str
    signature: PolyFuncType

    def deserialize(self) -> ops.FuncDecl:
        return ops.FuncDecl(self.name, self.signature.deserialize())


class CustomConst(ConfiguredBaseModel):
    c: str
    v: Any


class BaseValue(ABC, ConfiguredBaseModel):
    @abstractmethod
    def deserialize(self) -> val.Value: ...


class CustomValue(BaseValue):
    """An extension constant value, that can check it is of a given [CustomType]."""

    v: Literal["Extension"] = Field(default="Extension", title="ValueTag")
    typ: Type
    value: CustomConst

    def deserialize(self) -> val.Value:
        return val.Extension(
            name=self.value.c,
            typ=self.typ.deserialize(),
            val=self.value.v,
        )


class FunctionValue(BaseValue):
    """A higher-order function value."""

    v: Literal["Function"] = Field(default="Function", title="ValueTag")
    hugr: Any

    def deserialize(self) -> val.Value:
        from hugr._serialization.serial_hugr import SerialHugr
        from hugr.hugr import Hugr

        # pydantic stores the serialized dictionary because of the "Any" annotation
        return val.Function(Hugr._from_serial(SerialHugr(**self.hugr)))


class TupleValue(BaseValue):
    """A constant tuple value."""

    v: Literal["Tuple"] = Field(default="Tuple", title="ValueTag")
    vs: list[Value]

    def deserialize(self) -> val.Value:
        return val.Tuple(*deser_it(v.root for v in self.vs))


class SumValue(BaseValue):
    """A Sum variant.

    For any Sum type where this value meets the type of the variant indicated by the tag
    """

    v: Literal["Sum"] = Field(default="Sum", title="ValueTag")
    tag: int
    typ: SumType
    vs: list[Value]
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A Sum variant For any Sum type where this value meets the type "
                "of the variant indicated by the tag."
            ),
        }
    )

    def deserialize(self) -> val.Value:
        return val.Sum(
            self.tag, self.typ.deserialize(), deser_it(v.root for v in self.vs)
        )


class Value(RootModel):
    """A constant Value."""

    root: CustomValue | FunctionValue | TupleValue | SumValue = Field(discriminator="v")

    model_config = ConfigDict(json_schema_extra={"required": ["v"]})

    def deserialize(self) -> val.Value:
        return self.root.deserialize()


class Const(BaseOp):
    """A Const operation definition."""

    op: Literal["Const"] = "Const"
    v: Value = Field()

    def deserialize(self) -> ops.Const:
        return ops.Const(self.v.deserialize())


# -----------------------------------------------
# --------------- BasicBlock types ------------------
# -----------------------------------------------


class DataflowBlock(BaseOp):
    """A CFG basic block node. The signature is that of the internal Dataflow
    graph.
    """

    op: Literal["DataflowBlock"] = "DataflowBlock"
    inputs: TypeRow = Field(default_factory=list)
    other_outputs: TypeRow = Field(default_factory=list)
    sum_rows: list[TypeRow]

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        num_cases = len(out_types)
        self.sum_rows = [[] for _ in range(num_cases)]

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.inputs = inputs
        pred = outputs[0].root
        assert isinstance(pred, stys.SumType)
        if isinstance(pred.root, stys.UnitSum):
            self.sum_rows = [[] for _ in range(pred.root.size)]
        else:
            self.sum_rows = []
            for variant in pred.root.rows:
                self.sum_rows.append(variant)
        self.other_outputs = outputs[1:]

        # Needed to avoid random '\n's in the pydantic description

    def deserialize(self) -> ops.DataflowBlock:
        return ops.DataflowBlock(
            inputs=deser_it(self.inputs),
            _sum=tys.Sum([deser_it(r) for r in self.sum_rows]),
            _other_outputs=deser_it(self.other_outputs),
        )

    model_config = ConfigDict(
        json_schema_extra={
            "description": "A CFG basic block node."
            " The signature is that of the internal Dataflow graph.",
        }
    )


class ExitBlock(BaseOp):
    """The single exit node of the CFG, has no children, stores the types of
    the CFG node output.
    """

    op: Literal["ExitBlock"] = "ExitBlock"
    cfg_outputs: TypeRow

    model_config = ConfigDict(
        json_schema_extra={
            # Needed to avoid random '\n's in the pydantic description
            "description": "The single exit node of the CFG, has no children,"
            " stores the types of the CFG node output.",
        }
    )

    def deserialize(self) -> ops.ExitBlock:
        return ops.ExitBlock(deser_it(self.cfg_outputs))


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

    def deserialize(self) -> ops.Input:
        return ops.Input(types=[t.deserialize() for t in self.types])


class Output(DataflowOp):
    """An output node. The inputs are the outputs of the function."""

    op: Literal["Output"] = "Output"
    types: TypeRow = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert len(out_types) == 0
        self.types = list(in_types)

    def deserialize(self) -> ops.Output:
        return ops.Output(deser_it(self.types))


class Call(DataflowOp):
    """Call a function directly.

    The first port is connected to the def/declare of the function being called
    directly, with a `ConstE<Graph>` edge. The signature of the remaining ports matches
    the function being called.
    """

    op: Literal["Call"] = "Call"
    func_sig: PolyFuncType
    type_args: list[stys.TypeArg]
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

    def deserialize(self) -> ops.Call:
        return ops.Call(
            self.func_sig.deserialize(),
            self.instantiation.deserialize(),
            deser_it(self.type_args),
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

    def deserialize(self) -> ops.CallIndirect:
        return ops.CallIndirect(self.signature.deserialize())


class LoadConstant(DataflowOp):
    """An operation that loads a static constant in to the local dataflow graph."""

    op: Literal["LoadConstant"] = "LoadConstant"
    datatype: Type

    def deserialize(self) -> ops.LoadConst:
        return ops.LoadConst(self.datatype.deserialize())


class LoadFunction(DataflowOp):
    """Load a static function in to the local dataflow graph."""

    op: Literal["LoadFunction"] = "LoadFunction"
    func_sig: PolyFuncType
    type_args: list[stys.TypeArg]
    instantiation: FunctionType

    def deserialize(self) -> ops.LoadFunc:
        return ops.LoadFunc(
            self.func_sig.deserialize(),
            self.instantiation.deserialize(),
            deser_it(self.type_args),
        )


class DFG(DataflowOp):
    """A simply nested dataflow graph."""

    op: Literal["DFG"] = "DFG"
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.signature = FunctionType(input=list(inputs), output=list(outputs))

    def deserialize(self) -> ops.DFG:
        sig = self.signature.deserialize()
        return ops.DFG(sig.input, sig.output)


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

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        # First port is a predicate, i.e. a sum of tuple types. We need to unpack
        # those into a list of type rows
        pred = in_types[0]
        assert isinstance(pred.root, stys.SumType)
        sum = pred.root.root
        if isinstance(sum, stys.UnitSum):
            self.sum_rows = [[] for _ in range(sum.size)]
        else:
            assert isinstance(sum, stys.GeneralSum)
            self.sum_rows = []
            for ty in sum.rows:
                self.sum_rows.append(ty)
        self.other_inputs = list(in_types[1:])
        self.outputs = list(out_types)

    def deserialize(self) -> ops.Conditional:
        return ops.Conditional(
            tys.Sum([deser_it(r) for r in self.sum_rows]),
            deser_it(self.other_inputs),
            deser_it(self.outputs),
        )


class Case(BaseOp):
    """Case ops - nodes valid inside Conditional nodes."""

    op: Literal["Case"] = "Case"
    # The signature of the contained dataflow graph.
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_child_dfg_signature(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.signature = stys.FunctionType(input=list(inputs), output=list(outputs))

    def deserialize(self) -> ops.Case:
        sig = self.signature.deserialize()
        return ops.Case(inputs=sig.input, _outputs=sig.output)


class TailLoop(DataflowOp):
    """Tail-controlled loop."""

    op: Literal["TailLoop"] = "TailLoop"
    # Types that are only input
    just_inputs: TypeRow = Field(default_factory=list)
    # Types that are only output
    just_outputs: TypeRow = Field(default_factory=list)
    # Types that are appended to both input and output:
    rest: TypeRow = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        assert in_types == out_types
        # self.just_inputs = list(in_types)
        # self.just_outputs = list(out_types)
        self.rest = list(in_types)

    def deserialize(self) -> ops.TailLoop:
        return ops.TailLoop(
            just_inputs=deser_it(self.just_inputs),
            _just_outputs=deser_it(self.just_outputs),
            rest=deser_it(self.rest),
        )


class CFG(DataflowOp):
    """A dataflow node which is defined by a child CFG."""

    op: Literal["CFG"] = "CFG"
    signature: FunctionType = Field(default_factory=FunctionType.empty)

    def insert_port_types(self, inputs: TypeRow, outputs: TypeRow) -> None:
        self.signature = FunctionType(
            input=list(inputs),
            output=list(outputs),
        )

    def deserialize(self) -> ops.CFG:
        sig = self.signature.deserialize()
        return ops.CFG(inputs=sig.input, _outputs=sig.output)


ControlFlowOp = Conditional | TailLoop | CFG


class ExtensionOp(DataflowOp):
    """A user-defined operation that can be downcasted by the extensions that define
    it.
    """

    op: Literal["Extension"] = "Extension"
    extension: ExtensionId
    name: str
    signature: stys.FunctionType = Field(default_factory=stys.FunctionType.empty)
    args: list[stys.TypeArg] = Field(default_factory=list)

    def insert_port_types(self, in_types: TypeRow, out_types: TypeRow) -> None:
        self.signature = stys.FunctionType(input=list(in_types), output=list(out_types))

    def display_name(self) -> str:
        return self.name

    def deserialize(self) -> ops.Custom:
        return ops.Custom(
            extension=self.extension,
            op_name=self.name,
            signature=self.signature.deserialize(),
            args=deser_it(self.args),
        )

    model_config = ConfigDict(
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra={
            "description": (
                "A user-defined operation that can be downcasted by the extensions that"
                " define it."
            )
        }
    )


class Tag(DataflowOp):
    """An operation that creates a tagged sum value from one of its variants."""

    op: Literal["Tag"] = "Tag"
    tag: int  # The variant to create.
    variants: list[TypeRow]  # The variants of the sum type.

    def deserialize(self) -> ops.Tag:
        return ops.Tag(
            tag=self.tag,
            sum_ty=tys.Sum([deser_it(v) for v in self.variants]),
        )


class AliasDecl(BaseOp):
    op: Literal["AliasDecl"] = "AliasDecl"
    name: str
    bound: TypeBound

    def deserialize(self) -> ops.AliasDecl:
        return ops.AliasDecl(self.name, self.bound)


class AliasDefn(BaseOp):
    op: Literal["AliasDefn"] = "AliasDefn"
    name: str
    definition: Type

    def deserialize(self) -> ops.AliasDefn:
        return ops.AliasDefn(self.name, self.definition.deserialize())


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
        | ExtensionOp
        | Tag
        | DFG
        | AliasDecl
        | AliasDefn
    ) = Field(discriminator="op")

    model_config = ConfigDict(json_schema_extra={"required": ["parent", "op"]})


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

from hugr import (  # noqa: E402 # needed to avoid circular imports
    ops,
    tys,
    val,
)
