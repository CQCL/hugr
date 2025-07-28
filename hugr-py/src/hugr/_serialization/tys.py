from __future__ import annotations

import inspect
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)
from pydantic_core import PydanticCustomError

from hugr.utils import deser_it

if TYPE_CHECKING:
    from collections.abc import Mapping


def _json_custom_error_validator(
    value: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo
) -> Any:
    """Simplify the error message to avoid a gross error stemming
    from exhaustive checking of all union options.

    As suggested at
    https://docs.pydantic.dev/latest/concepts/types/#named-recursive-types


    Used to define named recursive alias types.
    """
    return handler(value)
    try:
        return handler(value)
    except ValidationError as err:
        msg = "invalid_json"
        raise PydanticCustomError(
            msg,
            "Input is not valid json",
        ) from err


ExtensionId = str
ExtensionSet = list[ExtensionId]

default_model_config = ConfigDict()


class ConfiguredBaseModel(BaseModel):
    model_config = default_model_config

    @classmethod
    def update_model_config(cls, config: ConfigDict):
        cls.model_config.update(config)


# --------------------------------------------
# --------------- TypeParam ------------------
# --------------------------------------------


class BaseTypeParam(ABC, ConfiguredBaseModel):
    @abstractmethod
    def deserialize(self) -> tys.TypeParam: ...


class TypeTypeParam(BaseTypeParam):
    tp: Literal["Type"] = "Type"
    b: TypeBound

    def deserialize(self) -> tys.TypeTypeParam:
        return tys.TypeTypeParam(bound=self.b)


class BoundedNatParam(BaseTypeParam):
    tp: Literal["BoundedNat"] = "BoundedNat"
    bound: int | None

    def deserialize(self) -> tys.BoundedNatParam:
        return tys.BoundedNatParam(upper_bound=self.bound)


class StringParam(BaseTypeParam):
    tp: Literal["String"] = "String"

    def deserialize(self) -> tys.StringParam:
        return tys.StringParam()


class ListParam(BaseTypeParam):
    tp: Literal["List"] = "List"
    param: TypeParam

    def deserialize(self) -> tys.ListParam:
        return tys.ListParam(param=self.param.deserialize())


class TupleParam(BaseTypeParam):
    tp: Literal["Tuple"] = "Tuple"
    params: list[TypeParam]

    def deserialize(self) -> tys.TupleParam:
        return tys.TupleParam(params=deser_it(self.params))


class TypeParam(RootModel):
    """A type parameter."""

    root: Annotated[
        TypeTypeParam | BoundedNatParam | StringParam | ListParam | TupleParam,
        WrapValidator(_json_custom_error_validator),
    ] = Field(discriminator="tp")

    model_config = ConfigDict(json_schema_extra={"required": ["tp"]})

    def deserialize(self) -> tys.TypeParam:
        return self.root.deserialize()


# ------------------------------------------
# --------------- TypeArg ------------------
# ------------------------------------------


class BaseTypeArg(ABC, ConfiguredBaseModel):
    @abstractmethod
    def deserialize(self) -> tys.TypeArg: ...


class TypeTypeArg(BaseTypeArg):
    tya: Literal["Type"] = "Type"
    ty: Type

    def deserialize(self) -> tys.TypeTypeArg:
        return tys.TypeTypeArg(ty=self.ty.deserialize())


class BoundedNatArg(BaseTypeArg):
    tya: Literal["BoundedNat"] = "BoundedNat"
    n: int

    def deserialize(self) -> tys.BoundedNatArg:
        return tys.BoundedNatArg(n=self.n)


class StringArg(BaseTypeArg):
    tya: Literal["String"] = "String"
    arg: str

    def deserialize(self) -> tys.StringArg:
        return tys.StringArg(value=self.arg)


class SequenceArg(BaseTypeArg):
    tya: Literal["Sequence"] = "Sequence"
    elems: list[TypeArg]

    def deserialize(self) -> tys.SequenceArg:
        return tys.SequenceArg(elems=deser_it(self.elems))


class VariableArg(BaseTypeArg):
    tya: Literal["Variable"] = "Variable"
    idx: int
    cached_decl: TypeParam

    def deserialize(self) -> tys.VariableArg:
        return tys.VariableArg(idx=self.idx, param=self.cached_decl.deserialize())


class TypeArg(RootModel):
    """A type argument."""

    root: Annotated[
        TypeTypeArg | BoundedNatArg | StringArg | SequenceArg | VariableArg,
        WrapValidator(_json_custom_error_validator),
    ] = Field(discriminator="tya")

    model_config = ConfigDict(json_schema_extra={"required": ["tya"]})

    def deserialize(self) -> tys.TypeArg:
        return self.root.deserialize()


# --------------------------------------------
# --------------- Container ------------------
# --------------------------------------------


class BaseType(ABC, ConfiguredBaseModel):
    @abstractmethod
    def deserialize(self) -> tys.Type: ...


class UnitSum(BaseType):
    """Simple sum type where all variants are empty tuples."""

    t: Literal["Sum"] = "Sum"
    s: Literal["Unit"] = "Unit"
    size: int

    def deserialize(self) -> tys.UnitSum:
        return tys.UnitSum(size=self.size)


class GeneralSum(BaseType):
    """General sum type that explicitly stores the types of the variants."""

    t: Literal["Sum"] = "Sum"
    s: Literal["General"] = "General"
    rows: list[TypeRow]

    def deserialize(self) -> tys.Sum:
        return tys.Sum(variant_rows=[[t.deserialize() for t in r] for r in self.rows])


class SumType(RootModel):
    root: Annotated[UnitSum | GeneralSum, Field(discriminator="s")]

    # This seems to be required for nested discriminated unions to work
    @property
    def t(self) -> str:
        return self.root.t

    model_config = ConfigDict(json_schema_extra={"required": ["s"]})

    def deserialize(self) -> tys.Sum | tys.UnitSum:
        return self.root.deserialize()


# ----------------------------------------------
# --------------- ClassicType ------------------
# ----------------------------------------------


class Variable(BaseType):
    """A type variable identified by an index into the array of TypeParams."""

    t: Literal["V"] = "V"
    i: int
    b: TypeBound

    def deserialize(self) -> tys.Variable:
        return tys.Variable(idx=self.i, bound=self.b)


class RowVar(BaseType):
    """A variable standing for a row of some (unknown) number of types.
    May occur only within a row; not a node input/output.
    """

    t: Literal["R"] = "R"
    i: int
    b: TypeBound

    def deserialize(self) -> tys.RowVariable:
        return tys.RowVariable(idx=self.i, bound=self.b)


class USize(BaseType):
    """Unsigned integer size type."""

    t: Literal["I"] = "I"

    def deserialize(self) -> tys.USize:
        return tys.USize()


class FunctionType(BaseType):
    """A graph encoded as a value. It contains a concrete signature and a set of
    required resources.
    """

    t: Literal["G"] = "G"

    input: TypeRow  # Value inputs of the function.
    output: TypeRow  # Value outputs of the function.

    @classmethod
    def empty(cls) -> FunctionType:
        return FunctionType(input=[], output=[])

    def deserialize(self) -> tys.FunctionType:
        return tys.FunctionType(
            input=deser_it(self.input),
            output=deser_it(self.output),
        )

    model_config = ConfigDict(
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra={
            "description": (
                "A graph encoded as a value. It contains a concrete signature and "
                "a set of required resources."
            )
        }
    )


class PolyFuncType(BaseType):
    """A polymorphic type scheme, i.e. of a FuncDecl, FuncDefn or OpDef.
    (Nodes/operations in the Hugr are not polymorphic.).
    """

    # The declared type parameters, i.e., these must be instantiated with the same
    # number of TypeArgs before the function can be called. This defines the indices
    # used for variables within the body.
    params: list[TypeParam]

    # Template for the function. May contain variables up to length of `params`
    body: FunctionType

    @classmethod
    def empty(cls) -> PolyFuncType:
        return PolyFuncType(params=[], body=FunctionType.empty())

    def deserialize(self) -> tys.PolyFuncType:
        return tys.PolyFuncType(
            params=deser_it(self.params),
            body=self.body.deserialize(),
        )

    model_config = ConfigDict(
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra={
            "description": (
                "A polymorphic type scheme, i.e. of a FuncDecl, FuncDefn or OpDef.  "
                "(Nodes/operations in the Hugr are not polymorphic.)"
            )
        }
    )


class TypeBound(Enum):
    Copyable = "C"
    Any = "A"

    @staticmethod
    def join(*bs: TypeBound) -> TypeBound:
        """Computes the least upper bound for a sequence of bounds."""
        res = TypeBound.Copyable
        for b in bs:
            if b == TypeBound.Any:
                return TypeBound.Any
            if res == TypeBound.Copyable:
                res = b
        return res

    def __str__(self) -> str:
        match self:
            case TypeBound.Copyable:
                return "Copyable"
            case TypeBound.Any:
                return "Any"


class Opaque(BaseType):
    """An opaque Type that can be downcasted by the extensions that define it."""

    t: Literal["Opaque"] = "Opaque"
    extension: ExtensionId
    id: str  # Unique identifier of the opaque type.
    args: list[TypeArg]
    bound: TypeBound

    def deserialize(self) -> tys.Opaque:
        return tys.Opaque(
            extension=self.extension,
            id=self.id,
            args=deser_it(self.args),
            bound=self.bound,
        )


class Alias(BaseType):
    """An Alias Type."""

    t: Literal["Alias"] = "Alias"
    bound: TypeBound
    name: str

    def deserialize(self) -> tys.Alias:
        return tys.Alias(name=self.name, bound=self.bound)


# ----------------------------------------------
# --------------- LinearType -------------------
# ----------------------------------------------


class Qubit(BaseType):
    """A qubit."""

    t: Literal["Q"] = "Q"

    def deserialize(self) -> tys._QubitDef:
        return tys.Qubit


class Type(RootModel):
    """A HUGR type."""

    root: Annotated[
        Qubit | Variable | RowVar | USize | FunctionType | SumType | Opaque | Alias,
        WrapValidator(_json_custom_error_validator),
        Field(discriminator="t"),
    ]

    model_config = ConfigDict(json_schema_extra={"required": ["t"]})

    def deserialize(self) -> tys.Type:
        return self.root.deserialize()


# -------------------------------------------
# --------------- TypeRow -------------------
# -------------------------------------------

TypeRow = list[Type]


# Now that all classes are defined, we need to update the ForwardRefs in all type
# annotations. We use some inspect magic to find all classes defined in this file
# and call model_rebuild()
classes = inspect.getmembers(
    sys.modules[__name__],
    lambda member: inspect.isclass(member) and member.__module__ == __name__,
)


def model_rebuild(
    classes: Mapping[str, type],
    config: ConfigDict | None = None,
    **kwargs,
):
    config = config or ConfigDict()
    for c in classes.values():
        if issubclass(c, ConfiguredBaseModel):
            c.update_model_config(config)
            c.model_rebuild(**kwargs)


model_rebuild(dict(classes))


from hugr import tys  # noqa: E402  # needed to avoid circular imports
