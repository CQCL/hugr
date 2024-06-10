import inspect
import sys
from enum import Enum
from typing import Annotated, Any, Literal, Union, Mapping

from pydantic import (
    BaseModel,
    Field,
    RootModel,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    ConfigDict,
)
from pydantic_core import PydanticCustomError


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
        raise PydanticCustomError(
            "invalid_json",
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


class TypeTypeParam(ConfiguredBaseModel):
    tp: Literal["Type"] = "Type"
    b: "TypeBound"


class BoundedNatParam(ConfiguredBaseModel):
    tp: Literal["BoundedNat"] = "BoundedNat"
    bound: int | None


class OpaqueParam(ConfiguredBaseModel):
    tp: Literal["Opaque"] = "Opaque"
    ty: "Opaque"


class ListParam(ConfiguredBaseModel):
    tp: Literal["List"] = "List"
    param: "TypeParam"


class TupleParam(ConfiguredBaseModel):
    tp: Literal["Tuple"] = "Tuple"
    params: list["TypeParam"]


class ExtensionsParam(ConfiguredBaseModel):
    tp: Literal["Extensions"] = "Extensions"


class TypeParam(RootModel):
    """A type parameter."""

    root: Annotated[
        TypeTypeParam
        | BoundedNatParam
        | OpaqueParam
        | ListParam
        | TupleParam
        | ExtensionsParam,
        WrapValidator(_json_custom_error_validator),
    ] = Field(discriminator="tp")

    model_config = ConfigDict(json_schema_extra={"required": ["tp"]})


# ------------------------------------------
# --------------- TypeArg ------------------
# ------------------------------------------


class TypeTypeArg(ConfiguredBaseModel):
    tya: Literal["Type"] = "Type"
    ty: "Type"


class BoundedNatArg(ConfiguredBaseModel):
    tya: Literal["BoundedNat"] = "BoundedNat"
    n: int


class OpaqueArg(ConfiguredBaseModel):
    tya: Literal["Opaque"] = "Opaque"
    typ: "Opaque"
    value: Any


class SequenceArg(ConfiguredBaseModel):
    tya: Literal["Sequence"] = "Sequence"
    elems: list["TypeArg"]


class ExtensionsArg(ConfiguredBaseModel):
    tya: Literal["Extensions"] = "Extensions"
    es: ExtensionSet


class VariableArg(BaseModel):
    tya: Literal["Variable"] = "Variable"
    idx: int
    cached_decl: TypeParam


class TypeArg(RootModel):
    """A type argument."""

    root: Annotated[
        TypeTypeArg
        | BoundedNatArg
        | OpaqueArg
        | SequenceArg
        | ExtensionsArg
        | VariableArg,
        WrapValidator(_json_custom_error_validator),
    ] = Field(discriminator="tya")

    model_config = ConfigDict(json_schema_extra={"required": ["tya"]})


# --------------------------------------------
# --------------- Container ------------------
# --------------------------------------------


class MultiContainer(ConfiguredBaseModel):
    ty: "Type"


class Array(MultiContainer):
    """Known size array whose elements are of the same type."""

    t: Literal["Array"] = "Array"
    len: int


class UnitSum(ConfiguredBaseModel):
    """Simple sum type where all variants are empty tuples."""

    t: Literal["Sum"] = "Sum"
    s: Literal["Unit"] = "Unit"
    size: int


class GeneralSum(ConfiguredBaseModel):
    """General sum type that explicitly stores the types of the variants."""

    t: Literal["Sum"] = "Sum"
    s: Literal["General"] = "General"
    rows: list["TypeRow"]


class SumType(RootModel):
    root: Annotated[Union[UnitSum, GeneralSum], Field(discriminator="s")]

    # This seems to be required for nested discriminated unions to work
    @property
    def t(self) -> str:
        return self.root.t

    model_config = ConfigDict(json_schema_extra={"required": ["s"]})


# ----------------------------------------------
# --------------- ClassicType ------------------
# ----------------------------------------------


class Variable(ConfiguredBaseModel):
    """A type variable identified by an index into the array of TypeParams."""

    t: Literal["V"] = "V"
    i: int
    b: "TypeBound"


class RowVar(ConfiguredBaseModel):
    """A variable standing for a row of some (unknown) number of types.
    May occur only within a row; not a node input/output."""

    t: Literal["R"] = "R"
    i: int
    b: "TypeBound"


class USize(ConfiguredBaseModel):
    """Unsigned integer size type."""

    t: Literal["I"] = "I"


class FunctionType(ConfiguredBaseModel):
    """A graph encoded as a value. It contains a concrete signature and a set of
    required resources."""

    t: Literal["G"] = "G"

    input: "TypeRow"  # Value inputs of the function.
    output: "TypeRow"  # Value outputs of the function.
    # The extension requirements which are added by the operation
    extension_reqs: ExtensionSet = Field(default_factory=ExtensionSet)

    @classmethod
    def empty(cls) -> "FunctionType":
        return FunctionType(input=[], output=[], extension_reqs=[])

    model_config = ConfigDict(
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra={
            "description": (
                "A graph encoded as a value. It contains a concrete signature and "
                "a set of required resources."
            )
        }
    )


class PolyFuncType(ConfiguredBaseModel):
    """A polymorphic type scheme, i.e. of a FuncDecl, FuncDefn or OpDef.
    (Nodes/operations in the Hugr are not polymorphic.)"""

    # The declared type parameters, i.e., these must be instantiated with the same
    # number of TypeArgs before the function can be called. This defines the indices
    # used for variables within the body.
    params: list[TypeParam]

    # Template for the function. May contain variables up to length of `params`
    body: FunctionType

    @classmethod
    def empty(cls) -> "PolyFuncType":
        return PolyFuncType(params=[], body=FunctionType.empty())

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
    Eq = "E"
    Copyable = "C"
    Any = "A"

    @staticmethod
    def join(*bs: "TypeBound") -> "TypeBound":
        """Computes the least upper bound for a sequence of bounds."""
        res = TypeBound.Eq
        for b in bs:
            if b == TypeBound.Any:
                return TypeBound.Any
            if res == TypeBound.Eq:
                res = b
        return res


class Opaque(ConfiguredBaseModel):
    """An opaque Type that can be downcasted by the extensions that define it."""

    t: Literal["Opaque"] = "Opaque"
    extension: ExtensionId
    id: str  # Unique identifier of the opaque type.
    args: list[TypeArg]
    bound: TypeBound


class Alias(ConfiguredBaseModel):
    """An Alias Type"""

    t: Literal["Alias"] = "Alias"
    bound: TypeBound
    name: str


# ----------------------------------------------
# --------------- LinearType -------------------
# ----------------------------------------------


class Qubit(ConfiguredBaseModel):
    """A qubit."""

    t: Literal["Q"] = "Q"


class Type(RootModel):
    """A HUGR type."""

    root: Annotated[
        Qubit
        | Variable
        | RowVar
        | USize
        | FunctionType
        | Array
        | SumType
        | Opaque
        | Alias,
        WrapValidator(_json_custom_error_validator),
        Field(discriminator="t"),
    ]

    model_config = ConfigDict(json_schema_extra={"required": ["t"]})


# -------------------------------------------
# --------------- TypeRow -------------------
# -------------------------------------------

TypeRow = list[Type]


# -------------------------------------------
# --------------- Signature -----------------
# -------------------------------------------


class Signature(ConfiguredBaseModel):
    """Describes the edges required to/from a node.

    This includes both the concept of "signature" in the spec, and also the target
    (value) of a call (constant).
    """

    signature: "PolyFuncType"  # The underlying signature

    # ALAN is it worth keeping this now/still? (Also, surprised this is "Poly"...??)


# Now that all classes are defined, we need to update the ForwardRefs in all type
# annotations. We use some inspect magic to find all classes defined in this file
# and call model_rebuild()
classes = inspect.getmembers(
    sys.modules[__name__],
    lambda member: inspect.isclass(member) and member.__module__ == __name__,
)


def model_rebuild(
    classes: Mapping[str, type],
    config: ConfigDict = ConfigDict(),
    **kwargs,
):
    for c in classes.values():
        if issubclass(c, ConfiguredBaseModel):
            c.update_model_config(config)
            c.model_rebuild(**kwargs)


model_rebuild(dict(classes))
