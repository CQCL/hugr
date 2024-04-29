import inspect
import sys
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union, Tuple
from contextlib import contextmanager

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
    try:
        return handler(value)
    except ValidationError as err:
        raise PydanticCustomError(
            "invalid_json",
            "Input is not valid json",
        ) from err


ExtensionId = str

global default_model_config
default_model_config = ConfigDict()
global current_model_config
current_model_config = default_model_config


class ConfiguredBaseModel(BaseModel):
    model_config = default_model_config

    @classmethod
    def set_model_config(cls, config: ConfigDict):
        cls.model_config = config


@contextmanager
def hugr_config(**kwargs):
    global current_model_config
    old_config = current_model_config
    current_model_config = ConfigDict(default_model_config, **kwargs)
    ConfiguredBaseModel.model_config = current_model_config
    print(f"new model_config: {current_model_config}")
    try:
        yield None
    finally:
        current_model_config = old_config
        ConfiguredBaseModel.model_config = old_config


class ExtensionSet(RootModel):
    """A set of extensions ids."""

    root: Optional[list[ExtensionId]] = Field(default=None)


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


# ------------------------------------------
# --------------- TypeArg ------------------
# ------------------------------------------


class CustomTypeArg(ConfiguredBaseModel):
    typ: None  # TODO
    value: str


class TypeTypeArg(ConfiguredBaseModel):
    tya: Literal["Type"] = "Type"
    ty: "Type"


class BoundedNatArg(ConfiguredBaseModel):
    tya: Literal["BoundedNat"] = "BoundedNat"
    n: int


class OpaqueArg(ConfiguredBaseModel):
    tya: Literal["Opaque"] = "Opaque"
    arg: CustomTypeArg


class SequenceArg(ConfiguredBaseModel):
    tya: Literal["Sequence"] = "Sequence"
    args: list["TypeArg"]


class ExtensionsArg(ConfiguredBaseModel):
    tya: Literal["Extensions"] = "Extensions"
    es: ExtensionSet


class TypeArg(RootModel):
    """A type argument."""

    root: Annotated[
        TypeTypeArg | BoundedNatArg | OpaqueArg | SequenceArg | ExtensionsArg,
        WrapValidator(_json_custom_error_validator),
    ] = Field(discriminator="tya")


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

    s: Literal["Unit"] = "Unit"
    size: int


class GeneralSum(ConfiguredBaseModel):
    """General sum type that explicitly stores the types of the variants."""

    s: Literal["General"] = "General"
    rows: list["TypeRow"]


class SumType(RootModel):
    root: Union[UnitSum, GeneralSum] = Field(discriminator="s")


class TaggedSumType(ConfiguredBaseModel):
    t: Literal["Sum"] = "Sum"
    st: SumType


# ----------------------------------------------
# --------------- ClassicType ------------------
# ----------------------------------------------


class Variable(ConfiguredBaseModel):
    """A type variable identified by an index into the array of TypeParams."""

    t: Literal["V"] = "V"
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
        return FunctionType(input=[], output=[], extension_reqs=ExtensionSet([]))

    class Config:
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra = {
            "description": (
                "A graph encoded as a value. It contains a concrete signature and "
                "a set of required resources."
            )
        }


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

    class Config:
        # Needed to avoid random '\n's in the pydantic description
        json_schema_extra = {
            "description": (
                "A polymorphic type scheme, i.e. of a FuncDecl, FuncDefn or OpDef.  "
                "(Nodes/operations in the Hugr are not polymorphic.)"
            )
        }


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
        | USize
        | FunctionType
        | Array
        | TaggedSumType
        | Opaque
        | Alias,
        WrapValidator(_json_custom_error_validator),
    ] = Field(discriminator="t")


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

    # The extensions which are associated with all the inputs and carried through
    input_extensions: ExtensionSet


# Now that all classes are defined, we need to update the ForwardRefs in all type
# annotations. We use some inspect magic to find all classes defined in this file.
classes = inspect.getmembers(
    sys.modules[__name__],
    lambda member: inspect.isclass(member) and member.__module__ == __name__,
)


def _model_rebuild(
    classes: list[Tuple[Any, type]] = classes,
    config: ConfigDict = ConfigDict(),
    **kwargs,
):
    for _, c in classes:
        if issubclass(c, ConfiguredBaseModel):
            c.set_model_config(config)
            c.model_rebuild(**kwargs)


_model_rebuild()
