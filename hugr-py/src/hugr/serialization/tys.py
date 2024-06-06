from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
import sys
from enum import Enum
from typing import Annotated, Any, Literal, Union, Mapping

from hugr.utils import deser_it
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


class BaseTypeParam(ABC, ConfiguredBaseModel):
    @abstractmethod
    def deserialize(self) -> _tys.TypeParam: ...


class TypeTypeParam(BaseTypeParam):
    tp: Literal["Type"] = "Type"
    b: "TypeBound"

    def deserialize(self) -> _tys.TypeTypeParam:
        return _tys.TypeTypeParam(bound=self.b)


class BoundedNatParam(BaseTypeParam):
    tp: Literal["BoundedNat"] = "BoundedNat"
    bound: int | None

    def deserialize(self) -> _tys.BoundedNatParam:
        return _tys.BoundedNatParam(upper_bound=self.bound)


class OpaqueParam(BaseTypeParam):
    tp: Literal["Opaque"] = "Opaque"
    ty: "Opaque"

    def deserialize(self) -> _tys.OpaqueParam:
        return _tys.OpaqueParam(ty=self.ty.deserialize())


class ListParam(BaseTypeParam):
    tp: Literal["List"] = "List"
    param: "TypeParam"

    def deserialize(self) -> _tys.ListParam:
        return _tys.ListParam(param=self.param.deserialize())


class TupleParam(BaseTypeParam):
    tp: Literal["Tuple"] = "Tuple"
    params: list["TypeParam"]

    def deserialize(self) -> _tys.TupleParam:
        return _tys.TupleParam(params=deser_it(self.params))


class ExtensionsParam(BaseTypeParam):
    tp: Literal["Extensions"] = "Extensions"

    def deserialize(self) -> _tys.ExtensionsParam:
        return _tys.ExtensionsParam()


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

    def deserialize(self) -> _tys.TypeParam:
        return self.root.deserialize()


# ------------------------------------------
# --------------- TypeArg ------------------
# ------------------------------------------


class BaseTypeArg(ABC, ConfiguredBaseModel):
    @abstractmethod
    def deserialize(self) -> _tys.TypeArg: ...


class TypeTypeArg(BaseTypeArg):
    tya: Literal["Type"] = "Type"
    ty: "Type"

    def deserialize(self) -> _tys.TypeTypeArg:
        return _tys.TypeTypeArg(ty=self.ty.deserialize())


class BoundedNatArg(BaseTypeArg):
    tya: Literal["BoundedNat"] = "BoundedNat"
    n: int

    def deserialize(self) -> _tys.BoundedNatArg:
        return _tys.BoundedNatArg(n=self.n)


class OpaqueArg(BaseTypeArg):
    tya: Literal["Opaque"] = "Opaque"
    typ: "Opaque"
    value: Any

    def deserialize(self) -> _tys.OpaqueArg:
        return _tys.OpaqueArg(ty=self.typ.deserialize(), value=self.value)


class SequenceArg(BaseTypeArg):
    tya: Literal["Sequence"] = "Sequence"
    elems: list["TypeArg"]

    def deserialize(self) -> _tys.SequenceArg:
        return _tys.SequenceArg(elems=deser_it(self.elems))


class ExtensionsArg(BaseTypeArg):
    tya: Literal["Extensions"] = "Extensions"
    es: ExtensionSet

    def deserialize(self) -> _tys.ExtensionsArg:
        return _tys.ExtensionsArg(extensions=self.es)


class VariableArg(BaseTypeArg):
    tya: Literal["Variable"] = "Variable"
    idx: int
    cached_decl: TypeParam

    def deserialize(self) -> _tys.VariableArg:
        return _tys.VariableArg(idx=self.idx, param=self.cached_decl.deserialize())


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

    def deserialize(self) -> _tys.TypeArg:
        return self.root.deserialize()


# --------------------------------------------
# --------------- Container ------------------
# --------------------------------------------


class BaseType(ABC, ConfiguredBaseModel):
    @abstractmethod
    def deserialize(self) -> _tys.Type: ...


class MultiContainer(BaseType):
    ty: "Type"


class Array(MultiContainer):
    """Known size array whose elements are of the same type."""

    t: Literal["Array"] = "Array"
    len: int

    def deserialize(self) -> _tys.Array:
        return _tys.Array(ty=self.ty.deserialize(), size=self.len)


class UnitSum(BaseType):
    """Simple sum type where all variants are empty tuples."""

    t: Literal["Sum"] = "Sum"
    s: Literal["Unit"] = "Unit"
    size: int

    def deserialize(self) -> _tys.UnitSum:
        return _tys.UnitSum(size=self.size)


class GeneralSum(BaseType):
    """General sum type that explicitly stores the types of the variants."""

    t: Literal["Sum"] = "Sum"
    s: Literal["General"] = "General"
    rows: list["TypeRow"]

    def deserialize(self) -> _tys.Sum:
        return _tys.Sum(variant_rows=[[t.deserialize() for t in r] for r in self.rows])


class SumType(RootModel):
    root: Annotated[Union[UnitSum, GeneralSum], Field(discriminator="s")]

    # This seems to be required for nested discriminated unions to work
    @property
    def t(self) -> str:
        return self.root.t

    model_config = ConfigDict(json_schema_extra={"required": ["s"]})

    def deserialize(self) -> _tys.Sum | _tys.UnitSum:
        return self.root.deserialize()


# ----------------------------------------------
# --------------- ClassicType ------------------
# ----------------------------------------------


class Variable(BaseType):
    """A type variable identified by an index into the array of TypeParams."""

    t: Literal["V"] = "V"
    i: int
    b: "TypeBound"

    def deserialize(self) -> _tys.Variable:
        return _tys.Variable(idx=self.i, bound=self.b)


class RowVar(BaseType):
    """A variable standing for a row of some (unknown) number of types.
    May occur only within a row; not a node input/output."""

    t: Literal["R"] = "R"
    i: int
    b: "TypeBound"

    def deserialize(self) -> _tys.RowVariable:
        return _tys.RowVariable(idx=self.i, bound=self.b)


class USize(BaseType):
    """Unsigned integer size type."""

    t: Literal["I"] = "I"

    def deserialize(self) -> _tys.USize:
        return _tys.USize()


class FunctionType(BaseType):
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

    def deserialize(self) -> _tys.FunctionType:
        return _tys.FunctionType(
            input=deser_it(self.input),
            output=deser_it(self.output),
            extension_reqs=self.extension_reqs,
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

    def deserialize(self) -> _tys.PolyFuncType:
        return _tys.PolyFuncType(
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


class Opaque(BaseType):
    """An opaque Type that can be downcasted by the extensions that define it."""

    t: Literal["Opaque"] = "Opaque"
    extension: ExtensionId
    id: str  # Unique identifier of the opaque type.
    args: list[TypeArg]
    bound: TypeBound

    def deserialize(self) -> _tys.Opaque:
        return _tys.Opaque(
            extension=self.extension,
            id=self.id,
            args=deser_it(self.args),
            bound=self.bound,
        )


class Alias(BaseType):
    """An Alias Type"""

    t: Literal["Alias"] = "Alias"
    bound: TypeBound
    name: str

    def deserialize(self) -> _tys.Alias:
        return _tys.Alias(name=self.name, bound=self.bound)


# ----------------------------------------------
# --------------- LinearType -------------------
# ----------------------------------------------


class Qubit(BaseType):
    """A qubit."""

    t: Literal["Q"] = "Q"

    def deserialize(self) -> _tys.QubitDef:
        return _tys.Qubit


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

    def deserialize(self) -> _tys.Type:
        return self.root.deserialize()


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


from hugr import _tys  # noqa: E402  # needed to avoid circular imports
