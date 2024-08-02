from typing import Annotated, Any, Literal

import pydantic as pd
from pydantic_extra_types.semantic_version import SemanticVersion

from .ops import Value
from .serial_hugr import SerialHugr
from .tys import (
    ConfiguredBaseModel,
    ExtensionId,
    ExtensionSet,
    PolyFuncType,
    TypeBound,
    TypeParam,
)


class ExplicitBound(ConfiguredBaseModel):
    b: Literal["Explicit"] = "Explicit"
    bound: TypeBound


class FromParamsBound(ConfiguredBaseModel):
    b: Literal["FromParams"] = "FromParams"
    indices: list[int]


class TypeDefBound(pd.RootModel):
    root: Annotated[ExplicitBound | FromParamsBound, pd.Field(discriminator="b")]


class TypeDef(ConfiguredBaseModel):
    extension: ExtensionId
    name: str
    description: str
    params: list[TypeParam]
    bound: TypeDefBound


class ExtensionValue(ConfiguredBaseModel):
    extension: ExtensionId
    name: str
    typed_value: Value


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


class Extension(ConfiguredBaseModel):
    version: SemanticVersion
    name: ExtensionId
    extension_reqs: set[ExtensionId]
    types: dict[str, TypeDef]
    values: dict[str, ExtensionValue]
    operations: dict[str, OpDef]

    @classmethod
    def get_version(cls) -> str:
        from hugr import get_serialization_version

        return get_serialization_version()


class Package(ConfiguredBaseModel):
    modules: list[SerialHugr]
    extensions: list[Extension] = pd.Field(default_factory=list)

    @classmethod
    def get_version(cls) -> str:
        from hugr import get_serialization_version

        return get_serialization_version()
