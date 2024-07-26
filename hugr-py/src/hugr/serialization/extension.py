from typing import Annotated, Any, Literal

import pydantic as pd
from pydantic_extra_types.semantic_version import SemanticVersion

from hugr import get_serialisation_version

from .ops import Value
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
        return get_serialisation_version()

    @classmethod
    def _pydantic_rebuild(cls, config: pd.ConfigDict | None = None, **kwargs):
        pass
