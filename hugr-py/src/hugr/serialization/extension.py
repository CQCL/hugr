from typing import Annotated, Any, Literal

import pydantic as pd
from pydantic_extra_types.semantic_version import SemanticVersion

from .ops import Value
from .tys import ExtensionId, ExtensionSet, PolyFuncType, TypeBound, TypeParam


class ExplicitBound(pd.BaseModel):
    b: Literal["Explicit"] = "Explicit"
    bound: TypeBound


class FromParamsBound(pd.BaseModel):
    b: Literal["FromParams"] = "FromParams"
    indices: list[int]


class TypeDefBound(pd.RootModel):
    root: Annotated[ExplicitBound | FromParamsBound, pd.Field(discriminator="b")]


class TypeDef(pd.BaseModel):
    extension: ExtensionId
    name: str
    description: str
    params: list[TypeParam]
    bound: TypeDefBound


class ExtensionValue(pd.BaseModel):
    extension: ExtensionId
    name: str
    typed_value: Value


# --------------------------------------
# --------------- OpDef ----------------
# --------------------------------------


class FixedHugr(pd.BaseModel):
    extensions: ExtensionSet
    hugr: Any


class OpDef(pd.BaseModel, populate_by_name=True):
    """Serializable definition for dynamically loaded operations."""

    extension: ExtensionId
    name: str  # Unique identifier of the operation.
    description: str  # Human readable description of the operation.
    misc: dict[str, Any] | None = None
    signature: PolyFuncType | None = None
    lower_funcs: list[FixedHugr]


class Extension(pd.BaseModel):
    # TODO schema version
    version: SemanticVersion
    name: ExtensionId
    extension_reqs: set[ExtensionId]
    types: dict[str, TypeDef]
    values: dict[str, ExtensionValue]
    operations: dict[str, OpDef]
