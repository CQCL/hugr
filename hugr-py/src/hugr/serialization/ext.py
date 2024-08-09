from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic as pd
from pydantic_extra_types.semantic_version import SemanticVersion  # noqa: TCH002

from hugr.utils import deser_it

from .ops import Value
from .serial_hugr import SerialHugr, serialization_version
from .tys import (
    ConfiguredBaseModel,
    ExtensionId,
    ExtensionSet,
    PolyFuncType,
    TypeBound,
    TypeParam,
)

if TYPE_CHECKING:
    from .ops import Value
    from .serial_hugr import SerialHugr


class ExplicitBound(ConfiguredBaseModel):
    b: Literal["Explicit"] = "Explicit"
    bound: TypeBound

    def deserialize(self) -> ext.ExplicitBound:
        return ext.ExplicitBound(bound=self.bound)


class FromParamsBound(ConfiguredBaseModel):
    b: Literal["FromParams"] = "FromParams"
    indices: list[int]

    def deserialize(self) -> ext.FromParamsBound:
        return ext.FromParamsBound(indices=self.indices)


class TypeDefBound(pd.RootModel):
    root: Annotated[ExplicitBound | FromParamsBound, pd.Field(discriminator="b")]


class TypeDef(ConfiguredBaseModel):
    extension: ExtensionId
    name: str
    description: str
    params: list[TypeParam]
    bound: TypeDefBound

    def deserialize(self, extension: ext.Extension) -> ext.TypeDef:
        return extension.add_type_def(
            ext.TypeDef(
                name=self.name,
                description=self.description,
                params=deser_it(self.params),
                bound=self.bound.root.deserialize(),
            )
        )


class ExtensionValue(ConfiguredBaseModel):
    extension: ExtensionId
    name: str
    typed_value: Value

    def deserialize(self) -> ext.ExtensionValue:
        return ext.ExtensionValue(
            extension=self.extension,
            name=self.name,
            typed_value=self.typed_value.deserialize(),
        )


# --------------------------------------
# --------------- OpDef ----------------
# --------------------------------------


class FixedHugr(ConfiguredBaseModel):
    extensions: ExtensionSet
    hugr: Any

    def deserialize(self) -> ext.FixedHugr:
        return ext.FixedHugr(extensions=self.extensions, hugr=self.hugr)


class OpDef(ConfiguredBaseModel, populate_by_name=True):
    """Serializable definition for dynamically loaded operations."""

    extension: ExtensionId
    name: str  # Unique identifier of the operation.
    description: str  # Human readable description of the operation.
    misc: dict[str, Any] | None = None
    signature: PolyFuncType | None = None
    binary: bool = False
    lower_funcs: list[FixedHugr]

    def deserialize(self) -> ext.OpDef:
        return ext.OpDef(
            extension=self.extension,
            name=self.name,
            description=self.description,
            misc=self.misc or {},
            signature=ext.OpDefSig(
                self.signature.deserialize() if self.signature else None, self.binary
            ),
            lower_funcs=[f.deserialize() for f in self.lower_funcs],
        )


class Extension(ConfiguredBaseModel):
    version: SemanticVersion
    name: ExtensionId
    extension_reqs: set[ExtensionId]
    types: dict[str, TypeDef]
    values: dict[str, ExtensionValue]
    operations: dict[str, OpDef]

    @classmethod
    def get_version(cls) -> str:
        return serialization_version()

    def deserialize(self) -> ext.Extension:
        e = ext.Extension(
            version=self.version,  # type: ignore[arg-type]
            name=self.name,
            extension_reqs=self.extension_reqs,
            # types={k: v.deserialize() for k, v in self.types.items()},
            values={k: v.deserialize() for k, v in self.values.items()},
            operations={k: v.deserialize() for k, v in self.operations.items()},
        )

        for v in self.types.values():
            e.add_type_def(v.deserialize(e))

        return e


class Package(ConfiguredBaseModel):
    modules: list[SerialHugr]
    extensions: list[Extension] = pd.Field(default_factory=list)

    @classmethod
    def get_version(cls) -> str:
        return serialization_version()


from hugr import ext  # noqa: E402
