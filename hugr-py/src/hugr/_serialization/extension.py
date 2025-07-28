from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic as pd
from pydantic_extra_types.semantic_version import SemanticVersion  # noqa: TCH002

from hugr.hugr.base import Hugr
from hugr.utils import deser_it

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


# --------------------------------------
# --------------- OpDef ----------------
# --------------------------------------


class FixedHugr(ConfiguredBaseModel):
    """Fixed HUGR used to define the lowering of an operation.

    Args:
        extensions: Extensions used in the HUGR.
        hugr: Base64-encoded HUGR envelope.
    """

    extensions: ExtensionSet
    hugr: str

    def deserialize(self) -> ext.FixedHugr:
        # Loading fixed HUGRs requires reading hugr-model envelopes,
        # which is not currently supported in Python.
        # TODO: Add support for loading fixed HUGRs in Python.
        # https://github.com/CQCL/hugr/issues/2287
        msg = (
            "Loading extensions with operation lowering functions is not "
            + "supported in Python"
        )
        raise NotImplementedError(msg)


class OpDef(ConfiguredBaseModel, populate_by_name=True):
    """Serializable definition for dynamically loaded operations."""

    extension: ExtensionId
    name: str  # Unique identifier of the operation.
    description: str  # Human readable description of the operation.
    misc: dict[str, Any] | None = None
    signature: PolyFuncType | None = None
    binary: bool = False
    lower_funcs: list[FixedHugr] = pd.Field(default_factory=list)

    def deserialize(self, extension: ext.Extension) -> ext.OpDef:
        signature = ext.OpDefSig(
            self.signature.deserialize() if self.signature else None,
            self.binary,
        )

        # Loading fixed HUGRs requires reading hugr-model envelopes,
        # which is not currently supported in Python.
        # We currently ignore any lower functions instead of raising an error.
        #
        # TODO: Add support for loading fixed HUGRs in Python.
        # https://github.com/CQCL/hugr/issues/2287
        lower_funcs: list[ext.FixedHugr] = []

        return extension.add_op_def(
            ext.OpDef(
                name=self.name,
                description=self.description,
                misc=self.misc or {},
                signature=signature,
                lower_funcs=lower_funcs,
            )
        )


class Extension(ConfiguredBaseModel):
    version: SemanticVersion
    name: ExtensionId
    types: dict[str, TypeDef]
    operations: dict[str, OpDef]

    @classmethod
    def get_version(cls) -> str:
        return serialization_version()

    def deserialize(self) -> ext.Extension:
        e = ext.Extension(
            version=self.version,  # type: ignore[arg-type]
            name=self.name,
        )

        for k, t in self.types.items():
            assert k == t.name, "Type name must match key"
            e.add_type_def(t.deserialize(e))

        for k, o in self.operations.items():
            assert k == o.name, "Operation name must match key"
            e.add_op_def(o.deserialize(e))

        return e


class Package(ConfiguredBaseModel):
    modules: list[SerialHugr]
    extensions: list[Extension] = pd.Field(default_factory=list)

    @classmethod
    def get_version(cls) -> str:
        return serialization_version()

    def deserialize(self) -> package.Package:
        return package.Package(
            modules=[Hugr._from_serial(m) for m in self.modules],
            extensions=[e.deserialize() for e in self.extensions],
        )


from hugr import (  # noqa: E402
    ext,
    package,
)
