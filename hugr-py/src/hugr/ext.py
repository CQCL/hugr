"""HUGR extensions and packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from semver import Version

import hugr.serialization.extension as ext_s
from hugr import tys, val
from hugr.utils import ser_it

__all__ = [
    "ExplicitBound",
    "FromParamsBound",
    "TypeDef",
    "FixedHugr",
    "OpDefSig",
    "OpDef",
    "ExtensionValue",
    "Extension",
    "Package",
    "Version",
]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hugr.hugr import Hugr
    from hugr.tys import ExtensionId


@dataclass
class ExplicitBound:  # noqa: D101
    bound: tys.TypeBound

    def to_serial(self) -> ext_s.ExplicitBound:
        return ext_s.ExplicitBound(bound=self.bound)

    def to_serial_root(self) -> ext_s.TypeDefBound:
        return ext_s.TypeDefBound(root=self.to_serial())


@dataclass
class FromParamsBound:  # noqa: D101
    indices: list[int]

    def to_serial(self) -> ext_s.FromParamsBound:
        return ext_s.FromParamsBound(indices=self.indices)

    def to_serial_root(self) -> ext_s.TypeDefBound:
        return ext_s.TypeDefBound(root=self.to_serial())


@dataclass
class TypeDef:  # noqa: D101
    name: str
    description: str
    params: list[tys.TypeParam]
    bound: ExplicitBound | FromParamsBound
    _extension: Extension | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def to_serial(self) -> ext_s.TypeDef:
        assert self._extension is not None, "Extension must be initialised."
        return ext_s.TypeDef(
            extension=self._extension.name,
            name=self.name,
            description=self.description,
            params=ser_it(self.params),
            bound=ext_s.TypeDefBound(root=self.bound.to_serial()),
        )

    def instantiate(self, args: Sequence[tys.TypeArg]) -> tys.ExtType:
        return tys.ExtType(self, list(args))


@dataclass
class FixedHugr:  # noqa: D101
    extensions: tys.ExtensionSet
    hugr: Hugr

    def to_serial(self) -> ext_s.FixedHugr:
        return ext_s.FixedHugr(extensions=self.extensions, hugr=self.hugr)


@dataclass
class OpDefSig:  # noqa: D101
    poly_func: tys.PolyFuncType | None
    binary: bool

    def __init__(
        self,
        poly_func: tys.PolyFuncType | tys.FunctionType | None,
        binary: bool = False,
    ) -> None:
        if poly_func is None and not binary:
            msg = (
                "Signature must be provided if binary"
                " signature computation is not expected."
            )
            raise ValueError(msg)
        if isinstance(poly_func, tys.FunctionType):
            poly_func = tys.PolyFuncType([], poly_func)
        self.poly_func = poly_func
        self.binary = binary


@dataclass
class OpDef:  # noqa: D101
    name: str
    signature: OpDefSig
    description: str = ""
    misc: dict[str, Any] = field(default_factory=dict)
    lower_funcs: list[FixedHugr] = field(default_factory=list, repr=False)
    _extension: Extension | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def to_serial(self) -> ext_s.OpDef:
        assert self._extension is not None, "Extension must be initialised."
        return ext_s.OpDef(
            extension=self._extension.name,
            name=self.name,
            description=self.description,
            misc=self.misc,
            signature=self.signature.poly_func.to_serial()
            if self.signature.poly_func
            else None,
            binary=self.signature.binary,
            lower_funcs=[f.to_serial() for f in self.lower_funcs],
        )


@dataclass
class ExtensionValue:  # noqa: D101
    name: str
    typed_value: val.Value
    _extension: Extension | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def to_serial(self) -> ext_s.ExtensionValue:
        assert self._extension is not None, "Extension must be initialised."
        return ext_s.ExtensionValue(
            extension=self._extension.name,
            name=self.name,
            typed_value=self.typed_value.to_serial_root(),
        )


@dataclass
class Extension:  # noqa: D101
    name: ExtensionId
    version: Version
    extension_reqs: set[ExtensionId] = field(default_factory=set)
    types: dict[str, TypeDef] = field(default_factory=dict)
    values: dict[str, ExtensionValue] = field(default_factory=dict)
    operations: dict[str, OpDef] = field(default_factory=dict)

    @dataclass
    class NotFound(Exception):
        """An object was not found in the extension."""

        name: str

    def to_serial(self) -> ext_s.Extension:
        return ext_s.Extension(
            name=self.name,
            version=self.version,  # type: ignore[arg-type]
            extension_reqs=self.extension_reqs,
            types={k: v.to_serial() for k, v in self.types.items()},
            values={k: v.to_serial() for k, v in self.values.items()},
            operations={k: v.to_serial() for k, v in self.operations.items()},
        )

    def add_op_def(self, op_def: OpDef) -> OpDef:
        op_def._extension = self
        self.operations[op_def.name] = op_def
        return self.operations[op_def.name]

    def add_type_def(self, type_def: TypeDef) -> TypeDef:
        type_def._extension = self
        self.types[type_def.name] = type_def
        return self.types[type_def.name]

    def add_extension_value(self, extension_value: ExtensionValue) -> ExtensionValue:
        extension_value._extension = self
        self.values[extension_value.name] = extension_value
        return self.values[extension_value.name]

    @dataclass
    class OperationNotFound(NotFound):
        """Operation not found in extension."""

    def get_op(self, name: str) -> OpDef:
        try:
            return self.operations[name]
        except KeyError as e:
            raise self.OperationNotFound(name) from e

    @dataclass
    class TypeNotFound(NotFound):
        """Type not found in extension."""

    def get_type(self, name: str) -> TypeDef:
        try:
            return self.types[name]
        except KeyError as e:
            raise self.TypeNotFound(name) from e

    @dataclass
    class ValueNotFound(NotFound):
        """Value not found in extension."""

    def get_value(self, name: str) -> ExtensionValue:
        try:
            return self.values[name]
        except KeyError as e:
            raise self.ValueNotFound(name) from e


@dataclass
class ExtensionRegistry:
    extensions: dict[ExtensionId, Extension] = field(default_factory=dict)

    @dataclass
    class ExtensionNotFound(Exception):
        extension_id: ExtensionId

    @dataclass
    class ExtensionExists(Exception):
        extension_id: ExtensionId

    def add_extension(self, extension: Extension) -> Extension:
        if extension.name in self.extensions:
            raise self.ExtensionExists(extension.name)
        # TODO version updates
        self.extensions[extension.name] = extension
        return self.extensions[extension.name]

    def get_extension(self, name: ExtensionId) -> Extension:
        try:
            return self.extensions[name]
        except KeyError as e:
            raise self.ExtensionNotFound(name) from e


@dataclass
class Package:  # noqa: D101
    modules: list[Hugr]
    extensions: list[Extension] = field(default_factory=list)

    def to_serial(self) -> ext_s.Package:
        return ext_s.Package(
            modules=[m.to_serial() for m in self.modules],
            extensions=[e.to_serial() for e in self.extensions],
        )
