from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hugr.serialization.extension as ext_s
from hugr import tys, val
from hugr.utils import ser_it

if TYPE_CHECKING:
    from semver import Version

    from hugr.hugr import Hugr
    from hugr.tys import ExtensionId


@dataclass
class ExplicitBound:
    bound: tys.TypeBound

    def to_serial(self) -> ext_s.ExplicitBound:
        return ext_s.ExplicitBound(bound=self.bound)


@dataclass
class FromParamsBound:
    indices: list[int]

    def to_serial(self) -> ext_s.FromParamsBound:
        return ext_s.FromParamsBound(indices=self.indices)


@dataclass
class TypeDef:
    extension: ExtensionId
    name: str
    description: str
    params: list[tys.TypeParam]
    bound: ExplicitBound | FromParamsBound

    def to_serial(self) -> ext_s.TypeDef:
        return ext_s.TypeDef(
            extension=self.extension,
            name=self.name,
            description=self.description,
            params=ser_it(self.params),
            bound=ext_s.TypeDefBound(root=self.bound.to_serial()),
        )


@dataclass
class FixedHugr:
    extensions: tys.ExtensionSet
    hugr: Hugr

    def to_serial(self) -> ext_s.FixedHugr:
        return ext_s.FixedHugr(extensions=self.extensions, hugr=self.hugr)


class OpDefSig:
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
class OpDef:
    name: str
    signature: OpDefSig
    extension: ExtensionId | None = None
    description: str = ""
    misc: dict[str, Any] = field(default_factory=dict)
    lower_funcs: list[FixedHugr] = field(default_factory=list)

    def to_serial(self) -> ext_s.OpDef:
        assert self.extension is not None, "Extension must be initialised."
        return ext_s.OpDef(
            extension=self.extension,
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
class ExtensionValue:
    extension: ExtensionId
    name: str
    typed_value: val.Value

    def to_serial(self) -> ext_s.ExtensionValue:
        return ext_s.ExtensionValue(
            extension=self.extension,
            name=self.name,
            typed_value=self.typed_value.to_serial_root(),
        )


@dataclass
class Extension:
    name: ExtensionId
    version: Version
    extension_reqs: set[ExtensionId] = field(default_factory=set)
    types: dict[str, TypeDef] = field(default_factory=dict)
    values: dict[str, ExtensionValue] = field(default_factory=dict)
    operations: dict[str, OpDef] = field(default_factory=dict)

    def to_serial(self) -> ext_s.Extension:
        return ext_s.Extension(
            name=self.name,
            version=self.version,  # type: ignore[arg-type]
            extension_reqs=self.extension_reqs,
            types={k: v.to_serial() for k, v in self.types.items()},
            values={k: v.to_serial() for k, v in self.values.items()},
            operations={k: v.to_serial() for k, v in self.operations.items()},
        )

    def add_op_def(self, op_def: OpDef) -> None:
        self.operations[op_def.name] = op_def


@dataclass
class Package:
    modules: list[Hugr]
    extensions: list[Extension] = field(default_factory=list)

    def to_serial(self) -> ext_s.Package:
        return ext_s.Package(
            modules=[m.to_serial() for m in self.modules],
            extensions=[e.to_serial() for e in self.extensions],
        )
