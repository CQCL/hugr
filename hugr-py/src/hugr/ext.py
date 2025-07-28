"""HUGR extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from semver import Version

import hugr._serialization.extension as ext_s
from hugr import ops, tys
from hugr.utils import ser_it

__all__ = [
    "ExplicitBound",
    "FromParamsBound",
    "TypeDef",
    "FixedHugr",
    "OpDefSig",
    "OpDef",
    "Extension",
    "Version",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hugr.hugr import Hugr
    from hugr.tys import ExtensionId


@dataclass
class ExplicitBound:
    """An explicit type bound on an :class:`OpDef`.


    Examples:
        >>> ExplicitBound(tys.TypeBound.Copyable)
        ExplicitBound(bound=<TypeBound.Copyable: 'C'>)
    """

    bound: tys.TypeBound

    def _to_serial(self) -> ext_s.ExplicitBound:
        return ext_s.ExplicitBound(bound=self.bound)

    def _to_serial_root(self) -> ext_s.TypeDefBound:
        return ext_s.TypeDefBound(root=self._to_serial())


@dataclass
class FromParamsBound:
    """Calculate the type bound of an :class:`OpDef` from the join of its parameters at
    the given indices.


    Examples:
        >>> FromParamsBound(indices=[0, 1])
        FromParamsBound(indices=[0, 1])
    """

    indices: list[int]

    def _to_serial(self) -> ext_s.FromParamsBound:
        return ext_s.FromParamsBound(indices=self.indices)

    def _to_serial_root(self) -> ext_s.TypeDefBound:
        return ext_s.TypeDefBound(root=self._to_serial())


@dataclass
class NoParentExtension(Exception):
    """Parent extension must be set."""

    kind: str

    def __str__(self):
        return f"{self.kind} does not belong to an extension."


@dataclass(init=False)
class ExtensionObject:
    """An object associated with an :class:`Extension`."""

    _extension: Extension | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def get_extension(self) -> Extension:
        """Retrieve the extension associated with the object.

        Returns:
            The extension associated with the object.

        Raises:
            NoParentExtension: If the object is not associated with an extension.
        """
        if self._extension is None:
            msg = self.__class__.__name__
            raise NoParentExtension(msg)
        return self._extension


@dataclass
class TypeDef(ExtensionObject):
    """Type definition in an :class:`Extension`.


    Examples:
        >>> td = TypeDef(
        ...     name="MyType",
        ...     description="A type definition.",
        ...     params=[tys.TypeTypeParam(tys.TypeBound.Copyable)],
        ...     bound=FromParamsBound([0]),
        ... )
        >>> td.name
        'MyType'
    """

    #: The name of the type.
    name: str
    #: A description of the type.
    description: str
    #: The type parameters of the type if polymorphic.
    params: list[tys.TypeParam]
    #: The type bound of the type.
    bound: ExplicitBound | FromParamsBound

    def _to_serial(self) -> ext_s.TypeDef:
        return ext_s.TypeDef(
            extension=self.get_extension().name,
            name=self.name,
            description=self.description,
            params=ser_it(self.params),
            bound=ext_s.TypeDefBound(root=self.bound._to_serial()),
        )

    def instantiate(self, args: Sequence[tys.TypeArg]) -> tys.ExtType:
        """Instantiate a concrete type from this type definition.

        Args:
            args: Type arguments corresponding to the type parameters of the definition.
        """
        return tys.ExtType(self, list(args))


@dataclass
class FixedHugr:
    """A HUGR used to define lowerings of operations in an :class:`OpDef`."""

    #: Extensions used in the HUGR.
    extensions: tys.ExtensionSet
    #: HUGR defining operation lowering.
    hugr: Hugr

    def _to_serial(self) -> ext_s.FixedHugr:
        return ext_s.FixedHugr(extensions=self.extensions, hugr=self.hugr.to_str())


@dataclass
class OpDefSig:
    """Type signature of an :class:`OpDef`.

    Args:
        poly_func: The polymorphic function type of the operation.
        binary: If no static type scheme known, flag indicates a computation of the
            signature
    """

    #: The polymorphic function type of the operation (type scheme).
    poly_func: tys.PolyFuncType | None
    #: If no static type scheme known, flag indicates a computation of the signature.
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
class OpDef(ExtensionObject):
    """Operation definition in an :class:`Extension`."""

    #: The name of the operation.
    name: str
    #: The type signature of the operation.
    signature: OpDefSig
    #: A description of the operation.
    description: str = ""
    #: Miscellaneous information about the operation.
    misc: dict[str, Any] = field(default_factory=dict)
    #: Lowerings of the operation.
    lower_funcs: list[FixedHugr] = field(default_factory=list, repr=False)

    def _to_serial(self) -> ext_s.OpDef:
        return ext_s.OpDef(
            extension=self.get_extension().name,
            name=self.name,
            description=self.description,
            misc=self.misc,
            signature=self.signature.poly_func._to_serial()
            if self.signature.poly_func
            else None,
            binary=self.signature.binary,
            lower_funcs=[f._to_serial() for f in self.lower_funcs],
        )

    def qualified_name(self) -> str:
        ext_name = self._extension.name if self._extension else ""
        if ext_name:
            return f"{ext_name}.{self.name}"
        return self.name

    def instantiate(
        self,
        args: Sequence[tys.TypeArg] | None = None,
        concrete_signature: tys.FunctionType | None = None,
    ) -> ops.ExtOp:
        """Instantiate an operation from this definition.

        Args:
            args: Type arguments corresponding to the type parameters of the definition.
            concrete_signature: Concrete function type of the operation, only required
            if the operation is polymorphic.
        """
        return ops.ExtOp(self, concrete_signature, list(args or []))


T = TypeVar("T", bound=ops.RegisteredOp)


@dataclass
class Extension:
    """HUGR extension declaration."""

    #: The name of the extension.
    name: ExtensionId
    #: The version of the extension.
    version: Version
    #: Type definitions in the extension.
    types: dict[str, TypeDef] = field(default_factory=dict)
    #: Operation definitions in the extension.
    operations: dict[str, OpDef] = field(default_factory=dict)

    @dataclass
    class NotFound(Exception):
        """An object was not found in the extension."""

        name: str

    def _to_serial(self) -> ext_s.Extension:
        return ext_s.Extension(
            name=self.name,
            version=self.version,  # type: ignore[arg-type]
            types={k: v._to_serial() for k, v in self.types.items()},
            operations={k: v._to_serial() for k, v in self.operations.items()},
        )

    def to_json(self) -> str:
        """Serialize the extension to a JSON string."""
        return self._to_serial().model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> Extension:
        """Deserialize a JSON string to a Extension object.

        Args:
            json_str: The JSON string representing a Extension.

        Returns:
            The deserialized Extension object.
        """
        return ext_s.Extension.model_validate_json(json_str).deserialize()

    def add_op_def(self, op_def: OpDef) -> OpDef:
        """Add an operation definition to the extension.

        Args:
            op_def: The operation definition to add.

        Returns:
            The added operation definition, now associated with the extension.
        """
        op_def._extension = self
        self.operations[op_def.name] = op_def
        return self.operations[op_def.name]

    def add_type_def(self, type_def: TypeDef) -> TypeDef:
        """Add a type definition to the extension.

        Args:
            type_def: The type definition to add.

        Returns:
            The added type definition, now associated with the extension.
        """
        type_def._extension = self
        self.types[type_def.name] = type_def
        return self.types[type_def.name]

    @dataclass
    class OperationNotFound(NotFound):
        """Operation not found in extension."""

    def get_op(self, name: str) -> OpDef:
        """Retrieve an operation definition by name.

        Args:
            name: The name of the operation.

        Returns:
            The operation definition.

        Raises:
            OperationNotFound: If the operation is not found in the extension.
        """
        try:
            return self.operations[name]
        except KeyError as e:
            raise self.OperationNotFound(name) from e

    @dataclass
    class TypeNotFound(NotFound):
        """Type not found in extension."""

    def get_type(self, name: str) -> TypeDef:
        """Retrieve a type definition by name.

        Args:
            name: The name of the type.

        Returns:
            The type definition.

        Raises:
            TypeNotFound: If the type is not found in the extension.
        """
        try:
            return self.types[name]
        except KeyError as e:
            raise self.TypeNotFound(name) from e

    @dataclass
    class ValueNotFound(NotFound):
        """Value not found in extension."""

    T = TypeVar("T", bound=ops.RegisteredOp)

    def register_op(
        self,
        name: str | None = None,
        signature: OpDefSig | tys.PolyFuncType | tys.FunctionType | None = None,
        description: str | None = None,
        misc: dict[str, Any] | None = None,
        lower_funcs: list[FixedHugr] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Register a class as corresponding to an operation definition.

        If `name` is not provided, the class name is used.
        If `signature` is not provided, a binary signature is assumed.
        If `description` is not provided, the class docstring is used.

        See :class:`OpDef` for other parameters.
        """
        if not isinstance(signature, OpDefSig):
            binary = signature is None
            signature = OpDefSig(signature, binary)

        def _inner(cls: type[T]) -> type[T]:
            new_description = cls.__doc__ if description is None and cls.__doc__ else ""
            new_name = cls.__name__ if name is None else name
            op_def = self.add_op_def(
                OpDef(
                    new_name,
                    signature,
                    new_description,
                    misc or {},
                    lower_funcs or [],
                )
            )
            cls.const_op_def = op_def
            return cls

        return _inner


@dataclass
class ExtensionRegistry:
    """Registry of extensions."""

    #: Extensions in the registry, indexed by name.
    extensions: dict[ExtensionId, Extension] = field(default_factory=dict)

    @dataclass
    class ExtensionNotFound(Exception):
        """Extension not found in registry."""

        extension_id: ExtensionId

    @dataclass
    class ExtensionExists(Exception):
        """Extension already exists in registry."""

        extension_id: ExtensionId

    def add_extension(self, extension: Extension) -> Extension:
        """Add an extension to the registry.

        Args:
            extension: The extension to add.

        Returns:
            The added extension.

        Raises:
            ExtensionExists: If an extension with the same name already exists.
        """
        if extension.name in self.extensions:
            raise self.ExtensionExists(extension.name)
        self.extensions[extension.name] = extension
        return self.extensions[extension.name]

    def get_extension(self, name: ExtensionId) -> Extension:
        """Retrieve an extension by name.

        Args:
            name: The name of the extension.

        Returns:
            Extension in the registry.

        Raises:
            ExtensionNotFound: If the extension is not found in the registry.
        """
        try:
            return self.extensions[name]
        except KeyError as e:
            raise self.ExtensionNotFound(name) from e
