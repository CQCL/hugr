"""HUGR extensions."""

from __future__ import annotations

import base64
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
    "ExtensionRegistry",
    "ExtensionResolutionResult",
    "UsedExtensionResolver",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from hugr.hugr import Hugr
    from hugr.tys import ExtensionId


@dataclass
class ExplicitBound:
    """An explicit type bound on an :class:`OpDef`.


    Examples:
        >>> ExplicitBound(tys.TypeBound.Copyable)
        ExplicitBound(bound=TypeBound.Copyable)
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

    def qualified_name(self) -> str:
        """Get the fully qualified name of the type definition.

        Returns the extension name prefixed to the type name if the type
        belongs to an extension (e.g., "my_extension.MyType"), otherwise
        returns just the type name.

        Returns:
            The qualified name of the type definition.
        """
        ext_name = self._extension.name if self._extension else ""
        if ext_name:
            return f"{ext_name}.{self.name}"
        return self.name

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
        hugr_64: str = base64.b64encode(self.hugr.to_bytes()).decode()
        return ext_s.FixedHugr(extensions=self.extensions, hugr=hugr_64)


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
        """Get the fully qualified name of the operation definition.

        Returns the extension name prefixed to the operation name if the operation
        belongs to an extension (e.g., "my_extension.MyOp"), otherwise
        returns just the operation name.

        Returns:
            The qualified name of the operation definition.
        """
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

        Raises:
            ValueError: if a `concrete_signature` arg is not provided to a `OpDef`
                which requires type parameters, or if the wrong number of args are
                provided for the `OpDef`.

        """
        num_args = len(args) if args else 0
        if self.signature.poly_func is not None:
            params = self.signature.poly_func.params
            if num_args > len(params):
                msg = (
                    f"Too many args: Op {self.qualified_name()} takes {len(params)}"
                    f" args, but was given {num_args}"
                )
                raise ValueError(msg)
            elif len(params) > num_args:
                msg = (
                    f"Not enough args: Op {self.qualified_name()} takes {len(params)}"
                    f" args, but was given {num_args}"
                )
                raise ValueError(msg)

        if num_args > 0 and concrete_signature is None:
            url_base = "https://quantinuum.github.io/hugr/generated"
            url = f"{url_base}/hugr.ext.OpDef.html#hugr.ext.OpDef.instantiate"
            msg = (
                f"Signature not provided:\n"
                f"  Concrete signature required for polymorphic Op "
                f"`{self.qualified_name()}`.\n"
                f"  See documentation for `OpDef.instantiate`:\n"
                f"  {url}"
            )
            raise ValueError(msg)

        return ops.ExtOp(self, concrete_signature, list(args or []))


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
    def from_json(
        cls, json_str: str, registry: ExtensionRegistry | None = None
    ) -> Extension:
        """Deserialize a JSON string to a Extension object.

        Args:
            json_str: The JSON string representing a Extension.
            registry: If present, performs extension resolution on the
                extension types referenced in the type and operation definitions
                of the deserialized Extension.

        Returns:
            The deserialized Extension object.
        """
        ext = ext_s.Extension.model_validate_json(json_str).deserialize()
        if registry is not None:
            ext._resolve_used_extensions(registry)
        return ext

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

    def _resolve_used_extensions(
        self, registry: ExtensionRegistry | None = None
    ) -> ExtensionResolutionResult:
        """Resolve and collect dependencies throughout this extension definition."""
        if registry is not None and self.name not in registry:
            registry.register(self)

        resolver = UsedExtensionResolver()
        result = ExtensionResolutionResult()
        for type_def in self.types.values():
            type_def.params = [
                param._resolve_used_extensions(resolver, registry)
                for param in type_def.params
            ]

        for op_def in self.operations.values():
            poly_func = op_def.signature.poly_func
            if poly_func is None:
                continue
            resolved_poly_func = poly_func._resolve_used_extensions(resolver, registry)
            assert isinstance(resolved_poly_func, tys.PolyFuncType)
            op_def.signature.poly_func = resolved_poly_func

            for lower_func in op_def.lower_funcs:
                lower_result = lower_func.hugr.used_extensions(registry)
                result.extend(lower_result)

        result.extend(resolver.result())

        return result

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

        T = TypeVar("T", bound=ops.RegisteredOp)

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
class ExtensionVersions:
    """Set of versions of an extension.

    Older versions of an extension may be kept for backwards compatibility.
    """

    _id: ExtensionId
    _latest_version: Version
    _exts: dict[Version, Extension]

    def __init__(self, latest: Extension) -> None:
        self._id = latest.name
        self._latest_version = latest.version
        self._exts = {latest.version: latest}

    @property
    def id(self) -> ExtensionId:
        """Get the ID of the extension."""
        return self._id

    @property
    def latest(self) -> Extension:
        """Get latest version of the extension."""
        return self._exts[self._latest_version]

    @property
    def versions(self) -> frozenset[Version]:
        """Get all versions of the extension."""
        return frozenset(self._exts.keys())

    def _coalesce_single(self, resolve_on: Version) -> None:
        compatible_versions = [
            version
            for version in self._exts
            if _same_compatibility_group(version, resolve_on)
        ]
        if len(compatible_versions) > 0:
            latest_compatible = max(compatible_versions)
            for version in compatible_versions:
                if version != latest_compatible:
                    del self._exts[version]

    def coalesce(self) -> None:
        """Ensures the latest version is up to date and any semver compatible versions
        are coalesced, with only the latest of that group retained.

        This is done automatically by `add`, but not by `add_lazy`.
        """
        self._latest_version = max(self._exts)

        for version in list(self._exts.keys()):
            # body mutates the dictionary, so check whether the version is still present
            if version in self._exts:
                self._coalesce_single(version)

    def add(self, extension: Extension) -> None:
        """Add an extension to the set, coalescing compatible older versions."""
        if extension.name != self._id:
            msg = f"Extension {extension.name} has a different name than {self._id}"
            raise ValueError(msg)

        self.add_lazy(extension)
        self._latest_version = max(self._exts)
        self._coalesce_single(extension.version)

    def add_lazy(self, extension: Extension) -> None:
        """Add an extension to the set, NOT coalescing compatible older versions.
        Use `coalesce` to subsequently coalesce all compatible versions.
        """
        if extension.name != self._id:
            msg = f"Extension {extension.name} has a different name than {self._id}"
            raise ValueError(msg)
        self._exts[extension.version] = extension

    def get_compatible(self, version: Version) -> Extension:
        """Get the highest registered extension compatible with ``version``."""
        compatible = [
            extension
            for ext_version, extension in self._exts.items()
            if _semver_compatible(ext_version, version)
        ]
        if not compatible:
            raise KeyError(version)
        return max(compatible, key=lambda extension: extension.version)

    def __contains__(self, version: Version) -> bool:
        """Check if a version is in the set."""
        return version in self._exts

    def __getitem__(self, version: Version) -> Extension:
        """Get an extension by version."""
        return self._exts[version]

    def __iter__(self) -> Iterator[Extension]:
        """Iterate over the extensions in the set.

        Yields extensions.
        """
        return iter(self._exts.values())

    def __len__(self) -> int:
        """Get the number of extensions in the set."""
        return len(self._exts)


@dataclass
class ExtensionRegistry:
    """Registry of extensions."""

    #: Set of different versions of each extension in the registry.
    versioned_extensions: dict[ExtensionId, ExtensionVersions] = field(
        default_factory=dict
    )

    @property
    def extensions(self) -> Iterator[Extension]:
        """Get the latest version of each extension in the registry.

        To get all registered versions of all extensions, use :meth:`all_extensions`.
        """
        for versions in self.versioned_extensions.values():
            yield versions.latest

    @property
    def all_extensions(self) -> Iterator[Extension]:
        """Get all extensions in the registry.

        This may contain different versions of the same extension.
        To get only the latest version of each extension, use :meth:`extensions`.
        """
        for versions in self.versioned_extensions.values():
            yield from versions

    @dataclass
    class ExtensionNotFound(Exception):
        """Extension not found in registry."""

        extension_id: ExtensionId

    @classmethod
    def from_extensions(cls, extensions: Iterable[Extension]) -> ExtensionRegistry:
        """Create an extension registry from a list of extensions."""
        self = cls()
        for extension in extensions:
            self.register(extension)
        return self

    def ids(self) -> set[ExtensionId]:
        """Get the set of extension IDs in the registry.

        Returns:
            Set of extension IDs.
        """
        return set(self.versioned_extensions.keys())

    def register(self, extension: Extension) -> Extension:
        """Add an extension to the registry.

        If a different non semver-compatible version of the same extension already
        exists, keeps both.
        If a semver-compatible version of the same extension already exists, keeps the
        latest one.

        Args:
            extension: The extension to add.

        Returns:
            The latest extension in the registry compatible with the given one.

        Raises:
            ExtensionExists: If an extension with the same name already exists.
        """
        if extension.name not in self.versioned_extensions:
            self.versioned_extensions[extension.name] = ExtensionVersions(extension)
        else:
            self.versioned_extensions[extension.name].add(extension)
        return self.versioned_extensions[extension.name].get_compatible(
            extension.version
        )

    def get_extension(
        self, name: ExtensionId, version: Version | None = None
    ) -> Extension:
        """Retrieve an extension by name.

        Args:
            name: The name of the extension.
            version: Optional serialized version requirement. If set, the
                highest compatible registered version is returned.

        Returns:
            Extension in the registry.

        Raises:
            ExtensionNotFound: If the extension is not found in the registry.
        """
        try:
            versions = self.versioned_extensions[name]
        except KeyError as e:
            raise self.ExtensionNotFound(name) from e
        if version is None:
            return versions.latest
        try:
            return versions.get_compatible(version)
        except KeyError as e:
            raise self.ExtensionNotFound(name) from e

    def extend(self, other: ExtensionRegistry) -> None:
        """Add a registry of extensions to this registry.

        If an extension with the same name already exists, the one with the
        higher version is kept.

        Args:
            other: The extension registry to add.
        """
        for ext in other.all_extensions:
            self.register(ext)

    def __str__(self) -> str:
        return "ExtensionRegistry(" + ", ".join(self.ids()) + ")"

    def __contains__(self, name: ExtensionId) -> bool:
        return name in self.versioned_extensions


@dataclass
class UsedExtensionResolver:
    """A stateful helper class for resolving used extensions, enabling to ensure that
    certain extensions are present and unresolved operations are added efficiently.
    """

    _used_extensions: dict[ExtensionId, ExtensionVersions] = field(default_factory=dict)
    _unresolved_extensions: set[ExtensionId] = field(default_factory=set)
    _unresolved_ops: dict[tuple[tys.ExtensionId, str], ops.Custom] = field(
        default_factory=dict
    )
    _unresolved_types: dict[tuple[tys.ExtensionId, str], tys.Opaque] = field(
        default_factory=dict
    )

    def register(self, extension: Extension) -> None:
        if extension.name not in self._used_extensions:
            self._used_extensions[extension.name] = ExtensionVersions(extension)
        else:
            self._used_extensions[extension.name].add_lazy(extension)

    def ensure_unresolved(self, ext_id: ExtensionId) -> None:
        self._unresolved_extensions.add(ext_id)

    def ensure_unresolved_op(
        self, ext_id: ExtensionId, type_id: str, op: ops.Custom
    ) -> None:
        if (ext_id, type_id) not in self._unresolved_ops:
            self._unresolved_ops[(ext_id, type_id)] = op

    def ensure_unresolved_type(
        self, ext_id: ExtensionId, type_id: str, opaque_type: tys.Opaque
    ) -> None:
        if (ext_id, type_id) not in self._unresolved_types:
            self._unresolved_types[(ext_id, type_id)] = opaque_type

    def extend_with_result(self, result: ExtensionResolutionResult) -> None:
        """Add the values from another result to this resolver.

        Args:
            result: The result of resolving extensions to add.
        """
        for ext in result.used_extensions.all_extensions:
            self.register(ext)
        self._unresolved_extensions.update(result.unresolved_extensions)
        self._unresolved_ops.update(result.unresolved_ops)
        self._unresolved_types.update(result.unresolved_types)

    def result(self) -> ExtensionResolutionResult:
        for versions in self._used_extensions.values():
            versions.coalesce()
        return ExtensionResolutionResult(
            ExtensionRegistry(self._used_extensions),
            self._unresolved_extensions,
            self._unresolved_ops,
            self._unresolved_types,
        )


@dataclass
class ExtensionResolutionResult:
    """Result of resolving extensions in a HUGR.

    Args:
        used_extensions: The extensions used by the HUGR.
        unresolved_extensions: A set of extension IDs referenced in the HUGR but
            not found in the given registry.
        unresolved_ops: The Custom operations that could not be resolved to an
            ExtOp because the operation was not found in the registry.
            Indexed by (extension ID, operation name).
        unresolved_types: The Opaque types that could not be resolved to an
            ExtType because the type was not found in the registry.
            Indexed by (extension ID, type name).
    """

    used_extensions: ExtensionRegistry = field(default_factory=ExtensionRegistry)
    unresolved_extensions: set[ExtensionId] = field(default_factory=set)
    unresolved_ops: dict[tuple[tys.ExtensionId, str], ops.Custom] = field(
        default_factory=dict
    )
    unresolved_types: dict[tuple[tys.ExtensionId, str], tys.Opaque] = field(
        default_factory=dict
    )

    def ids(self) -> set[ExtensionId]:
        """Get the set of used extension IDs.

        This includes both resolved and unresolved extensions referenced in the
        HUGR.
        """
        return self.used_extensions.ids() | self.unresolved_extensions

    def extend(self, other: ExtensionResolutionResult) -> None:
        """Add the extensions from another result to this result.

        Args:
            other: The result of resolving extensions to add.
        """
        self.used_extensions.extend(other.used_extensions)
        self.unresolved_extensions.update(other.unresolved_extensions)
        self.unresolved_ops.update(other.unresolved_ops)
        self.unresolved_types.update(other.unresolved_types)

    def _extend_with_transitive_ops(
        self, registry: ExtensionRegistry | None = None
    ) -> None:
        """Extend the set of extensions with transitive dependencies required by
        the OpDefs in each extension definition.
        """
        queue: list[Extension] = list(self.used_extensions.all_extensions)

        while queue:
            ext = queue.pop()
            op_result = ext._resolve_used_extensions(registry)

            self.unresolved_extensions.update(op_result.unresolved_extensions)

            for new_ext in op_result.used_extensions.extensions:
                if new_ext.name not in self.used_extensions:
                    self.used_extensions.register(new_ext)
                    queue.append(new_ext)


def _semver_compatible(candidate: Version, requested: Version) -> bool:
    """Return whether ``candidate`` can satisfy a request for ``requested``."""
    # python-semver treats all 0.x releases as incompatible unless they are an
    # exact match. For extension migration we follow caret-style semver
    # compatibility, where 0.minor.patch releases may update the patch version.
    if requested.major == 0 and requested.minor > 0:
        return (
            candidate >= requested
            and candidate.major == requested.major
            and candidate.minor == requested.minor
        )
    return requested.is_compatible(candidate)


def _same_compatibility_group(left: Version, right: Version) -> bool:
    """Return whether two versions are in the same semver-compatible group."""
    return _semver_compatible(left, right) or _semver_compatible(right, left)
