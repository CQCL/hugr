"""HUGR package and pointed package interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import hugr._serialization.extension as ext_s
from hugr.ops import FuncDecl, FuncDefn, Op

if TYPE_CHECKING:
    from hugr.ext import Extension
    from hugr.hugr.base import Hugr
    from hugr.hugr.node_port import Node

__all__ = [
    "Package",
    "PackagePointer",
    "ModulePointer",
    "ExtensionPointer",
    "NodePointer",
    "FuncDeclPointer",
    "FuncDefnPointer",
]


@dataclass(frozen=True)
class Package:
    """A package of HUGR modules and extensions.


    The HUGRs may refer to the included extensions or those not included.
    """

    #: HUGR modules in the package.
    modules: list[Hugr]
    #: Extensions included in the package.
    extensions: list[Extension] = field(default_factory=list)

    def _to_serial(self) -> ext_s.Package:
        return ext_s.Package(
            modules=[m._to_serial() for m in self.modules],
            extensions=[e._to_serial() for e in self.extensions],
        )

    def to_json(self) -> str:
        return self._to_serial().model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> Package:
        """Deserialize a JSON string to a Package object.

        Args:
            json_str: The JSON string representing a Package.

        Returns:
            The deserialized Package object.
        """
        return ext_s.Package.model_validate_json(json_str).deserialize()


@dataclass(frozen=True)
class PackagePointer:
    """Classes that point to packages and their inner contents."""

    #: Package pointed to.
    package: Package


@dataclass(frozen=True)
class ModulePointer(PackagePointer):
    """Pointer to a module in a package.

    Args:
        package: Package pointed to.
        module_index: Index of the module in the package.
    """

    #: Index of the module in the package.
    module_index: int

    @property
    def module(self) -> Hugr:
        """Hugr definition of the module."""
        return self.package.modules[self.module_index]

    def to_executable_package(self) -> ExecutablePackage:
        """Create an executable package from a module containing a main function.

        Raises:
            ValueError: If the module does not contain a main function.
        """
        module = self.module
        try:
            main_node = next(
                n
                for n in module.children()
                if isinstance((f_def := module[n].op), FuncDefn)
                and f_def.f_name == "main"
            )
        except StopIteration as e:
            msg = "Module does not contain a main function"
            raise ValueError(msg) from e
        return ExecutablePackage(self.package, self.module_index, main_node)


@dataclass(frozen=True)
class ExtensionPointer(PackagePointer):
    """Pointer to an extension in a package.

    Args:
        package: Package pointed to.
        extension_index: Index of the extension in the package.
    """

    #: Index of the extension in the package.
    extension_index: int

    @property
    def extension(self) -> Extension:
        """Extension definition."""
        return self.package.extensions[self.extension_index]


OpType = TypeVar("OpType", bound=Op)


@dataclass(frozen=True)
class NodePointer(Generic[OpType], ModulePointer):
    """Pointer to a node in a module.

    Args:
        package: Package pointed to.
        module_index: Index of the module in the package.
        node: Node pointed to
    """

    #: Node pointed to.
    node: Node

    @property
    def node_op(self) -> OpType:
        """Get the operation of the node."""
        return cast(OpType, self.module[self.node].op)


@dataclass(frozen=True)
class FuncDeclPointer(NodePointer[FuncDecl]):
    """Pointer to a function declaration in a module.

    Args:
        package: Package pointed to.
        module_index: Index of the module in the package.
        node: Node containing the function declaration.
    """

    @property
    def func_decl(self) -> FuncDecl:
        """Function declaration."""
        return self.node_op


@dataclass(frozen=True)
class FuncDefnPointer(NodePointer[FuncDefn]):
    """Pointer to a function definition in a module.

    Args:
        package: Package pointed to.
        module_index: Index of the module in the package.
        node: Node containing the function definition
    """

    @property
    def func_defn(self) -> FuncDefn:
        """Function definition."""
        return self.node_op


@dataclass(frozen=True)
class ExecutablePackage(FuncDefnPointer):
    """PackagePointer with a defined entrypoint node.

    Args:
        package: Package pointed to.
        module_index: Index of the module in the package.
        node: Node containing the entry point function definition.
    """

    @property
    def entry_point_node(self) -> Node:
        """Get the entry point node of the package."""
        return self.node
