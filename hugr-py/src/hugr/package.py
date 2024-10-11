"""HUGR package and pointed package interfaces."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

import hugr._serialization.extension as ext_s
from hugr.ext import Extension
from hugr.hugr.base import Hugr
from hugr.hugr.node_port import Node
from hugr.ops import FuncDecl, FuncDefn, Op

__all__ = [
    "Package",
    "PackagePointer",
    "ModulePointer",
    "ExtensionPointer",
    "NodePointer",
    "FuncDeclPointer",
    "FuncDefnPointer",
]


@dataclass
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


@dataclass
class PackagePointer:
    """Classes that point to packages and their inner contents."""

    package: Package

    def get_package(self) -> Package:
        """Get the package pointed to."""
        return self.package


@dataclass
class ModulePointer(PackagePointer):
    """Pointer to a module in a package."""

    module_index: int

    def module(self) -> Hugr:
        """Hugr definition of the module."""
        return self.package.modules[self.module_index]

    def to_executable_package(self) -> "ExecutablePackage":
        """Create an executable package from a module containing a main function.

        Raises:
            StopIteration: If the module does not contain a main function.
        """
        module = self.module()
        main_node = next(
            n
            for n in module.children()
            if isinstance((f_def := module[n].op), FuncDefn) and f_def.f_name == "main"
        )

        return ExecutablePackage(self.package, self.module_index, main_node)


@dataclass
class ExtensionPointer(PackagePointer):
    """Pointer to an extension in a package."""

    extension_index: int

    def extension(self) -> Extension:
        """Extension definition."""
        return self.package.extensions[self.extension_index]


OpType = TypeVar("OpType", bound=Op)


@dataclass
class NodePointer(Generic[OpType], ModulePointer):
    """Pointer to a node in a module."""

    node: Node

    def node_op(self) -> OpType:
        """Get the operation of the node."""
        return cast(OpType, self.module()[self.node].op)


@dataclass
class FuncDeclPointer(NodePointer[FuncDecl]):
    """Pointer to a function declaration in a module."""

    def func_decl(self) -> FuncDecl:
        """Function declaration."""
        return self.node_op()


@dataclass
class FuncDefnPointer(NodePointer[FuncDefn]):
    """Pointer to a function definition in a module."""

    def func_defn(self) -> FuncDefn:
        """Function definition."""
        return self.node_op()


@dataclass
class ExecutablePackage(FuncDefnPointer):
    def entry_point_node(self) -> Node:
        """Get the entry point node of the package."""
        return self.node
