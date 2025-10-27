"""A Protocol for a composable pass."""

from typing import Protocol, Self

from hugr.hugr.base import Hugr
from hugr.node_port import Node


class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr) -> None: ...

    def then(self, other: Self) -> Self: ...

    def with_entrypoint(self, node: Node) -> Self: ...

    @property
    def is_global(self) -> bool: ...

    @property
    def is_recursive(self) -> bool: ...
