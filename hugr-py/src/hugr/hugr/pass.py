"""A Protocol for a composable pass."""

from typing import Protocol

from typing_extensions import Self

from hugr.hugr.base import Hugr
from hugr.hugr.node_port import Node
from hugr.ops import Op


class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr) -> None: ...

    def then(self, other: Self) -> Self: ...

    def with_entrypoint(self, node: Node) -> Self: ...

    @property
    def is_global(self) -> bool: ...

    @property
    def is_recursive(self) -> bool: ...
