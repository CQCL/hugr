"""A Protocol for a composable pass."""

from typing import Protocol, Self

from hugr.hugr.base import Hugr
from hugr.node_port import Node
from hugr.ops import Op


class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr) -> None: ...

    def then(self, other: Self) -> Self: ...

    def with_entrypoint(self, node: Node) -> Self: ...

    @classmethod
    def from_dict(cls, dictionary: dict) -> Self: ...

    def to_dict(self) -> dict: ...

    @property
    def supported_ops(self) -> set[Op]: ...

    @property
    def is_global(self) -> bool: ...

    @property
    def is_recursive(self) -> bool: ...
