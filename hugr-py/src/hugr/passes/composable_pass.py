"""A Protocol for a composable pass."""

from typing import Protocol

from typing_extensions import Self

from hugr.hugr.base import Hugr


class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr) -> None: ...

    def then(self, other: Self) -> Self: ...
