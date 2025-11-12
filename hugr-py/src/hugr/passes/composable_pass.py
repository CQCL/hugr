"""A Protocol for a composable pass."""

from typing import Protocol
from dataclasses import dataclass

from typing_extensions import Self

from hugr.hugr.base import Hugr


class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr) -> None:
        """Call the pass to transform a HUGR."""
        ...

    def then(self, other: Self) -> Self:
        """Perform another composable pass after this pass."""
        ...


@dataclass
class ComposedPass(ComposablePass):
    """A sequence of composable passes."""

    passes: list[ComposablePass]

    def __call__(self, hugr: Hugr):
        """Call all of the passes in sequence."""
        for comp_pass in self.passes:
            comp_pass(hugr)
