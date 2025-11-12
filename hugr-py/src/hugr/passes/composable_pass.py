"""A Protocol for a composable pass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from typing_extensions import Self

from hugr.hugr.base import Hugr


class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr) -> None:
        """Call the pass to transform a HUGR."""
        ...

    def then(self, other: Self) -> ComposablePass:
        """Perform another composable pass after this pass."""
        # Provide a default implementation for composing passes.
        pass_list = []
        if isinstance(self, ComposedPass):
            pass_list.extend(self.passes)
        else:
            pass_list.append(self)

        if isinstance(other, ComposedPass):
            pass_list.extend(other.passes)
        else:
            pass_list.append(other)

        return ComposedPass(pass_list)


@dataclass
class ComposedPass(ComposablePass):
    """A sequence of composable passes."""

    passes: list[ComposablePass]

    def __call__(self, hugr: Hugr):
        """Call all of the passes in sequence."""
        for comp_pass in self.passes:
            comp_pass(hugr)
