"""A Protocol for a composable pass."""

from dataclasses import dataclass
from typing import Protocol

from typing_extensions import Self

from hugr.hugr.base import Hugr


class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr) -> None:
        """Call the pass to transform a HUGR."""
        ...

    def then(self, other: Self) -> Self:
        """Perform another composable pass after this pass."""
        pass_list = [] 
        if isinstance(self, ComposedPass) or isinstance(other, ComposedPass):
            if isinstance(self, ComposedPass):
                pass_list.append(self.passes)
            elif isinstance(other, ComposedPass):
                pass_list.append(other.passes)
        else:
            return ComposedPass([self, other])



@dataclass
class ComposedPass(ComposablePass):
    """A sequence of composable passes."""

    passes: list[ComposablePass]

    def __call__(self, hugr: Hugr):
        """Call all of the passes in sequence."""
        for comp_pass in self.passes:
            comp_pass(hugr)
