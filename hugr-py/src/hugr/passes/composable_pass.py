"""A Protocol for a composable pass."""

from typing import Protocol
from dataclasses import dataclass

from typing_extensions import Self

from hugr.hugr.base import Hugr


class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr) -> None: ...

    def then(self, other: Self) -> Self: ...


@dataclass
class ComposedPass(ComposablePass):
   """A Sequence of composable passes."""
   passes: list[ComposablePass]

   def __call__(self, hugr: Hugr):
       for comp_pass in self.passes:
           comp_pass(hugr)
       
