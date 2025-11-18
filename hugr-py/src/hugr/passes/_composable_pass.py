"""A Protocol for a composable pass.

Currently unstable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hugr.hugr.base import Hugr


@runtime_checkable
class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr, inplace: bool = True) -> Hugr:
        """Call the pass to transform a HUGR."""
        if inplace:
            self._apply_inplace(hugr)
            return hugr
        else:
            return self._apply(hugr)

    # At least one of the following must be ovewritten
    def _apply(self, hugr: Hugr) -> Hugr:
        hugr = deepcopy(hugr)
        self._apply_inplace(hugr)
        return hugr

    def _apply_inplace(self, hugr: Hugr) -> None:
        new_hugr = self._apply(hugr)
        hugr._overwrite_hugr(new_hugr)

    @property
    def name(self) -> str:
        """Returns the name of the pass."""
        return self.__class__.__name__

    def then(self, other: ComposablePass) -> ComposablePass:
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

    def __call__(self, hugr: Hugr, inplace: bool = True) -> Hugr:
        """Call all of the passes in sequence."""
        for comp_pass in self.passes:
            comp_pass(hugr, inplace)
        return hugr
