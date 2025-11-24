"""A Protocol for a composable pass.

Currently unstable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from hugr.hugr.base import Hugr


@runtime_checkable
class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr, *, inplace: bool = True) -> Hugr:
        """Call the pass to transform a HUGR.

        See :func:`_impl_pass_call` for a helper function to implement this method.
        """

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


def impl_pass_call(
    *,
    hugr: Hugr,
    inplace: bool,
    inplace_call: Callable[[Hugr], None] | None = None,
    copy_call: Callable[[Hugr], Hugr] | None = None,
) -> Hugr:
    """Helper function to implement a ComposablePass.__call__ method, given an
    inplace or copy-returning pass methods.

    At least one of the `inplace_call` or `copy_call` arguments must be provided.

    :param hugr: The Hugr to apply the pass to.
    :param inplace: Whether to apply the pass inplace.
    :param inplace_call: The method to apply the pass inplace.
    :param copy_call: The method to apply the pass by copying the Hugr.
    :return: The transformed Hugr.
    """
    if inplace and inplace_call is not None:
        inplace_call(hugr)
        return hugr
    elif inplace and copy_call is not None:
        new_hugr = copy_call(hugr)
        hugr._overwrite_hugr(new_hugr)
        return hugr
    elif not inplace and copy_call is not None:
        return copy_call(hugr)
    elif not inplace and inplace_call is not None:
        new_hugr = deepcopy(hugr)
        inplace_call(new_hugr)
        return new_hugr
    else:
        msg = "Pass must implement at least an inplace or copy run method"
        raise ValueError(msg)


@dataclass
class ComposedPass(ComposablePass):
    """A sequence of composable passes."""

    passes: list[ComposablePass]

    def __call__(self, hugr: Hugr, *, inplace: bool = True) -> Hugr:
        def apply(hugr: Hugr) -> Hugr:
            result_hugr = hugr
            for comp_pass in self.passes:
                result_hugr = comp_pass(result_hugr, inplace=False)
            return result_hugr

        def apply_inplace(hugr: Hugr) -> None:
            for comp_pass in self.passes:
                comp_pass(hugr, inplace=True)

        return impl_pass_call(
            hugr=hugr,
            inplace=inplace,
            inplace_call=apply_inplace,
            copy_call=apply,
        )

    @property
    def name(self) -> str:
        return f"Composed({ ', '.join(pass_.name for pass_ in self.passes) })"
