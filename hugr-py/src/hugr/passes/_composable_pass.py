"""A Protocol for a composable pass.

Currently unstable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from hugr.hugr.base import Hugr


# Type alias for a pass name
PassName = str


@runtime_checkable
class ComposablePass(Protocol):
    """A Protocol which represents a composable Hugr transformation."""

    def __call__(self, hugr: Hugr, *, inplace: bool = True) -> Hugr:
        """Call the pass to transform a HUGR, returning a Hugr."""
        return self.run(hugr, inplace=inplace).hugr

    def run(self, hugr: Hugr, *, inplace: bool = True) -> PassResult:
        """Run the pass to transform a HUGR, returning a PassResult.

        See :func:`_impl_pass_call` for a helper function to implement this method.
        """

    @property
    def name(self) -> PassName:
        """Returns the name of the pass."""
        return self.__class__.__name__

    def then(self, other: ComposablePass) -> ComposablePass:
        """Perform another composable pass after this pass."""
        return ComposedPass(self, other)


def impl_pass_run(
    *,
    hugr: Hugr,
    inplace: bool,
    inplace_call: Callable[[Hugr], PassResult] | None = None,
    copy_call: Callable[[Hugr], PassResult] | None = None,
) -> PassResult:
    """Helper function to implement a ComposablePass.run method, given an
    inplace or copy-returning pass methods.

    At least one of the `inplace_call` or `copy_call` arguments must be provided.

    :param hugr: The Hugr to apply the pass to.
    :param inplace: Whether to apply the pass inplace.
    :param inplace_call: The method to apply the pass inplace.
    :param copy_call: The method to apply the pass by copying the Hugr.
    :return: The result of the pass application.
    :raises ValueError: If neither `inplace_call` nor `copy_call` is provided.
    """
    if inplace and inplace_call is not None:
        return inplace_call(hugr)
    elif inplace and copy_call is not None:
        pass_result = copy_call(hugr)
        pass_result.hugr = hugr
        if pass_result.modified:
            hugr._overwrite_hugr(pass_result.hugr)
            pass_result.original_dirty = True
        return pass_result
    elif not inplace and copy_call is not None:
        return copy_call(hugr)
    elif not inplace and inplace_call is not None:
        new_hugr = deepcopy(hugr)
        pass_result = inplace_call(new_hugr)
        pass_result.original_dirty = False
        return pass_result
    else:
        msg = "Pass must implement at least an inplace or copy run method"
        raise ValueError(msg)


@dataclass
class ComposedPass(ComposablePass):
    """A sequence of composable passes."""

    passes: list[ComposablePass]

    def __init__(self, *passes: ComposablePass) -> None:
        self.passes = []
        for pass_ in passes:
            if isinstance(pass_, ComposedPass):
                self.passes.extend(pass_.passes)
            else:
                self.passes.append(pass_)

    def run(self, hugr: Hugr, *, inplace: bool = True) -> PassResult:
        def apply(hugr: Hugr) -> PassResult:
            pass_result = PassResult(hugr=hugr)
            for comp_pass in self.passes:
                new_result = comp_pass.run(pass_result.hugr, inplace=inplace)
                pass_result = pass_result.then(new_result)
            return pass_result

        return impl_pass_run(
            hugr=hugr,
            inplace=inplace,
            inplace_call=apply,
            copy_call=apply,
        )

    @property
    def name(self) -> PassName:
        return f"Composed({ ', '.join(pass_.name for pass_ in self.passes) })"


@dataclass
class PassResult:
    """The result of a series of composed passes applied to a HUGR.

    Includes a flag indicating whether the passes modified the HUGR, and an
    arbitrary result object for each pass.

    In some cases, `modified` may be set to `True` even if the pass did not
    modify the program.

    :attr hugr: The transformed Hugr.
    :attr original_dirty: Whether the original HUGR was modified by the pass.
    :attr modified: Whether the pass made changes to the HUGR.
    :attr results: The result of each applied pass, as a tuple of the pass and
        the result.
    """

    hugr: Hugr
    original_dirty: bool = False
    modified: bool = False
    results: list[tuple[PassName, Any]] = field(default_factory=list)

    @classmethod
    def for_pass(
        cls,
        pass_: ComposablePass,
        hugr: Hugr,
        *,
        result: Any,
        inline: bool,
        modified: bool = True,
    ) -> PassResult:
        """Create a new PassResult after a pass application.

        :param hugr: The Hugr that was transformed.
        :param pass_: The pass that was applied.
        :param result: The result of the pass application.
        :param inline: Whether the pass was applied inplace.
        :param modified: Whether the pass modified the HUGR.
        """
        return cls(
            hugr=hugr,
            original_dirty=inline and modified,
            modified=modified,
            results=[(pass_.name, result)],
        )

    def then(self, other: PassResult) -> PassResult:
        """Extend the PassResult with the results of another PassResult.

        Keeps the hugr returned by the last pass.
        """
        return PassResult(
            hugr=other.hugr,
            original_dirty=self.original_dirty or other.original_dirty,
            modified=self.modified or other.modified,
            results=self.results + other.results,
        )
