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

        See :func:`implement_pass_run` for a helper function to implement this method.
        """
        ...

    @property
    def name(self) -> PassName:
        """Returns the name of the pass."""
        return self.__class__.__name__

    def then(self, other: ComposablePass) -> ComposablePass:
        """Perform another composable pass after this pass."""
        return ComposedPass(self, other)


def implement_pass_run(
    composable_pass: ComposablePass,
    *,
    hugr: Hugr,
    inplace: bool,
    inplace_call: Callable[[Hugr], PassResult] | None = None,
    copy_call: Callable[[Hugr], PassResult] | None = None,
) -> PassResult:
    """Helper function to implement a ComposablePass.run method, given an
    inplace or copy-returning pass method.

    At least one of the `inplace_call` or `copy_call` arguments must be provided.

    :param composable_pass: The pass being run. Used for error messages.
    :param hugr: The Hugr to apply the pass to.
    :param inplace: Whether to apply the pass inplace.
    :param inplace_call: The method to apply the pass inplace.
    :param copy_call: The method to apply the pass by copying the Hugr.
    :return: The result of the pass application.
    :raises ValueError: If neither `inplace_call` nor `copy_call` is provided.
    """
    if inplace:
        if inplace_call is not None:
            return inplace_call(hugr)
        elif copy_call is not None:
            pass_result = copy_call(hugr)
            if pass_result.modified:
                hugr._overwrite_hugr(pass_result.hugr)
                pass_result.inplace = True
            pass_result.hugr = hugr
            return pass_result
    elif not inplace:
        if copy_call is not None:
            return copy_call(hugr)
        elif inplace_call is not None:
            new_hugr = deepcopy(hugr)
            pass_result = inplace_call(new_hugr)
            pass_result.inplace = False
            return pass_result

    msg = (
        f"{composable_pass.name} needs to implement at least "
        + "an inplace or copy run method"
    )
    raise ValueError(msg)


@dataclass
class ComposedPass(ComposablePass):
    """A sequence of composable passes."""

    passes: list[ComposablePass]

    def __init__(self, *passes: ComposablePass) -> None:
        self.passes = []
        for composable_pass in passes:
            if isinstance(composable_pass, ComposedPass):
                self.passes.extend(composable_pass.passes)
            else:
                self.passes.append(composable_pass)

    def run(self, hugr: Hugr, *, inplace: bool = True) -> PassResult:
        def apply(inplace: bool, hugr: Hugr) -> PassResult:
            pass_result = PassResult(hugr=hugr, inplace=inplace)
            for comp_pass in self.passes:
                new_result = comp_pass.run(pass_result.hugr, inplace=inplace)
                pass_result = pass_result.then(new_result)
            return pass_result

        return implement_pass_run(
            self,
            hugr=hugr,
            inplace=inplace,
            inplace_call=lambda hugr: apply(True, hugr),
            copy_call=lambda hugr: apply(False, hugr),
        )

    @property
    def name(self) -> PassName:
        names = [composable_pass.name for composable_pass in self.passes]
        return f"Composed({ ', '.join(names) })"


@dataclass
class PassResult:
    """The result of a series of composed passes applied to a HUGR.

    Includes a flag indicating whether the passes modified the HUGR, and an
    arbitrary result object for each pass.

    :attr hugr: The transformed Hugr.
    :attr inplace: Whether the pass was applied inplace.
        If this is `True`, `hugr` will be the same object passed as input.
        If this is `False`, `hugr` will be an independent copy of the original Hugr.
    :attr modified: Whether the pass made changes to the HUGR.
        If `False`, `hugr` will have the same contents as the original Hugr.
        If `True`, no guarantees are made about the contents of `hugr`.
    :attr results: The result of each applied pass, as a tuple of the pass name
        and the result.
    """

    hugr: Hugr
    inplace: bool = False
    modified: bool = False
    results: list[tuple[PassName, Any]] = field(default_factory=list)

    @classmethod
    def for_pass(
        cls,
        composable_pass: ComposablePass,
        hugr: Hugr,
        *,
        result: Any,
        inplace: bool,
        modified: bool = True,
    ) -> PassResult:
        """Create a new PassResult after a pass application.

        :param hugr: The Hugr that was transformed.
        :param composable_pass: The pass that was applied.
        :param result: The result of the pass application.
        :param inplace: Whether the pass was applied inplace.
        :param modified: Whether the pass modified the HUGR.
        """
        return cls(
            hugr=hugr,
            inplace=inplace,
            modified=modified,
            results=[(composable_pass.name, result)],
        )

    def then(self, other: PassResult) -> PassResult:
        """Extend the PassResult with the results of another PassResult.

        Keeps the hugr returned by the last pass.
        """
        return PassResult(
            hugr=other.hugr,
            inplace=self.inplace and other.inplace,
            modified=self.modified or other.modified,
            results=self.results + other.results,
        )
