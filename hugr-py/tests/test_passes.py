from copy import deepcopy

import pytest

from hugr.hugr.base import Hugr
from hugr.passes._composable_pass import (
    ComposablePass,
    ComposedPass,
    PassResult,
    implement_pass_run,
)


def test_composable_pass() -> None:
    class DummyInlinePass(ComposablePass):
        def run(self, hugr: Hugr, inplace: bool = True) -> PassResult:
            return implement_pass_run(
                self,
                hugr=hugr,
                inplace=inplace,
                inplace_call=lambda hugr: PassResult.for_pass(
                    self,
                    hugr,
                    result=None,
                    inplace=True,
                    # Say that we modified the HUGR even though we didn't
                    modified=True,
                ),
            )

    class DummyCopyPass(ComposablePass):
        def run(self, hugr: Hugr, inplace: bool = True) -> PassResult:
            return implement_pass_run(
                self,
                hugr=hugr,
                inplace=inplace,
                copy_call=lambda hugr: PassResult.for_pass(
                    self,
                    deepcopy(hugr),
                    result=None,
                    inplace=False,
                    # Say that we modified the HUGR even though we didn't
                    modified=True,
                ),
            )

    dummy_inline = DummyInlinePass()
    dummy_copy = DummyCopyPass()

    composed_dummies = dummy_inline.then(dummy_copy)
    assert isinstance(composed_dummies, ComposedPass)

    assert dummy_inline.name == "DummyInlinePass"
    assert dummy_copy.name == "DummyCopyPass"
    assert composed_dummies.name == "Composed(DummyInlinePass, DummyCopyPass)"
    assert composed_dummies.then(dummy_inline).then(composed_dummies).name == (
        "Composed("
        + "DummyInlinePass, DummyCopyPass, "
        + "DummyInlinePass, "
        + "DummyInlinePass, DummyCopyPass)"
    )

    # Apply the passes
    hugr: Hugr = Hugr()
    new_hugr = composed_dummies(hugr, inplace=False)
    assert hugr == new_hugr
    assert new_hugr is not hugr

    # Verify the pass results
    hugr = Hugr()
    inplace_result = composed_dummies.run(hugr, inplace=True)
    assert inplace_result.modified
    assert inplace_result.inplace
    assert inplace_result.results == [
        ("DummyInlinePass", None),
        ("DummyCopyPass", None),
    ]
    assert inplace_result.hugr is hugr

    hugr = Hugr()
    copy_result = composed_dummies.run(hugr, inplace=False)
    assert copy_result.modified
    assert not copy_result.inplace
    assert copy_result.results == [
        ("DummyInlinePass", None),
        ("DummyCopyPass", None),
    ]
    assert copy_result.hugr is not hugr


def test_invalid_composable_pass() -> None:
    class DummyInvalidPass(ComposablePass):
        def run(self, hugr: Hugr, inplace: bool = True) -> PassResult:
            return implement_pass_run(
                self,
                hugr=hugr,
                inplace=inplace,
            )

    dummy_invalid = DummyInvalidPass()
    with pytest.raises(
        ValueError,
        match="DummyInvalidPass needs to implement at least an inplace or copy run method",  # noqa: E501
    ):
        dummy_invalid.run(Hugr())
