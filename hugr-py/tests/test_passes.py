from copy import deepcopy

import pytest

from hugr.hugr.base import Hugr
from hugr.passes._composable_pass import (
    ComposablePass,
    ComposedPass,
    PassResult,
    impl_pass_run,
)


def test_composable_pass() -> None:
    class MyDummyInlinePass(ComposablePass):
        def run(self, hugr: Hugr, inplace: bool = True) -> PassResult:
            return impl_pass_run(
                self,
                hugr=hugr,
                inplace=inplace,
                inplace_call=lambda hugr: PassResult.for_pass(
                    self,
                    hugr,
                    result=None,
                    inline=True,
                    # Say that we modified the HUGR even though we didn't
                    modified=True,
                ),
            )

    class MyDummyCopyPass(ComposablePass):
        def run(self, hugr: Hugr, inplace: bool = True) -> PassResult:
            return impl_pass_run(
                self,
                hugr=hugr,
                inplace=inplace,
                copy_call=lambda hugr: PassResult.for_pass(
                    self,
                    deepcopy(hugr),
                    result=None,
                    inline=False,
                    # Say that we modified the HUGR even though we didn't
                    modified=True,
                ),
            )

    dummy_inline = MyDummyInlinePass()
    dummy_copy = MyDummyCopyPass()

    composed_dummies = dummy_inline.then(dummy_copy)
    assert isinstance(composed_dummies, ComposedPass)

    assert dummy_inline.name == "MyDummyInlinePass"
    assert dummy_copy.name == "MyDummyCopyPass"
    assert composed_dummies.name == "Composed(MyDummyInlinePass, MyDummyCopyPass)"
    assert composed_dummies.then(dummy_inline).then(composed_dummies).name == (
        "Composed("
        + "MyDummyInlinePass, MyDummyCopyPass, "
        + "MyDummyInlinePass, "
        + "MyDummyInlinePass, MyDummyCopyPass)"
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
    assert inplace_result.original_dirty
    assert inplace_result.results == [
        ("MyDummyInlinePass", None),
        ("MyDummyCopyPass", None),
    ]
    assert inplace_result.hugr is hugr

    hugr = Hugr()
    copy_result = composed_dummies.run(hugr, inplace=False)
    assert copy_result.modified
    assert not copy_result.original_dirty
    assert copy_result.results == [
        ("MyDummyInlinePass", None),
        ("MyDummyCopyPass", None),
    ]
    assert copy_result.hugr is not hugr


def test_invalid_composable_pass() -> None:
    class MyDummyInvalidPass(ComposablePass):
        def run(self, hugr: Hugr, inplace: bool = True) -> PassResult:
            return impl_pass_run(
                self,
                hugr=hugr,
                inplace=inplace,
            )

    dummy_invalid = MyDummyInvalidPass()
    with pytest.raises(
        ValueError,
        match="MyDummyInvalidPass needs to implement at least an inplace or copy run method",  # noqa: E501
    ):
        dummy_invalid.run(Hugr())
