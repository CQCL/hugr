from hugr.hugr.base import Hugr
from hugr.passes._composable_pass import (
    ComposablePass,
    ComposedPass,
    PassResult,
    impl_pass_run,
)


def test_composable_pass() -> None:
    class MyDummyPass(ComposablePass):
        def run(self, hugr: Hugr, inplace: bool = True) -> PassResult:
            return impl_pass_run(
                hugr=hugr,
                inplace=inplace,
                inplace_call=lambda hugr: PassResult.for_pass(
                    self,
                    hugr,
                    result=None,
                    inline=True,
                    modified=False,
                ),
            )

    dummy = MyDummyPass()

    composed_dummies = dummy.then(dummy)

    my_composed_pass = ComposedPass(dummy, dummy)
    assert my_composed_pass.passes == [dummy, dummy]

    assert isinstance(composed_dummies, ComposablePass)
    assert composed_dummies == my_composed_pass

    assert dummy.name == "MyDummyPass"
    assert composed_dummies.name == "Composed(MyDummyPass, MyDummyPass)"

    assert (
        composed_dummies.then(my_composed_pass).name
        == "Composed(MyDummyPass, MyDummyPass, MyDummyPass, MyDummyPass)"
    )
