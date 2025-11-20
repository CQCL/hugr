from hugr.hugr.base import Hugr
from hugr.passes._composable_pass import ComposablePass, ComposedPass, impl_pass_call


def test_composable_pass() -> None:
    class MyDummyPass(ComposablePass):
        def __call__(self, hugr: Hugr, inplace: bool = True) -> Hugr:
            return impl_pass_call(
                hugr=hugr,
                inplace=inplace,
                inplace_call=lambda hugr: None,
            )

    dummy = MyDummyPass()

    composed_dummies = dummy.then(dummy)

    my_composed_pass = ComposedPass([dummy, dummy])
    assert my_composed_pass.passes == [dummy, dummy]

    assert isinstance(composed_dummies, ComposablePass)
    assert composed_dummies == my_composed_pass

    assert dummy.name == "MyDummyPass"
    assert composed_dummies.name == "Composed(MyDummyPass, MyDummyPass)"

    assert (
        composed_dummies.then(my_composed_pass).name
        == "Composed(MyDummyPass, MyDummyPass, MyDummyPass, MyDummyPass)"
    )
