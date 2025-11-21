from hugr.hugr.base import Hugr
from hugr.passes._composable_pass import ComposablePass, ComposedPass


def test_composable_pass() -> None:
    class MyDummyPass(ComposablePass):
        def __call__(self, hugr: Hugr, inplace: bool = True) -> Hugr:
            return self(hugr, inplace)

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
