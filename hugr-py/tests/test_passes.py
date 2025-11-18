from hugr.hugr.base import Hugr
from hugr.passes._composable_pass import ComposablePass, ComposedPass


def test_composable_pass() -> None:
    class MyDummyPass(ComposablePass):
        def __call__(self, hugr: Hugr, inplace: bool = True) -> Hugr:
            return self(hugr, inplace)

        def then(self, other: ComposablePass) -> ComposablePass:
            return ComposedPass([self, other])

        @property
        def name(self) -> str:
            return "Dummy"

    dummy = MyDummyPass()

    composed = dummy.then(dummy)

    my_composed_pass = ComposedPass([dummy, dummy])

    assert my_composed_pass.passes == [dummy, dummy]
    assert isinstance(my_composed_pass, ComposablePass)
    assert isinstance(composed, ComposablePass)
    assert dummy.name == "Dummy"
