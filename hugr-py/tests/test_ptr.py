"""Check the generated pointer extension used by Python."""

import pytest

from hugr import tys
from hugr.std import ptr


@pytest.mark.parametrize("element", [tys.Bool, tys.Qubit])
def test_pointer_is_linear(element: tys.Type) -> None:
    assert ptr.Ptr(element).type_bound() == tys.TypeBound.Linear


def test_pointer_operations() -> None:
    assert set(ptr.EXTENSION.operations) == {
        "New",
        "Read",
        "Write",
        "Swap",
        "Dup",
        "Eq",
        "Free",
        "Map",
    }


def test_pointer_eq_signature() -> None:
    element = tys.Variable(0, tys.TypeBound.Linear)
    pointer = ptr.Ptr(element)
    signature = ptr.EXTENSION.get_op("Eq").signature.poly_func
    assert signature is not None
    expected = tys.PolyFuncType(
        [tys.TypeTypeParam(tys.TypeBound.Linear)],
        tys.FunctionType([pointer, pointer], [pointer, pointer, tys.Bool]),
    )
    assert signature._to_serial() == expected._to_serial()
