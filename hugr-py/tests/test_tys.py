from __future__ import annotations

import pytest

from hugr.tys import Bool, Either, Option, Qubit, Sum, Tuple, Type, UnitSum


def test_sums():
    assert Sum([[Bool, Qubit]]) == Tuple(Bool, Qubit)
    assert Tuple(Bool, Qubit) == Sum([[Bool, Qubit]])
    assert Sum([[Bool, Qubit]]).as_tuple() == Sum([[Bool, Qubit]])

    assert Sum([[], [Bool, Qubit]]) == Option(Bool, Qubit)
    assert Sum([[], [Bool, Qubit]]) == Either([], [Bool, Qubit])
    assert Option(Bool, Qubit) == Either([], [Bool, Qubit])
    assert Sum([[Qubit], [Bool]]) == Either([Qubit], [Bool])

    assert Tuple() == Sum([[]])
    assert UnitSum(0) == Sum([])
    assert UnitSum(1) == Tuple()
    assert UnitSum(4) == Sum([[], [], [], []])


@pytest.mark.parametrize(
    ("ty", "string", "repr_str"),
    [
        (
            Sum([[Bool], [Qubit], [Qubit, Bool]]),
            "Sum([[Bool], [Qubit], [Qubit, Bool]])",
            "Sum([[Bool], [Qubit], [Qubit, Bool]])",
        ),
        (UnitSum(1), "Unit", "Unit"),
        (UnitSum(2), "Bool", "Bool"),
        (UnitSum(3), "UnitSum(3)", "UnitSum(3)"),
        (Tuple(Bool, Qubit), "Tuple(Bool, Qubit)", "Tuple(Bool, Qubit)"),
        (Option(Bool, Qubit), "Option(Bool, Qubit)", "Option(Bool, Qubit)"),
        (
            Either([Bool, Qubit], [Bool]),
            "Either((Bool, Qubit), Bool)",
            "Either(left=[Bool, Qubit], right=[Bool])",
        ),
    ],
)
def test_tys_sum_str(ty: Type, string: str, repr_str: str):
    assert str(ty) == string
    assert repr(ty) == repr_str
