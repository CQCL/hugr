from __future__ import annotations

from hugr.tys import Bool, Qubit, Sum, Tuple, UnitSum


def test_sums():
    assert Sum([[Bool, Qubit]]) == Tuple(Bool, Qubit)
    assert Tuple(Bool, Qubit) == Sum([[Bool, Qubit]])
    assert Sum([[Bool, Qubit]]).as_tuple() == Sum([[Bool, Qubit]])

    assert Tuple() == Sum([[]])
    assert UnitSum(0) == Sum([])
    assert UnitSum(1) == Tuple()
    assert UnitSum(4) == Sum([[], [], [], []])
