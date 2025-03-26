from __future__ import annotations

import pytest

from hugr import tys
from hugr.build import dfg
from hugr.val import (
    FALSE,
    TRUE,
    Left,
    None_,
    Right,
    Some,
    Sum,
    Tuple,
    UnitSum,
    Value,
    bool_value,
)

from .conftest import validate


def test_sums():
    assert Sum(0, tys.Tuple(), []) == Tuple()
    assert Sum(0, tys.Tuple(tys.Bool, tys.Bool), [TRUE, FALSE]) == Tuple(TRUE, FALSE)

    ty = tys.Sum([[], [tys.Bool, tys.Bool]])
    assert Sum(1, ty, [TRUE, FALSE]) == Some(TRUE, FALSE)
    assert Sum(1, ty, [TRUE, FALSE]) == Right([], [TRUE, FALSE])
    assert Sum(0, ty, []) == None_(tys.Bool, tys.Bool)
    assert Sum(0, ty, []) == Left([], [tys.Bool, tys.Bool])

    ty = tys.Sum([[tys.Bool], [tys.Bool]])
    assert Sum(0, ty, [TRUE]) == Left([TRUE], [tys.Bool])
    assert Sum(1, ty, [FALSE]) == Right([tys.Bool], [FALSE])

    assert Tuple() == Sum(0, tys.Tuple(), [])
    assert UnitSum(0, size=1) == Tuple()
    assert UnitSum(2, size=4) == Sum(2, tys.UnitSum(size=4), [])


@pytest.mark.parametrize(
    ("value", "string", "repr_str"),
    [
        (
            Sum(0, tys.Sum([[tys.Bool], [tys.Qubit]]), [TRUE, FALSE]),
            "Sum(tag=0, typ=Sum([[Bool], [Qubit]]), vals=[TRUE, FALSE])",
            "Sum(tag=0, typ=Sum([[Bool], [Qubit]]), vals=[TRUE, FALSE])",
        ),
        (UnitSum(0, size=1), "Unit", "Unit"),
        (UnitSum(0, size=2), "FALSE", "FALSE"),
        (UnitSum(1, size=2), "TRUE", "TRUE"),
        (UnitSum(2, size=5), "UnitSum(2, 5)", "UnitSum(2, 5)"),
        (Tuple(TRUE, FALSE), "Tuple(TRUE, FALSE)", "Tuple(TRUE, FALSE)"),
        (Some(TRUE, FALSE), "Some(TRUE, FALSE)", "Some(TRUE, FALSE)"),
        (None_(tys.Bool, tys.Bool), "None", "None(Bool, Bool)"),
        (
            Left([TRUE, FALSE], [tys.Bool]),
            "Left(TRUE, FALSE)",
            "Left(vals=[TRUE, FALSE], right_typ=[Bool])",
        ),
        (
            Right([tys.Bool, tys.Bool], [FALSE]),
            "Right(FALSE)",
            "Right(left_typ=[Bool, Bool], vals=[FALSE])",
        ),
    ],
)
def test_val_sum_str(value: Value, string: str, repr_str: str):
    assert str(value) == string
    assert repr(value) == repr_str


def test_val_static_array():
    from hugr.std.collections.static_array import StaticArrayVal

    h = dfg.Dfg()
    load = h.load(
        StaticArrayVal([bool_value(x) for x in [True, False]], tys.Bool, "arr")
    )
    h.set_outputs(load)
    validate(h.hugr)
