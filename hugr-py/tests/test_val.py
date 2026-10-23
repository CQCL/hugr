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
            Sum(0, tys.Sum([[tys.Bool], [tys.Qubit], [tys.Bool]]), [TRUE]),
            "Sum(0, Sum([[Bool], [Qubit], [Bool]]), [TRUE])",
            "Sum(tag=0, typ=Sum([[Bool], [Qubit], [Bool]]), vals=[TRUE])",
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
def test_val_sum_str(value: Sum, string: str, repr_str: str):
    assert str(value) == string
    assert repr(value) == repr_str

    # Make sure the corresponding `Sum` also renders the same
    sum_val = Sum(value.tag, value.typ, value.vals)
    assert str(sum_val) == string
    assert repr(sum_val) == repr_str


def test_val_static_array():
    from hugr.std.collections.static_array import StaticArrayVal

    h = dfg.Dfg()
    load = h.load(
        StaticArrayVal([bool_value(x) for x in [True, False]], tys.Bool, "arr")
    )
    h.set_outputs(load)
    validate(h.hugr)


@pytest.mark.parametrize("contents", [[], [TRUE, FALSE]])
def test_val_list(contents):
    from hugr.std.collections.list import ListVal

    value = ListVal(contents, tys.Bool)
    assert value.to_value().val == [
        [v._to_serial_root() for v in contents],
        tys.Bool._to_serial_root(),
    ]
    h = dfg.Dfg()
    loaded = h.load(value)
    h.set_outputs(loaded)
    validate(h.hugr)


def test_val_list_rejects_wrong_element_type():
    from hugr.cli import HugrCliError
    from hugr.std.collections.list import ListVal
    from hugr.std.float import FLOAT_T

    h = dfg.Dfg()
    loaded = h.load(ListVal([TRUE], FLOAT_T))
    h.set_outputs(loaded)
    with pytest.raises(HugrCliError):
        validate(h.hugr)
