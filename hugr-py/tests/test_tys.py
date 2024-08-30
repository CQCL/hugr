from __future__ import annotations

import pytest

from hugr.std.float import FLOAT_T
from hugr.std.int import INT_T, _int_tv
from hugr.tys import (
    Alias,
    Array,
    Bool,
    BoundedNatArg,
    BoundedNatParam,
    Either,
    ExtensionsArg,
    ExtensionsParam,
    ExtType,
    FunctionType,
    ListParam,
    Option,
    PolyFuncType,
    Qubit,
    RowVariable,
    SequenceArg,
    StringArg,
    StringParam,
    Sum,
    Tuple,
    TupleParam,
    Type,
    TypeArg,
    TypeBound,
    TypeParam,
    TypeTypeArg,
    TypeTypeParam,
    UnitSum,
    USize,
    Variable,
    VariableArg,
)


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


@pytest.mark.parametrize(
    ("param", "string"),
    [
        (TypeTypeParam(TypeBound.Any), "Any"),
        (BoundedNatParam(3), "Nat(3)"),
        (BoundedNatParam(None), "Nat"),
        (StringParam(), "String"),
        (
            TupleParam([TypeTypeParam(TypeBound.Any), BoundedNatParam(3)]),
            "(Any, Nat(3))",
        ),
        (ListParam(StringParam()), "[String]"),
        (ExtensionsParam(), "Extensions"),
    ],
)
def test_params_str(param: TypeParam, string: str):
    assert str(param) == string


@pytest.mark.parametrize(
    ("arg", "string"),
    [
        (TypeTypeArg(Bool), "Type(Bool)"),
        (BoundedNatArg(3), "3"),
        (StringArg("hello"), '"hello"'),
        (
            SequenceArg([TypeTypeArg(Qubit), BoundedNatArg(3)]),
            "(Type(Qubit), 3)",
        ),
        (VariableArg(2, StringParam()), "$2"),
        (ExtensionsArg(["A", "B"]), "Extensions(A, B)"),
    ],
)
def test_args_str(arg: TypeArg, string: str):
    assert str(arg) == string


@pytest.mark.parametrize(
    ("ty", "string"),
    [
        (Array(Bool, 3), "Array<Bool, 3>"),
        (Variable(2, TypeBound.Any), "$2"),
        (RowVariable(4, TypeBound.Copyable), "$4"),
        (USize(), "USize"),
        (INT_T, "int<5>"),
        (FLOAT_T, "float64"),
        (Alias("Foo", TypeBound.Copyable), "Foo"),
        (FunctionType([Bool, Qubit], [Qubit, Bool]), "Bool, Qubit -> Qubit, Bool"),
        (
            PolyFuncType(
                [TypeTypeParam(TypeBound.Any), BoundedNatParam(7)],
                FunctionType([_int_tv(1)], [Variable(0, TypeBound.Copyable)]),
            ),
            "∀ Any, Nat(7). int<$1> -> $0",
        ),
    ],
)
def test_tys_str(ty: Type, string: str):
    assert str(ty) == string
    if isinstance(ty, ExtType):
        assert str(ty._to_opaque()) == string
