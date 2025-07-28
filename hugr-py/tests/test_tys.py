from __future__ import annotations

import pytest

from hugr import val
from hugr.std.collections.array import Array, ArrayVal
from hugr.std.collections.list import List, ListVal
from hugr.std.collections.static_array import StaticArray, StaticArrayVal
from hugr.std.collections.value_array import ValueArray, ValueArrayVal
from hugr.std.float import FLOAT_T
from hugr.std.int import INT_T, _int_tv
from hugr.tys import (
    Alias,
    Bool,
    BoundedNatArg,
    BoundedNatParam,
    Either,
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
    ],
)
def test_args_str(arg: TypeArg, string: str):
    assert str(arg) == string


@pytest.mark.parametrize(
    ("ty", "string"),
    [
        (Array(Bool, 3), "array<3, Type(Bool)>"),
        (StaticArray(Bool), "static_array<Type(Bool)>"),
        (ValueArray(Bool, 3), "value_array<3, Type(Bool)>"),
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


def test_list():
    ty_var = Variable(0, TypeBound.Copyable)

    ls = List(Bool)
    assert ls.ty == Bool

    ls = List(ty_var)
    assert ls.ty == ty_var

    l_val = ListVal([val.TRUE, val.FALSE], Bool)
    assert l_val.v == [val.TRUE, val.FALSE]
    assert l_val.ty == List(Bool)


def test_array():
    ty_var = Variable(0, TypeBound.Copyable)
    len_var = VariableArg(1, BoundedNatParam())

    ls = Array(Bool, 3)
    assert ls.ty == Bool
    assert ls.size == 3
    assert ls.type_bound() == TypeBound.Any

    ls = Array(ty_var, len_var)
    assert ls.ty == ty_var
    assert ls.size is None
    assert ls.type_bound() == TypeBound.Any

    ar_val = ArrayVal([val.TRUE, val.FALSE], Bool)
    assert ar_val.v == [val.TRUE, val.FALSE]
    assert ar_val.ty == Array(Bool, 2)


def test_value_array():
    ty_var = Variable(0, TypeBound.Any)
    len_var = VariableArg(1, BoundedNatParam())

    ls = ValueArray(Bool, 3)
    assert ls.ty == Bool
    assert ls.size == 3
    assert ls.type_bound() == TypeBound.Copyable

    ls = ValueArray(ty_var, len_var)
    assert ls.ty == ty_var
    assert ls.size is None
    assert ls.type_bound() == TypeBound.Any

    ar_val = ValueArrayVal([val.TRUE, val.FALSE], Bool)
    assert ar_val.v == [val.TRUE, val.FALSE]
    assert ar_val.ty == ValueArray(Bool, 2)


def test_static_array():
    ty_var = Variable(0, TypeBound.Copyable)

    ls = StaticArray(Bool)
    assert ls.ty == Bool

    ls = StaticArray(ty_var)
    assert ls.ty == ty_var

    name = "array_name"
    ar_val = StaticArrayVal([val.TRUE, val.FALSE], Bool, name)
    assert ar_val.v == [val.TRUE, val.FALSE]
    assert ar_val.ty == StaticArray(Bool)

    with pytest.raises(ValueError, match="Static array elements must be copyable"):
        StaticArray(Qubit)
