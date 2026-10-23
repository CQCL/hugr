from __future__ import annotations

import pytest
from semver import Version

from hugr import val
from hugr.std.collections.array import Array, ArrayVal
from hugr.std.collections.borrow_array import BorrowArray, BorrowArrayVal
from hugr.std.collections.list import List, ListVal
from hugr.std.collections.static_array import StaticArray, StaticArrayVal
from hugr.std.float import FLOAT_T
from hugr.std.int import INT_T, _int_tv
from hugr.tys import (
    Alias,
    Bool,
    BoundedNatArg,
    BoundedNatParam,
    BytesArg,
    BytesParam,
    ConstParam,
    Either,
    ExtType,
    FloatArg,
    FloatParam,
    FunctionType,
    ListArg,
    ListParam,
    Opaque,
    Option,
    PolyFuncType,
    Qubit,
    RowVariable,
    StringArg,
    StringParam,
    Sum,
    Tuple,
    TupleArg,
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
        (TypeTypeParam(TypeBound.Linear), "Linear"),
        (BoundedNatParam(3), "Nat(3)"),
        (BoundedNatParam(None), "Nat"),
        (StringParam(), "String"),
        (FloatParam(), "Float"),
        (BytesParam(), "Bytes"),
        (
            TupleParam([TypeTypeParam(TypeBound.Linear), BoundedNatParam(3)]),
            "(Linear, Nat(3))",
        ),
        (ListParam(StringParam()), "[String]"),
        (ConstParam(Qubit), "Const(Qubit)"),
    ],
)
def test_params_str(param: TypeParam, string: str):
    assert str(param) == string


@pytest.mark.parametrize(
    ("arg", "string"),
    [
        (TypeTypeArg(Bool), "Type(Bool)"),
        (BoundedNatArg(3), "3"),
        (FloatArg(3.141), "3.141"),
        (BytesArg(b"0"), "bytes"),
        (StringArg("hello"), '"hello"'),
        (
            ListArg([TypeTypeArg(Qubit), BoundedNatArg(3)]),
            "[Type(Qubit), 3]",
        ),
        (
            TupleArg([TypeTypeArg(Qubit), BoundedNatArg(3)]),
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
        (BorrowArray(Bool, 3), "borrow_array<3, Type(Bool)>"),
        (Variable(2, TypeBound.Linear), "$2"),
        (RowVariable(4, TypeBound.Copyable), "$4"),
        (USize(), "USize"),
        (INT_T, "int<5>"),
        (FLOAT_T, "float64"),
        (Alias("Foo", TypeBound.Copyable), "Foo"),
        (FunctionType([Bool, Qubit], [Qubit, Bool]), "Bool, Qubit -> Qubit, Bool"),
        (
            PolyFuncType(
                [TypeTypeParam(TypeBound.Linear), BoundedNatParam(7)],
                FunctionType([_int_tv(1)], [Variable(0, TypeBound.Copyable)]),
            ),
            "∀ Linear, Nat(7). int<$1> -> $0",
        ),
    ],
)
def test_tys_str(ty: Type, string: str):
    assert ty.render() == string
    assert str(ty) == string
    if isinstance(ty, ExtType):
        assert str(ty._to_opaque()) == string


def test_render_extension_version() -> None:
    assert INT_T.render() == str(INT_T)
    assert (
        INT_T.render(extension_version=True)
        == f"{INT_T}@{INT_T.get_extension_version()}"
    )


def test_render_qualified_extension_name() -> None:
    assert INT_T.render(qualified_name=True) == "arithmetic.int.types.int<5>"
    assert (
        INT_T.render(
            extension_version=True,
            qualified_name=True,
        )
        == f"arithmetic.int.types.int<5>@{INT_T.get_extension_version()}"
    )


def test_render_nested_extension_versions() -> None:
    inner = Opaque(
        "Inner",
        TypeBound.Copyable,
        extension="inner",
        extension_version=Version(1, 2, 3),
    )
    outer = Opaque(
        "Outer",
        TypeBound.Copyable,
        args=[TypeTypeArg(inner)],
        extension="outer",
        extension_version=Version(2, 3, 4),
    )
    nested = Tuple(outer, inner)

    assert nested.render() == "Tuple(Outer<Type(Inner)>, Inner)"
    assert nested.render(extension_version=True) == (
        "Tuple(Outer<Type(Inner@1.2.3)>@2.3.4, Inner@1.2.3)"
    )
    assert nested.render(qualified_name=True) == (
        "Tuple(outer.Outer<Type(inner.Inner)>, inner.Inner)"
    )
    assert nested.render(extension_version=True, qualified_name=True) == (
        "Tuple(outer.Outer<Type(inner.Inner@1.2.3)>@2.3.4, " "inner.Inner@1.2.3)"
    )


def test_render_qualified_opaque_without_extension() -> None:
    opaque = Opaque("Standalone", TypeBound.Copyable)

    assert opaque.render(qualified_name=True) == "Standalone"


def test_list():
    ty_var = Variable(0, TypeBound.Copyable)

    ls = List(Bool)
    assert ls.ty == Bool

    ls = List(ty_var)
    assert ls.ty == ty_var

    l_val = ListVal([val.TRUE, val.FALSE], Bool)
    assert l_val.v == [val.TRUE, val.FALSE]
    assert l_val.ty == List(Bool)


@pytest.mark.parametrize("contents", [[], [val.TRUE, val.FALSE]])
def test_list_serialization(contents):
    # Rust's ListValue tuple struct expects [values, element_type].
    serialized = ListVal(contents, Bool).to_value()
    assert serialized.name == "ListValue"
    assert serialized.val == [
        [v._to_serial_root() for v in contents],
        Bool._to_serial_root(),
    ]


def test_array():
    ty_var = Variable(0, TypeBound.Copyable)
    len_var = VariableArg(1, BoundedNatParam())

    ls = Array(Bool, 3)
    assert ls.ty == Bool
    assert ls.size == 3
    assert ls.type_bound() == TypeBound.Linear

    ls = Array(ty_var, len_var)
    assert ls.ty == ty_var
    assert ls.size is None
    assert ls.type_bound() == TypeBound.Linear

    ar_val = ArrayVal([val.TRUE, val.FALSE], Bool)
    assert ar_val.v == [val.TRUE, val.FALSE]
    assert ar_val.ty == Array(Bool, 2)


def test_borrow_array():
    ty_var = Variable(0, TypeBound.Copyable)
    len_var = VariableArg(1, BoundedNatParam())

    ls = BorrowArray(Bool, 3)
    assert ls.ty == Bool
    assert ls.size == 3
    assert ls.type_bound() == TypeBound.Linear

    ls = BorrowArray(ty_var, len_var)
    assert ls.ty == ty_var
    assert ls.size is None
    assert ls.type_bound() == TypeBound.Linear

    ar_val = BorrowArrayVal([val.TRUE, val.FALSE], Bool)
    assert ar_val.v == [val.TRUE, val.FALSE]
    assert ar_val.ty == BorrowArray(Bool, 2)


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
