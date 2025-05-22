import pytest

from hugr.hugr.node_port import InPort, Node, OutPort
from hugr.ops import (
    CFG,
    DFG,
    AliasDecl,
    AliasDefn,
    Call,
    CallIndirect,
    Case,
    Conditional,
    Const,
    DataflowBlock,
    ExitBlock,
    FuncDecl,
    FuncDefn,
    Input,
    LoadConst,
    LoadFunc,
    MakeTuple,
    Module,
    Noop,
    Op,
    Output,
    Tag,
    TailLoop,
    UnpackTuple,
)
from hugr.std.int import DivMod
from hugr.std.logic import Not
from hugr.tys import Bool, OrderKind, PolyFuncType, TypeBound
from hugr.val import TRUE


@pytest.mark.parametrize(
    ("op", "string"),
    [
        (Input([]), "Input"),
        (Output([]), "Output"),
        (Not, "logic.Not"),
        (DivMod, "arithmetic.int.idivmod_u<5>"),
        (MakeTuple(), "MakeTuple"),
        (UnpackTuple(), "UnpackTuple"),
        (Tag(0, Bool), "Tag(0)"),
        (CFG([]), "CFG"),
        (DFG([]), "DFG"),
        (DataflowBlock([]), "DataflowBlock"),
        (ExitBlock([]), "ExitBlock"),
        (LoadConst(), "LoadConst"),
        (Conditional(Bool, []), "Conditional"),
        (TailLoop([], []), "TailLoop"),
        (Case([]), "Case"),
        (Module(), "Module"),
        (Call(PolyFuncType.empty()), "Call"),
        (CallIndirect(), "CallIndirect"),
        (LoadFunc(PolyFuncType.empty()), "LoadFunc"),
        (FuncDefn("foo", []), "FuncDefn(foo)"),
        (FuncDecl("bar", PolyFuncType.empty()), "FuncDecl(bar)"),
        (Const(TRUE), "Const(TRUE)"),
        (Noop(), "Noop"),
        (AliasDecl("baz", TypeBound.Any), "AliasDecl(baz)"),
        (AliasDefn("baz", Bool), "AliasDefn(baz)"),
    ],
)
def test_ops_str(op: Op, string: str):
    assert op.name() == string


@pytest.mark.parametrize(
    "op",
    [Noop(), Call(PolyFuncType.empty()), LoadFunc(PolyFuncType.empty()), LoadConst()],
)
def test_order_ports(op: Op):
    assert op.port_kind(InPort(Node(0), -1)) == OrderKind()
    assert op.port_kind(OutPort(Node(0), -1)) == OrderKind()
