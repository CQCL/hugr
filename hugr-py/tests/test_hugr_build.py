from __future__ import annotations
from dataclasses import dataclass, field
import subprocess
import os
import pathlib
from hugr._hugr import Dfg, Hugr, Node, Wire
from hugr._ops import Custom, Command
import hugr._ops as ops
from hugr.serialization import SerialHugr
import hugr.serialization.tys as stys
import pytest
import json

BOOL_T = stys.Type(stys.SumType(stys.UnitSum(size=2)))
QB_T = stys.Type(stys.Qubit())
ARG_5 = stys.TypeArg(stys.BoundedNatArg(n=5))
INT_T = stys.Type(
    stys.Opaque(
        extension="arithmetic.int.types",
        id="int",
        args=[ARG_5],
        bound=stys.TypeBound.Eq,
    )
)


@dataclass
class LogicOps(Custom):
    extension: stys.ExtensionId = "logic"


# TODO get from YAML
@dataclass
class NotDef(LogicOps):
    num_out: int | None = 1
    op_name: str = "Not"
    signature: stys.FunctionType = field(
        default_factory=lambda: stys.FunctionType(input=[BOOL_T], output=[BOOL_T])
    )

    def __call__(self, a: Wire) -> Command:
        return super().__call__(a)


Not = NotDef()


@dataclass
class IntOps(Custom):
    extension: stys.ExtensionId = "arithmetic.int"


@dataclass
class DivModDef(IntOps):
    num_out: int | None = 2
    extension: stys.ExtensionId = "arithmetic.int"
    op_name: str = "idivmod_u"
    signature: stys.FunctionType = field(
        default_factory=lambda: stys.FunctionType(input=[INT_T] * 2, output=[INT_T] * 2)
    )
    args: list[stys.TypeArg] = field(default_factory=lambda: [ARG_5, ARG_5])


DivMod = DivModDef()


def _validate(h: Hugr, mermaid: bool = False, roundtrip: bool = True):
    workspace_dir = pathlib.Path(__file__).parent.parent.parent
    # use the HUGR_BIN environment variable if set, otherwise use the debug build
    bin_loc = os.environ.get("HUGR_BIN", str(workspace_dir / "target/debug/hugr"))
    cmd = [bin_loc, "-"]

    if mermaid:
        cmd.append("--mermaid")
    serial = h.to_serial().to_json()
    subprocess.run(cmd, check=True, input=serial.encode())

    if roundtrip:
        h2 = Hugr.from_serial(SerialHugr.load_json(json.loads(serial)))
        assert serial == h2.to_serial().to_json()


def test_stable_indices():
    h = Hugr(ops.DFG())

    nodes = [h.add_node(Not) for _ in range(3)]
    assert len(h) == 4

    h.add_link(nodes[0].out(0), nodes[1].inp(0))

    assert h.num_outgoing(nodes[0]) == 1
    assert h.num_incoming(nodes[1]) == 1

    assert h.delete_node(nodes[1]) is not None
    assert h._nodes[nodes[1].idx] is None

    assert len(h) == 3
    assert len(h._nodes) == 4
    assert h._free_nodes == [nodes[1]]

    assert h.num_outgoing(nodes[0]) == 0
    assert h.num_incoming(nodes[1]) == 0

    with pytest.raises(KeyError):
        _ = h[nodes[1]]
    with pytest.raises(KeyError):
        _ = h[Node(46)]

    new_n = h.add_node(Not)
    assert new_n == nodes[1]

    assert len(h) == 4
    assert h._free_nodes == []


def test_simple_id():
    h = Dfg.endo([QB_T] * 2)
    a, b = h.inputs()
    h.set_outputs(a, b)

    _validate(h.hugr)


def test_multiport():
    h = Dfg([BOOL_T], [BOOL_T] * 2)
    (a,) = h.inputs()
    h.set_outputs(a, a)
    in_n, ou_n = h.input_node, h.output_node
    assert list(h.hugr.outgoing_links(in_n)) == [
        (in_n.out(0), [ou_n.inp(0), ou_n.inp(1)]),
    ]

    assert list(h.hugr.incoming_links(ou_n)) == [
        (ou_n.inp(0), [in_n.out(0)]),
        (ou_n.inp(1), [in_n.out(0)]),
    ]

    assert list(h.hugr.linked_ports(in_n.out(0))) == [
        ou_n.inp(0),
        ou_n.inp(1),
    ]

    assert list(h.hugr.linked_ports(ou_n.inp(0))) == [in_n.out(0)]
    _validate(h.hugr)


def test_add_op():
    h = Dfg.endo([BOOL_T])
    (a,) = h.inputs()
    nt = h.add_op(Not, a)
    h.set_outputs(nt)

    _validate(h.hugr)


def test_tuple():
    row = [BOOL_T, QB_T]
    h = Dfg.endo(row)
    a, b = h.inputs()
    t = h.add(ops.MakeTuple(row)(a, b))
    a, b = h.add(ops.UnpackTuple(row)(t))
    h.set_outputs(a, b)

    _validate(h.hugr)

    h1 = Dfg.endo(row)
    a, b = h1.inputs()
    mt = h1.add_op(ops.MakeTuple(row), a, b)
    a, b = h1.add_op(ops.UnpackTuple(row), mt)[0, 1]
    h1.set_outputs(a, b)

    assert h.hugr.to_serial() == h1.hugr.to_serial()


def test_multi_out():
    h = Dfg([INT_T] * 2, [INT_T] * 2)
    a, b = h.inputs()
    a, b = h.add(DivMod(a, b))
    h.set_outputs(a, b)
    _validate(h.hugr)


def test_insert():
    h1 = Dfg.endo([BOOL_T])
    (a1,) = h1.inputs()
    nt = h1.add(Not(a1))
    h1.set_outputs(nt)

    assert len(h1.hugr) == 4

    new_h = Hugr(ops.DFG())
    mapping = h1.hugr.insert_hugr(new_h, h1.hugr.root)
    assert mapping == {new_h.root: Node(4)}


def test_insert_nested():
    h1 = Dfg.endo([BOOL_T])
    (a1,) = h1.inputs()
    nt = h1.add(Not(a1))
    h1.set_outputs(nt)

    h = Dfg.endo([BOOL_T])
    (a,) = h.inputs()
    nested = h.insert_nested(h1, a)
    h.set_outputs(nested)

    _validate(h.hugr)


def test_build_nested():
    def _nested_nop(dfg: Dfg):
        (a1,) = dfg.inputs()
        nt = dfg.add(Not(a1))
        dfg.set_outputs(nt)

    h = Dfg.endo([BOOL_T])
    (a,) = h.inputs()
    nested = h.add_nested([BOOL_T], [BOOL_T], a)

    _nested_nop(nested)

    h.set_outputs(nested.root)

    _validate(h.hugr)


def test_build_inter_graph():
    h = Dfg.endo([BOOL_T])
    (a,) = h.inputs()
    nested = h.add_nested([], [BOOL_T])

    nt = nested.add(Not(a))
    nested.set_outputs(nt)
    # TODO a context manager could add this state order edge on
    # exit by tracking parents of source nodes
    h.add_state_order(h.input_node, nested.root)
    h.set_outputs(nested.root)

    _validate(h.hugr)
