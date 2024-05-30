from __future__ import annotations
from dataclasses import dataclass
import subprocess
import os
import pathlib
from hugr._hugr import Dfg, Hugr, DummyOp, Node, Command, Wire, Op
from hugr.serialization import SerialHugr
import hugr.serialization.tys as stys
import hugr.serialization.ops as sops
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

NOT_OP = DummyOp(
    # TODO get from YAML
    sops.CustomOp(
        parent=-1,
        extension="logic",
        op_name="Not",
        signature=stys.FunctionType(input=[BOOL_T], output=[BOOL_T]),
    )
)


@dataclass
class Not(Command):
    a: Wire

    def incoming(self) -> list[Wire]:
        return [self.a]

    def num_out(self) -> int | None:
        return 1

    def op(self) -> Op:
        return NOT_OP


@dataclass
class DivMod(Command):
    a: Wire
    b: Wire

    def incoming(self) -> list[Wire]:
        return [self.a, self.b]

    def num_out(self) -> int | None:
        return 2

    def op(self) -> Op:
        return DummyOp(
            sops.CustomOp(
                parent=-1,
                extension="arithmetic.int",
                op_name="idivmod_u",
                signature=stys.FunctionType(input=[INT_T] * 2, output=[INT_T] * 2),
                args=[ARG_5, ARG_5],
            )
        )


@dataclass
class MakeTuple(Command):
    types: list[stys.Type]
    wires: list[Wire]

    def incoming(self) -> list[Wire]:
        return self.wires

    def num_out(self) -> int | None:
        return 1

    def op(self) -> Op:
        return DummyOp(sops.MakeTuple(parent=-1, tys=self.types))


@dataclass
class UnpackTuple(Command):
    types: list[stys.Type]
    wire: Wire

    def incoming(self) -> list[Wire]:
        return [self.wire]

    def num_out(self) -> int | None:
        return len(self.types)

    def op(self) -> Op:
        return DummyOp(sops.UnpackTuple(parent=-1, tys=self.types))


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
    h = Hugr(DummyOp(sops.DFG(parent=-1)))

    nodes = [h.add_node(NOT_OP) for _ in range(3)]
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

    new_n = h.add_node(NOT_OP)
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
    nt = h.add_op(NOT_OP, a)
    h.set_outputs(nt)

    _validate(h.hugr)


def test_tuple():
    row = [BOOL_T, QB_T]
    h = Dfg.endo(row)
    a, b = h.inputs()
    t = h.add(MakeTuple(row, [a, b]))
    a, b = h.add(UnpackTuple(row, t))
    h.set_outputs(a, b)

    _validate(h.hugr)

    h1 = Dfg.endo(row)
    a, b = h1.inputs()
    mt = h1.add_op(DummyOp(sops.MakeTuple(parent=-1, tys=row)), a, b)
    a, b = h1.add_op(DummyOp(sops.UnpackTuple(parent=-1, tys=row)), mt)[0, 1]
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

    new_h = Hugr(DummyOp(sops.DFG(parent=-1)))
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
