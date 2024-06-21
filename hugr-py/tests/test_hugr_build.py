from __future__ import annotations
from dataclasses import dataclass, field
import subprocess
import os
import pathlib
from hugr._hugr import Hugr, Node, Wire, _SubPort
from hugr._dfg import Dfg, _ancestral_sibling
from hugr._ops import Custom, Command
import hugr._ops as ops
from hugr.serialization import SerialHugr
import hugr._tys as tys
import hugr._val as val
import pytest
import json


def int_t(width: int) -> tys.Opaque:
    return tys.Opaque(
        extension="arithmetic.int.types",
        id="int",
        args=[tys.BoundedNatArg(n=width)],
        bound=tys.TypeBound.Eq,
    )


INT_T = int_t(5)


@dataclass
class IntVal(val.ExtensionValue):
    v: int

    def to_value(self) -> val.Extension:
        return val.Extension("int", INT_T, self.v)


@dataclass
class LogicOps(Custom):
    extension: tys.ExtensionId = "logic"


# TODO get from YAML
@dataclass
class NotDef(LogicOps):
    num_out: int | None = 1
    op_name: str = "Not"
    signature: tys.FunctionType = tys.FunctionType.endo([tys.Bool])

    def __call__(self, a: Wire) -> Command:
        return super().__call__(a)


Not = NotDef()


@dataclass
class QuantumOps(Custom):
    extension: tys.ExtensionId = "tket2.quantum"


@dataclass
class OneQbGate(QuantumOps):
    op_name: str
    num_out: int | None = 1
    signature: tys.FunctionType = tys.FunctionType.endo([tys.Qubit])

    def __call__(self, q: Wire) -> Command:
        return super().__call__(q)


H = OneQbGate("H")


@dataclass
class MeasureDef(QuantumOps):
    op_name: str = "Measure"
    num_out: int | None = 2
    signature: tys.FunctionType = tys.FunctionType([tys.Qubit], [tys.Qubit, tys.Bool])

    def __call__(self, q: Wire) -> Command:
        return super().__call__(q)


Measure = MeasureDef()


@dataclass
class IntOps(Custom):
    extension: tys.ExtensionId = "arithmetic.int"


ARG_5 = tys.BoundedNatArg(n=5)


@dataclass
class DivModDef(IntOps):
    num_out: int | None = 2
    extension: tys.ExtensionId = "arithmetic.int"
    op_name: str = "idivmod_u"
    signature: tys.FunctionType = field(
        default_factory=lambda: tys.FunctionType(input=[INT_T] * 2, output=[INT_T] * 2)
    )
    args: list[tys.TypeArg] = field(default_factory=lambda: [ARG_5, ARG_5])


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
    h = Hugr(ops.DFG([]))

    nodes = [h.add_node(Not) for _ in range(3)]
    assert len(h) == 4

    h.add_link(nodes[0].out(0), nodes[1].inp(0))
    assert h.children() == nodes

    assert h.num_outgoing(nodes[0]) == 1
    assert h.num_incoming(nodes[1]) == 1

    assert h.delete_node(nodes[1]) is not None
    assert h._nodes[nodes[1].idx] is None
    assert nodes[1] not in h.children(h.root)

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


def simple_id() -> Dfg:
    h = Dfg(tys.Qubit, tys.Qubit)
    a, b = h.inputs()
    h.set_outputs(a, b)
    return h


def test_simple_id():
    _validate(simple_id().hugr)


def test_multiport():
    h = Dfg(tys.Bool)
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
    h = Dfg(tys.Bool)
    (a,) = h.inputs()
    nt = h.add_op(Not, a)
    h.set_outputs(nt)

    _validate(h.hugr)


def test_tuple():
    row = [tys.Bool, tys.Qubit]
    h = Dfg(*row)
    a, b = h.inputs()
    t = h.add(ops.MakeTuple(a, b))
    a, b = h.add(ops.UnpackTuple(t))
    h.set_outputs(a, b)

    _validate(h.hugr)

    h1 = Dfg(*row)
    a, b = h1.inputs()
    mt = h1.add_op(ops.MakeTuple, a, b)
    a, b = h1.add_op(ops.UnpackTuple, mt)[0, 1]
    h1.set_outputs(a, b)

    assert h.hugr.to_serial() == h1.hugr.to_serial()


def test_multi_out():
    h = Dfg(INT_T, INT_T)
    a, b = h.inputs()
    a, b = h.add(DivMod(a, b))
    h.set_outputs(a, b)
    _validate(h.hugr)


def test_insert():
    h1 = Dfg(tys.Bool)
    (a1,) = h1.inputs()
    nt = h1.add(Not(a1))
    h1.set_outputs(nt)

    assert len(h1.hugr) == 4

    new_h = Hugr(ops.DFG([]))
    mapping = h1.hugr.insert_hugr(new_h, h1.hugr.root)
    assert mapping == {new_h.root: Node(4)}


def test_insert_nested():
    h1 = Dfg(tys.Bool)
    (a1,) = h1.inputs()
    nt = h1.add(Not(a1))
    h1.set_outputs(nt)

    h = Dfg(tys.Bool)
    (a,) = h.inputs()
    nested = h.insert_nested(h1, a)
    h.set_outputs(nested)
    assert len(h.hugr.children(nested)) == 3
    _validate(h.hugr)


def test_build_nested():
    def _nested_nop(dfg: Dfg):
        (a1,) = dfg.inputs()
        nt = dfg.add(Not(a1))
        dfg.set_outputs(nt)

    h = Dfg(tys.Bool)
    (a,) = h.inputs()
    nested = h.add_nested(a)

    _nested_nop(nested)
    assert len(h.hugr.children(nested)) == 3
    h.set_outputs(nested)

    _validate(h.hugr)


def test_build_inter_graph():
    h = Dfg(tys.Bool, tys.Bool)
    (a, b) = h.inputs()
    nested = h.add_nested()

    nt = nested.add(Not(a))
    nested.set_outputs(nt)

    h.set_outputs(nested, b)

    _validate(h.hugr)

    assert _SubPort(h.input_node.out(-1)) in h.hugr._links
    assert h.hugr.num_outgoing(h.input_node) == 2  # doesn't count state order
    assert len(list(h.hugr.outgoing_order_links(h.input_node))) == 1
    assert len(list(h.hugr.incoming_order_links(nested))) == 1
    assert len(list(h.hugr.incoming_order_links(h.output_node))) == 0


def test_ancestral_sibling():
    h = Dfg(tys.Bool)
    (a,) = h.inputs()
    nested = h.add_nested()

    nt = nested.add(Not(a))

    assert _ancestral_sibling(h.hugr, h.input_node, nt) == nested.parent_node


@pytest.mark.parametrize(
    "val",
    [
        val.Function(simple_id().hugr),
        val.Sum(1, tys.Sum([[INT_T], [tys.Bool, INT_T]]), [IntVal(34)]),
        val.Tuple([val.TRUE, IntVal(23)]),
    ],
)
def test_vals(val: val.Value):
    d = Dfg()
    d.set_outputs(d.load(val))

    _validate(d.hugr)
