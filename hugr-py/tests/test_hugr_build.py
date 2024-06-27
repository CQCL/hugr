from __future__ import annotations
from hugr.node_port import Node, _SubPort

from hugr.hugr import Hugr
from hugr.dfg import Dfg, _ancestral_sibling
from hugr.ops import NoConcreteFunc
import hugr.ops as ops
import hugr.tys as tys
import hugr.val as val
from hugr.function import Module
import pytest

from .conftest import Not, INT_T, IntVal, validate, DivMod


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
    validate(simple_id().hugr)


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
    validate(h.hugr)


def test_add_op():
    h = Dfg(tys.Bool)
    (a,) = h.inputs()
    nt = h.add_op(Not, a)
    h.set_outputs(nt)

    validate(h.hugr)


def test_tuple():
    row = [tys.Bool, tys.Qubit]
    h = Dfg(*row)
    a, b = h.inputs()
    t = h.add(ops.MakeTuple(a, b))
    a, b = h.add(ops.UnpackTuple(t))
    h.set_outputs(a, b)

    validate(h.hugr)

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
    validate(h.hugr)


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
    validate(h.hugr)


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

    validate(h.hugr)


def test_build_inter_graph():
    h = Dfg(tys.Bool, tys.Bool)
    (a, b) = h.inputs()
    nested = h.add_nested()

    nt = nested.add(Not(a))
    nested.set_outputs(nt)

    h.set_outputs(nested, b)

    validate(h.hugr)

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
        val.Sum(1, tys.Sum([[INT_T], [tys.Bool, INT_T]]), [val.TRUE, IntVal(34)]),
        val.Tuple(val.TRUE, IntVal(23)),
    ],
)
def test_vals(val: val.Value):
    d = Dfg()
    d.set_outputs(d.load(val))

    validate(d.hugr)


@pytest.mark.parametrize("direct_call", [True, False])
def test_poly_function(direct_call: bool) -> None:
    mod = Module()
    f_id = mod.declare_function(
        "id",
        tys.PolyFuncType(
            [tys.TypeTypeParam(tys.TypeBound.Any)],
            tys.FunctionType.endo([tys.Variable(0, tys.TypeBound.Any)]),
        ),
    )

    f_main = mod.define_main([tys.Qubit])
    q = f_main.input_node[0]
    # for now concrete instantiations have to be provided.
    instantiation = tys.FunctionType.endo([tys.Qubit])
    type_args = [tys.Qubit.type_arg()]
    if direct_call:
        with pytest.raises(NoConcreteFunc, match="Missing instantiation"):
            f_main.call(f_id, q)
        call = f_main.call(f_id, q, instantiation=instantiation, type_args=type_args)
    else:
        with pytest.raises(NoConcreteFunc, match="Missing instantiation"):
            f_main.load_function(f_id)
        load = f_main.load_function(
            f_id, instantiation=instantiation, type_args=type_args
        )
        call = f_main.add(ops.CallIndirect(load, q))

    f_main.set_outputs(call)

    validate(mod.hugr, True)


@pytest.mark.parametrize("direct_call", [True, False])
def test_mono_function(direct_call: bool) -> None:
    mod = Module()
    f_id = mod.define_function("id", [tys.Qubit])
    f_id.set_outputs(f_id.input_node[0])

    f_main = mod.define_main([tys.Qubit])
    q = f_main.input_node[0]
    # monomorphic functions don't need instantiation specified
    if direct_call:
        call = f_main.call(f_id, q)
    else:
        load = f_main.load_function(f_id)
        call = f_main.add(ops.CallIndirect(load, q))
    f_main.set_outputs(call)

    validate(mod.hugr)


def test_higher_order() -> None:
    noop_fn = Dfg(tys.Qubit)
    noop_fn.set_outputs(noop_fn.add(ops.Noop(noop_fn.input_node[0])))

    d = Dfg(tys.Qubit)
    (q,) = d.inputs()
    f_val = d.load(val.Function(noop_fn.hugr))
    call = d.add(ops.CallIndirect(f_val, q))[0]
    d.set_outputs(call)

    validate(d.hugr)


def test_lift() -> None:
    d = Dfg(tys.Qubit, extension_delta=["X"])
    (q,) = d.inputs()
    lift = d.add(ops.Lift("X")(q))
    d.set_outputs(lift)
    validate(d.hugr)


def test_alias() -> None:
    mod = Module()
    _dfn = mod.add_alias_defn("my_int", INT_T)
    _dcl = mod.add_alias_decl("my_bool", tys.TypeBound.Eq)

    validate(mod.hugr)
