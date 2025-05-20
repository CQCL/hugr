from __future__ import annotations

import pytest

import hugr.ops as ops
import hugr.tys as tys
import hugr.val as val
from hugr.build.dfg import Dfg, _ancestral_sibling
from hugr.build.function import Module
from hugr.hugr import Hugr
from hugr.hugr.node_port import Node, _SubPort
from hugr.ops import NoConcreteFunc
from hugr.std.int import INT_T, DivMod, IntVal
from hugr.std.logic import Not

from .conftest import validate


def test_stable_indices():
    h = Hugr(ops.DFG([]))

    nodes = [h.add_node(Not, num_outs=1) for _ in range(3)]
    assert len(h) == 8
    assert len(list(h.descendants())) == 4
    assert list(iter(h)) == [Node(i) for i in range(8)]
    assert all(data is not None for node, data in h.nodes())

    assert len(list(nodes[0].outputs())) == 1
    assert list(nodes[0]) == list(nodes[0].outputs())

    h.add_link(nodes[0].out(0), nodes[1].inp(0))
    assert h.children() == nodes

    assert h.num_outgoing(nodes[0]) == 1
    assert h.num_incoming(nodes[1]) == 1

    assert nodes[1] in h.children(h.entrypoint)
    assert h.delete_node(nodes[1]) is not None
    assert h._nodes[nodes[1].idx] is None
    assert nodes[1] not in h.children(h.entrypoint)

    assert len(h) == 7
    assert len(h._nodes) == 8
    assert h._free_nodes == [nodes[1]]

    assert h.num_outgoing(nodes[0]) == 0
    assert h.num_incoming(nodes[1]) == 0

    with pytest.raises(KeyError):
        _ = h[nodes[1]]
    with pytest.raises(KeyError):
        _ = h[Node(46)]

    new_n = h.add_node(Not)
    assert new_n == nodes[1]

    assert len(h) == 8
    assert h._free_nodes == []
    assert list(iter(h)) == [Node(i) for i in range(len(h))]
    assert all(data is not None for node, data in h.nodes())


def simple_id() -> Dfg:
    h = Dfg(tys.Qubit, tys.Qubit)
    a, b = h.inputs()
    h.set_outputs(a, b)
    return h


def test_simple_id(snapshot):
    hugr = simple_id().hugr
    validate(hugr, snap=snapshot)


def test_metadata(snapshot):
    h = Dfg(tys.Bool)
    h.metadata["name"] = "simple_id"

    (b,) = h.inputs()
    b = h.add_op(Not, b, metadata={"name": "not"})

    h.set_outputs(b)
    validate(h.hugr, snap=snapshot)


def test_multiport(snapshot):
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
    validate(h.hugr, snap=snapshot)


def test_add_op(snapshot):
    h = Dfg(tys.Bool)
    (a,) = h.inputs()
    nt = h.add_op(Not, a)
    h.set_outputs(nt)

    validate(h.hugr, snap=snapshot)


def test_tuple(snapshot):
    row = [tys.Bool, tys.Qubit]
    h = Dfg(*row)
    a, b = h.inputs()
    t = h.add(ops.MakeTuple()(a, b))
    a, b = h.add(ops.UnpackTuple()(t))
    h.set_outputs(a, b)

    validate(h.hugr, snap=snapshot)

    h1 = Dfg(*row)
    a, b = h1.inputs()
    mt = h1.add_op(ops.MakeTuple(), a, b)
    a, b = h1.add_op(ops.UnpackTuple(), mt)[0, 1]
    h1.set_outputs(a, b)

    assert h.hugr._to_serial() == h1.hugr._to_serial()


def test_multi_out(snapshot):
    h = Dfg(INT_T, INT_T)
    a, b = h.inputs()
    a, b = h.add(DivMod(a, b))
    h.set_outputs(a, b)
    validate(h.hugr, snap=snapshot)


def test_insert():
    h1 = Dfg(tys.Bool)
    (a1,) = h1.inputs()
    nt = h1.add(Not(a1))
    h1.set_outputs(nt)

    assert len(h1.hugr) == 8

    new_h = Hugr(ops.DFG([]))
    mapping = h1.hugr.insert_hugr(new_h, h1.hugr.entrypoint)
    assert mapping == {new_h.entrypoint: Node(8)}


def test_insert_nested(snapshot):
    h1 = Dfg(tys.Bool)
    (a1,) = h1.inputs()
    nt = h1.add(Not(a1))
    h1.set_outputs(nt)

    h = Dfg(tys.Bool)
    (a,) = h.inputs()
    nested = h.insert_nested(h1, a)
    h.set_outputs(nested)
    assert len(h.hugr.children(nested)) == 3
    validate(h.hugr, snap=snapshot)


def test_build_nested(snapshot):
    h = Dfg(tys.Bool)
    (a,) = h.inputs()

    with h.add_nested(a) as nested:
        (a1,) = nested.inputs()
        nt = nested.add(Not(a1))
        nested.set_outputs(nt)

    assert len(h.hugr.children(nested)) == 3
    h.set_outputs(nested)

    validate(h.hugr, snap=snapshot)


def test_build_inter_graph(snapshot):
    h = Dfg(tys.Bool, tys.Bool)
    (a, b) = h.inputs()
    with h.add_nested() as nested:
        nt = nested.add(Not(a))
        nested.set_outputs(nt)

    h.set_outputs(nested, b)

    validate(h.hugr, snap=snapshot)

    assert _SubPort(h.input_node.out(-1)) in h.hugr._links
    assert h.hugr.num_outgoing(h.input_node) == 2  # doesn't count state order
    assert len(list(h.hugr.outgoing_order_links(h.input_node))) == 1
    assert len(list(h.hugr.incoming_order_links(nested))) == 1
    assert len(list(h.hugr.incoming_order_links(h.output_node))) == 0


def test_ancestral_sibling():
    h = Dfg(tys.Bool)
    (a,) = h.inputs()
    with h.add_nested() as nested:
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
        call = f_main.add(ops.CallIndirect()(load, q))

    f_main.set_outputs(call)

    validate(mod.hugr)


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
        call = f_main.add(ops.CallIndirect()(load, q))
    f_main.set_outputs(call)

    validate(mod.hugr)


def test_recursive_function(snapshot) -> None:
    mod = Module()

    f_recursive = mod.define_function("recurse", [tys.Qubit])
    f_recursive.declare_outputs([tys.Qubit])
    call = f_recursive.call(f_recursive, f_recursive.input_node[0])
    f_recursive.set_outputs(call)

    validate(mod.hugr, snap=snapshot)


def test_invalid_recursive_function() -> None:
    mod = Module()

    f_recursive = mod.define_function("recurse", [tys.Bool], [tys.Qubit])
    f_recursive.call(f_recursive, f_recursive.input_node[0])

    with pytest.raises(ValueError, match="The function has fixed output type"):
        f_recursive.set_outputs(f_recursive.input_node[0])


def test_higher_order(snapshot) -> None:
    noop_fn = Dfg(tys.Qubit)
    noop_fn.set_outputs(noop_fn.add(ops.Noop()(noop_fn.input_node[0])))

    d = Dfg(tys.Qubit)
    (q,) = d.inputs()
    f_val = d.load(val.Function(noop_fn.hugr))
    call = d.add(ops.CallIndirect()(f_val, q))[0]
    d.add_state_order(d.input_node, f_val)
    d.set_outputs(call)

    validate(d.hugr, snap=snapshot)


def test_state_order() -> None:
    mod = Module()
    f_id = mod.define_function("id", [tys.Bool])
    f_id.set_outputs(f_id.input_node[0])

    f_main = mod.define_main([tys.Bool])
    b = f_main.input_node[0]
    call1 = f_main.call(f_id, b)
    f_main.add_state_order(call1, f_main.output_node)
    # implicit discard of bool to test state order port logic
    f_main.set_outputs()
    validate(mod.hugr)


def test_alias() -> None:
    mod = Module()
    _dfn = mod.add_alias_defn("my_int", INT_T)
    _dcl = mod.add_alias_decl("my_bool", tys.TypeBound.Copyable)

    validate(mod.hugr)


# https://github.com/CQCL/hugr/issues/1625
def test_dfg_unpack() -> None:
    dfg = Dfg(tys.Tuple(tys.Bool, tys.Bool))
    bool1, _unused_bool2 = dfg.add_op(ops.UnpackTuple(), *dfg.inputs())
    cond = dfg.add_conditional(bool1)
    with cond.add_case(0) as case:
        case.set_outputs(bool1)
    with cond.add_case(1) as case:
        case.set_outputs(bool1)
    dfg.set_outputs(*cond.outputs())

    validate(dfg.hugr)


def test_option() -> None:
    dfg = Dfg(tys.Bool)
    b = dfg.inputs()[0]

    dfg.add_op(ops.Some(tys.Bool), b)

    dfg.set_outputs(b)

    validate(dfg.hugr)
