from hugr import Hugr, ops, tys
from hugr._serialization.serial_hugr import SerialHugr, serialization_version
from hugr.build.dfg import Dfg
from hugr.build.function import Module


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": serialization_version(),
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
        "entrypoint": None,
    }


def test_order_links():
    dfg = Dfg(tys.Bool)
    inp_0 = dfg.input_node.out(0)
    inp_order = dfg.input_node.out(-1)
    out_0 = dfg.output_node.inp(0)
    out_1 = dfg.output_node.inp(1)
    out_order = dfg.output_node.inp(-1)

    dfg.hugr.add_link(inp_0, out_0)
    dfg.hugr.add_link(inp_0, out_1)
    assert list(dfg.hugr.outgoing_links(dfg.input_node)) == [
        (inp_0, [out_0, out_1]),
    ]
    assert list(dfg.hugr.incoming_links(dfg.output_node)) == [
        (out_0, [inp_0]),
        (out_1, [inp_0]),
    ]

    # Now add an order link
    dfg.hugr.add_order_link(dfg.input_node, dfg.output_node)
    assert list(dfg.hugr.incoming_order_links(dfg.output_node)) == [dfg.input_node]
    assert list(dfg.hugr.outgoing_order_links(dfg.input_node)) == [dfg.output_node]
    assert list(dfg.hugr.outgoing_links(dfg.input_node)) == [
        (inp_0, [out_0, out_1]),
        (inp_order, [out_order]),
    ]
    assert list(dfg.hugr.incoming_links(dfg.output_node)) == [
        (out_0, [inp_0]),
        (out_1, [inp_0]),
        (out_order, [inp_order]),
    ]


def test_children():
    mod = Module()
    mod.declare_function("foo", tys.PolyFuncType([], tys.FunctionType.empty()))

    h = mod.hugr
    assert len(h.children()) == 1

    h2 = Hugr.from_str(h.to_str())

    assert len(h2.children()) == 1


def test_entrypoint():
    dfg = Dfg(tys.Bool)
    noop = dfg.add_op(ops.Noop(tys.Bool), *dfg.inputs())
    dfg.set_outputs(noop)

    h = dfg.hugr
    assert len(h.children()) == 3
    assert h[noop].parent == dfg.to_node()

    func = h[dfg].parent
    assert h[func].op == ops.FuncDefn(
        f_name="main", inputs=[tys.Bool], _outputs=[tys.Bool]
    )
    assert h[func].parent == h.module_root

    # Do a roundtrip, and test all again
    h2 = Hugr.from_str(h.to_str())

    dfg = h2.entrypoint
    assert h2[dfg].op == ops.DFG(inputs=[tys.Bool], _outputs=[tys.Bool])
    assert len(h2.children()) == 3

    noop = h2.children(dfg)[2]
    assert h2[noop].parent == dfg.to_node()

    func = h2[dfg].parent
    assert h2[func].op == ops.FuncDefn(
        f_name="main", inputs=[tys.Bool], _outputs=[tys.Bool]
    )
    assert h2[func].parent == h2.module_root
    assert h2[h2.module_root].op == ops.Module()
