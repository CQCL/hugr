from hugr import tys
from hugr.build.dfg import Dfg
from hugr.package import Package

from .conftest import QUANTUM_EXT, MeasureFree, QAlloc, validate


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


# https://github.com/CQCL/hugr/issues/2439
def test_order_unconnected(snapshot):
    dfg = Dfg(tys.Qubit)
    meas = dfg.add(MeasureFree(*dfg.inputs()))
    alloc = dfg.add_op(QAlloc)
    dfg.hugr.add_order_link(meas, alloc)
    dfg.set_outputs(alloc)

    validate(Package([dfg.hugr], [QUANTUM_EXT]), snap=snapshot)
