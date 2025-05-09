import pytest

from hugr import tys
from hugr.build.tracked_dfg import TrackedDfg
from hugr.package import Package
from hugr.std.float import FLOAT_T, FloatVal
from hugr.std.logic import Not

from .conftest import CX, QUANTUM_EXT, H, Measure, Rz, validate


def test_track_wire():
    dfg = TrackedDfg(tys.Bool, tys.Unit)
    inds = dfg.track_inputs()
    assert inds == [0, 1]
    assert dfg.tracked_wire(inds[0]) == dfg.inputs()[0]
    with pytest.raises(IndexError, match="Index 2 not a tracked wire."):
        dfg.tracked_wire(2)
    w1 = dfg.tracked_wire(inds[1])
    w1_removed = dfg.untrack_wire(inds[1])
    assert w1 == w1_removed
    with pytest.raises(IndexError, match="Index 1 not a tracked wire."):
        dfg.tracked_wire(inds[1])

    dfg.set_indexed_outputs(0)

    validate(dfg.hugr)


def simple_circuit(n_qb: int, float_in: int = 0) -> TrackedDfg:
    in_tys = [tys.Qubit] * n_qb + [FLOAT_T] * float_in
    return TrackedDfg(*in_tys, track_inputs=True)


def test_simple_circuit():
    circ = simple_circuit(2)
    circ.add(H(0))
    [_h, cx_n] = circ.extend(H(0), CX(0, 1))

    circ.set_tracked_outputs()

    assert len(circ.hugr) == 10

    # all nodes connected to output
    out_ins = {
        out.node
        for _, outs in circ.hugr.incoming_links(circ.output_node)
        for out in outs
    }
    assert out_ins == {cx_n}
    validate(Package([circ.hugr], [QUANTUM_EXT]))


def test_complex_circuit():
    circ = simple_circuit(2)
    fl = circ.load(FloatVal(0.5))

    circ.extend(H(0), Rz(0, fl))
    [_m0, m1] = circ.extend(*(Measure(i) for i in range(2)))

    m_idx = circ.track_wire(m1[1])  # track the bool out
    assert m_idx == 2
    circ.add(Not(m_idx))

    circ.set_tracked_outputs()

    assert len(circ.hugr) == 14

    assert circ._output_op().types == [tys.Qubit, tys.Qubit, tys.Bool]

    validate(Package([circ.hugr], [QUANTUM_EXT]))
