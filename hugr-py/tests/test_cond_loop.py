import pytest

from hugr import ops, tys, val
from hugr.build.cond_loop import Conditional, ConditionalError, TailLoop
from hugr.build.dfg import Dfg
from hugr.package import Package
from hugr.std.int import INT_T, IntVal

from .conftest import QUANTUM_EXT, H, Measure, validate

EITHER_T = tys.Either([tys.Qubit], [tys.Qubit, INT_T])


def build_cond(h: Conditional) -> None:
    with pytest.raises(ConditionalError, match="Case 2 out of possible range."):
        h.add_case(2)

    with h.add_case(0) as case0:
        q, b = case0.inputs()
        case0.set_outputs(q, b)

    with h.add_case(1) as case1:
        q, _i, b = case1.inputs()
        case1.set_outputs(q, b)


def test_cond() -> None:
    h = Conditional(EITHER_T, [tys.Bool])
    build_cond(h)
    validate(h.hugr)


def test_nested_cond() -> None:
    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    tagged_q = h.add(ops.Left(EITHER_T)(q))

    with h.add_conditional(tagged_q, h.load(val.TRUE)) as cond:
        build_cond(cond)

    h.set_outputs(*cond[:2])
    validate(h.hugr)

    # build then insert
    con = Conditional(EITHER_T, [tys.Bool])
    build_cond(con)

    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    tagged_q = h.add(ops.Left(EITHER_T)(q))
    cond_n = h.insert_conditional(con, tagged_q, h.load(val.TRUE))
    h.set_outputs(*cond_n[:2])
    validate(h.hugr)


def test_if_else() -> None:
    # apply an H if a bool is true.
    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    with h.add_if(h.load(val.TRUE), q) as if_:
        if_.set_outputs(if_.add(H(if_.input_node[0])))

    with if_.add_else() as else_:
        else_.set_outputs(else_.input_node[0])

    h.set_outputs(else_.conditional_node)

    validate(Package([h.hugr], [QUANTUM_EXT]))


def test_incomplete() -> None:
    def _build_incomplete():
        with Conditional(EITHER_T, [tys.Bool]) as c, c.add_case(0) as case0:
            q, b = case0.inputs()
            case0.set_outputs(q, b)

    with pytest.raises(
        ConditionalError, match="All cases must be added before exiting context."
    ):
        _build_incomplete()


def test_tail_loop() -> None:
    # apply H while measure is true
    def build_tl(tl: TailLoop) -> None:
        q, b = tl.add(Measure(tl.add(H(tl.input_node[0]))))[:]

        tl.set_loop_outputs(b, q)

    h = Dfg(tys.Qubit)
    (q,) = h.inputs()

    with h.add_tail_loop([], [q]) as tl:
        build_tl(tl)
    h.set_outputs(tl)

    validate(Package([h.hugr], [QUANTUM_EXT]))

    # build then insert
    tl = TailLoop([], [tys.Qubit])
    build_tl(tl)

    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    tl_n = h.insert_tail_loop(tl, [q], [])
    h.set_outputs(tl_n)
    validate(Package([h.hugr], [QUANTUM_EXT]))


def test_complex_tail_loop() -> None:
    h = Dfg(tys.Qubit)
    (q,) = h.inputs()

    # loop passes qubit to itself, and a bool as in-out
    with h.add_tail_loop([q], [h.load(val.TRUE)]) as tl:
        q, b = tl.inputs()

        # if b is true, return first variant (just qubit)
        with tl.add_if(b, q) as if_:
            (q,) = if_.inputs()
            tagged_q = if_.add(ops.Continue(EITHER_T)(q))
            if_.set_outputs(tagged_q)

        # else return second variant (qubit, int)
        with if_.add_else() as else_:
            (q,) = else_.inputs()
            tagged_q_i = else_.add(ops.Break(EITHER_T)(q, else_.load(IntVal(1))))
            else_.set_outputs(tagged_q_i)

        # finish with Sum output from if-else, and bool from inputs
        tl.set_loop_outputs(else_.conditional_node, b)

    # loop returns [qubit, int, bool]
    h.set_outputs(*tl[:3])

    validate(h.hugr)


def test_conditional_bug() -> None:
    # bug with case ordering https://github.com/CQCL/hugr/issues/1596
    cond = Conditional(tys.Either([tys.USize()], [tys.Unit]), [])
    with cond.add_case(1) as case:
        case.set_outputs()
    with cond.add_case(0) as case:
        case.set_outputs()
    validate(cond.hugr)
