from hugr.cond_loop import Conditional, ConditionalError, TailLoop
from hugr.dfg import Dfg
import hugr.tys as tys
import hugr.ops as ops
import hugr.val as val
import pytest
from .conftest import INT_T, validate, IntVal, H, Measure

SUM_T = tys.Sum([[tys.Qubit], [tys.Qubit, INT_T]])


def build_cond(h: Conditional) -> None:
    with pytest.raises(ConditionalError, match="Case 2 out of possible range."):
        h.add_case(2)

    case0 = h.add_case(0)
    q, b = case0.inputs()
    case0.set_outputs(q, b)

    case1 = h.add_case(1)
    q, _i, b = case1.inputs()
    case1.set_outputs(q, b)


def test_cond() -> None:
    h = Conditional(SUM_T, [tys.Bool])
    build_cond(h)
    validate(h.hugr)


def test_nested_cond() -> None:
    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    tagged_q = h.add(ops.Tag(0, SUM_T)(q))
    cond = h.add_conditional(tagged_q, h.load(val.TRUE))
    build_cond(cond)
    h.set_outputs(*cond[:2])
    validate(h.hugr)

    # build then insert
    con = Conditional(SUM_T, [tys.Bool])
    build_cond(con)

    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    tagged_q = h.add(ops.Tag(0, SUM_T)(q))
    cond_n = h.insert_conditional(con, tagged_q, h.load(val.TRUE))
    h.set_outputs(*cond_n[:2])
    validate(h.hugr)


def test_if_else() -> None:
    # apply an H if a bool is true.
    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    if_ = h.add_if(h.load(val.TRUE), q)

    if_.set_outputs(if_.add(H(if_.input_node[0])))

    else_ = if_.add_else()
    else_.set_outputs(else_.input_node[0])

    cond = else_.finish()
    h.set_outputs(cond)

    validate(h.hugr)


def test_tail_loop() -> None:
    # apply H while measure is true
    def build_tl(tl: TailLoop) -> None:
        q, b = tl.add(Measure(tl.add(H(tl.input_node[0]))))[:]

        tl.set_loop_outputs(b, q)

    h = Dfg(tys.Qubit)
    (q,) = h.inputs()

    tl = h.add_tail_loop([], [q])
    build_tl(tl)
    h.set_outputs(tl)

    validate(h.hugr)

    # build then insert
    tl = TailLoop([], [tys.Qubit])
    build_tl(tl)

    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    tl_n = h.insert_tail_loop(tl, q)
    h.set_outputs(tl_n)
    validate(h.hugr)


def test_complex_tail_loop() -> None:
    h = Dfg(tys.Qubit)
    (q,) = h.inputs()

    # loop passes qubit to itself, and a bool as in-out
    tl = h.add_tail_loop([q], [h.load(val.TRUE)])
    q, b = tl.inputs()

    # if b is true, return first variant (just qubit)
    if_ = tl.add_if(b, q)
    (q,) = if_.inputs()
    tagged_q = if_.add(ops.Tag(0, SUM_T)(q))
    if_.set_outputs(tagged_q)

    # else return second variant (qubit, int)
    else_ = if_.add_else()
    (q,) = else_.inputs()
    tagged_q_i = else_.add(ops.Tag(1, SUM_T)(q, else_.load(IntVal(1))))
    else_.set_outputs(tagged_q_i)

    # finish with Sum output from if-else, and bool from inputs
    tl.set_loop_outputs(else_.finish(), b)

    # loop returns [qubit, int, bool]
    h.set_outputs(*tl[:3])

    validate(h.hugr, True)

    # TODO rewrite with context managers
