from hugr._cond_loop import Conditional
from hugr._dfg import Dfg
import hugr._tys as tys
import hugr._ops as ops
import pytest
from .test_hugr_build import INT_T, _validate, IntVal

SUM_T = tys.Sum([[tys.Qubit], [tys.Qubit, INT_T]])


def build_cond(h: Conditional) -> None:
    with pytest.raises(AssertionError):
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
    _validate(h.hugr)


def test_nested_cond() -> None:
    h = Dfg(tys.Qubit)
    (q,) = h.inputs()
    tagged_q = h.add(ops.Tag(0, SUM_T)(q))
    cond = h.add_conditional(tagged_q, h.add_load_const(IntVal(1)))
    build_cond(cond)
    h.set_outputs(*cond[:2])
    _validate(h.hugr)
