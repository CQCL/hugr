from hugr.serialization import SerialHugr
from hugr.hugr import Dfg, Hugr, DummyOp
import hugr.serialization.tys as stys
import hugr.serialization.ops as sops

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

DIV_OP = DummyOp(
    sops.CustomOp(
        parent=-1,
        extension="arithmetic.int",
        op_name="idivmod_u",
        signature=stys.FunctionType(input=[INT_T] * 2, output=[INT_T] * 2),
        args=[ARG_5, ARG_5],
    )
)


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": "v1",
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
    }


def _validate(h: Hugr, mermaid: bool = False):
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w") as f:
        f.write(h.to_serial().to_json())
        f.flush()
        # TODO point to built hugr binary
        cmd = ["cargo", "run", "--"]

        if mermaid:
            cmd.append("--mermaid")
        subprocess.run(cmd + [f.name], check=True)


def test_simple_id():
    h = Dfg.endo([QB_T] * 2)
    a, b = h.inputs()
    h.set_outputs([a, b])

    _validate(h.hugr)


def test_multiport():
    h = Dfg([BOOL_T], [BOOL_T] * 2)
    (a,) = h.inputs()
    h.set_outputs([a, a])

    _validate(h.hugr)


def test_add_op():
    h = Dfg.endo([BOOL_T])
    (a,) = h.inputs()
    nt = h.add_op(NOT_OP, [a])
    h.set_outputs([nt])

    _validate(h.hugr)


def test_tuple():
    row = [BOOL_T, QB_T]
    h = Dfg.endo(row)
    a, b = h.inputs()
    t = h.make_tuple([a, b], row)
    a, b = h.split_tuple(t, row)
    h.set_outputs([a, b])

    _validate(h.hugr)


def test_multi_out():
    h = Dfg([INT_T] * 2, [INT_T] * 2)
    a, b = h.inputs()
    a, b = h.add_op(DIV_OP, [a, b])[:2]
    h.set_outputs([a, b])

    _validate(h.hugr, True)
