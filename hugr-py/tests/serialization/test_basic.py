from hugr.serialization import SerialHugr
from hugr.hugr import Dfg, Hugr, DummyOp
import hugr.serialization.tys as stys
import hugr.serialization.ops as sops

BOOL_T = stys.Type(stys.SumType(stys.UnitSum(size=2)))
QB_T = stys.Type(stys.Qubit())


NOT_OP = DummyOp(
    # TODO get from YAML
    sops.CustomOp(
        parent=-1,
        extension="logic",
        op_name="Not",
        signature=stys.FunctionType(input=[BOOL_T], output=[BOOL_T]),
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
