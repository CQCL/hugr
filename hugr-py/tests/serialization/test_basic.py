from hugr.serialization import SerialHugr
from hugr.hugr import Dfg, Hugr, QB_T, BOOL_T


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": "v1",
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
    }


def _validate(h: Hugr):
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w") as f:
        f.write(h.to_serial().to_json())
        f.flush()
        # TODO point to built hugr binary
        subprocess.run(["cargo", "run", f.name], check=True)


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
