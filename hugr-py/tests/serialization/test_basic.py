from hugr.serialization import SerialHugr
from hugr import get_serialisation_version


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": get_serialisation_version(),
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
    }
