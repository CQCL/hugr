from hugr.serialization import SerialHugr
from hugr import get_serialisation_version


def test_empty():
    h = SerialHugr(nodes=[], edges=[], hierarchy=[])
    assert h.model_dump() == {
        "version": get_serialisation_version(),
        "nodes": [],
        "edges": [],
        "hierarchy": [],
        "metadata": None,
        "encoder": None,
    }
