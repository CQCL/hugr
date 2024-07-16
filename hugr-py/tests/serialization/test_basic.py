from hugr import get_serialisation_version
from hugr.serialization.serial_hugr import SerialHugr


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": get_serialisation_version(),
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
    }
