from hugr import get_serialization_version
from hugr.serialization.serial_hugr import SerialHugr


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": get_serialization_version(),
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
    }
