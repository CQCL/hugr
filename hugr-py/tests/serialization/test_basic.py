from hugr._serialization.serial_hugr import SerialHugr, serialization_version


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": serialization_version(),
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
    }
