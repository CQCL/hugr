from hugr.serialization import SerialHugr


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": "v1",
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
    }
