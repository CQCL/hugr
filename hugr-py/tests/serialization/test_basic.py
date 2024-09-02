from hugr import Hugr, tys
from hugr._serialization.serial_hugr import SerialHugr, serialization_version
from hugr.build.function import Module


def test_empty():
    h = SerialHugr(nodes=[], edges=[])
    assert h.model_dump() == {
        "version": serialization_version(),
        "nodes": [],
        "edges": [],
        "metadata": None,
        "encoder": None,
    }


def test_children():
    mod = Module()
    mod.declare_function("foo", tys.PolyFuncType([], tys.FunctionType.empty()))

    h = mod.hugr
    assert len(h.children(h.root)) == 1

    h2 = Hugr.load_json(h.to_json())

    assert len(h2.children(h2.root)) == 1
