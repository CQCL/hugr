from pathlib import Path

import pytest
import semver

from hugr import ops, tys
from hugr.build.dfg import Dfg
from hugr.build.function import Module
from hugr.envelope import EnvelopeConfig, EnvelopeFormat
from hugr.hugr.node_port import Node
from hugr.package import Package

from .conftest import QUANTUM_EXT, H


@pytest.fixture
def package() -> Package:
    mod = Module()
    f_id = mod.define_function("id", [tys.Qubit])
    f_id.set_outputs(f_id.input_node[0])

    mod2 = Module()
    f_id_decl = mod2.declare_function(
        "id", tys.PolyFuncType([], tys.FunctionType([tys.Qubit], [tys.Qubit]))
    )
    f_main = mod2.define_main([tys.Qubit])
    q = f_main.input_node[0]
    call = f_main.call(f_id_decl, q)
    f_main.set_outputs(call)
    return Package([mod.hugr, mod2.hugr])


def test_envelope(package: Package):
    # Binary compression roundtrip
    for format in [
        EnvelopeFormat.JSON,
        EnvelopeFormat.MODEL,
        EnvelopeFormat.MODEL_WITH_EXTS,
    ]:
        for compression in [None, 0]:
            encoded = package.to_bytes(EnvelopeConfig(format=format, zstd=compression))
            decoded = Package.from_bytes(encoded)
            assert decoded == package

    # String roundtrip
    encoded_str = package.to_str(EnvelopeConfig.TEXT)
    decoded = Package.from_str(encoded_str)
    assert decoded == package


def test_model(package: Package):
    model_pkg = package.to_model()

    # This value is statically defined in the rust bindings.
    assert model_pkg.version >= semver.Version(major=1, minor=0, patch=0)


def test_legacy_funcdefn():
    p = Path(__file__).parents[2] / "resources" / "test" / "hugr-no-visibility.hugr"
    try:
        with p.open("rb") as f:
            pkg_bytes = f.read()
    except FileNotFoundError:
        pytest.skip("Missing test file")
    decoded = Package.from_bytes(pkg_bytes)
    h = decoded.modules[0]
    op1 = h[Node(1)].op
    assert isinstance(op1, ops.FuncDecl)
    assert op1.visibility == "Public"
    op2 = h[Node(2)].op
    assert isinstance(op2, ops.FuncDefn)
    assert op2.visibility == "Private"


def test_model_import_with_ext():
    dfg = Dfg(tys.Qubit)
    h_outs = dfg.add_op(H, dfg.inputs()[0])
    dfg.set_outputs(h_outs)
    pkg = Package(modules=[dfg.hugr], extensions=[QUANTUM_EXT])
    data = pkg.to_bytes(config=EnvelopeConfig.BINARY)
    pkg1 = Package.from_bytes(data)
    data1 = pkg1.to_bytes(config=EnvelopeConfig.BINARY)
    pkg2 = Package.from_bytes(data1)
    assert pkg2.modules[0].num_nodes() == 8
