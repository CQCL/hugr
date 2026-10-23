from pathlib import Path

import pytest
import semver

from hugr import cli, ops, tys
from hugr.build.cfg import Cfg
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


@pytest.mark.parametrize(
    "compression", [None, 0], ids=["compression:None", "compression:0"]
)
@pytest.mark.parametrize(
    "format",
    [
        EnvelopeFormat.JSON,
        EnvelopeFormat.MODEL,
        EnvelopeFormat.MODEL_WITH_EXTS,
    ],
)
def test_envelope_binary(
    package: Package, compression: int | None, format: EnvelopeFormat
):
    # Binary compression roundtrip
    encoded = package.to_bytes(EnvelopeConfig(format=format, zstd=compression))
    decoded = Package.from_bytes(encoded)
    assert decoded == package


def test_envelope_text(package: Package):
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


@pytest.mark.parametrize(
    ("sum_ty", "shared_outputs"),
    [
        (tys.Bool, []),
        (tys.Bool, [tys.Qubit]),
        (tys.Sum([[tys.Qubit], [tys.Qubit]]), []),
        (tys.Sum([[tys.Qubit], [tys.Qubit]]), [tys.Bool, tys.Qubit]),
    ],
    ids=["no-payload", "shared-linear", "variant-linear", "variant-and-shared"],
)
@pytest.mark.parametrize(
    "format",
    [
        EnvelopeFormat.JSON,
        EnvelopeFormat.MODEL,
        EnvelopeFormat.MODEL_WITH_EXTS,
        EnvelopeFormat.S_EXPRESSION_WITH_EXTS,
    ],
)
def test_cfg_block_output_roundtrip(
    sum_ty: tys.Sum, shared_outputs: list[tys.Type], format: EnvelopeFormat
):
    cfg = Cfg(sum_ty, *shared_outputs)
    entry = cfg.add_entry()
    entry.set_outputs(*entry.inputs())
    for branch in range(len(sum_ty.variant_rows)):
        successor = cfg.add_successor(entry[branch])
        successor.set_single_succ_outputs(*successor.inputs())
        cfg.branch_exit(successor[0])
    original = cfg.hugr.to_package()

    def block_signatures(package: Package):
        return [
            (data.op.inputs, data.op.sum_ty, data.op.other_outputs)
            for _, data in package.modules[0].nodes()
            if isinstance(data.op, ops.DataflowBlock)
        ]

    encoded = original.to_bytes(EnvelopeConfig(format=format))
    decoded = Package.from_bytes(encoded)
    assert block_signatures(decoded) == block_signatures(original)
    # JSON preserves the imported operation rather than recovering its signature
    # again through Rust model import, so it exposes any mismatch with the body.
    json_config = EnvelopeConfig(format=EnvelopeFormat.JSON)
    cli.validate(original.to_bytes(json_config))
    cli.validate(decoded.to_bytes(json_config))
