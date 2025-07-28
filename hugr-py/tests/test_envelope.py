from hugr import tys
from hugr.build.function import Module
from hugr.envelope import EnvelopeConfig, EnvelopeFormat
from hugr.package import Package


def test_envelope():
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
    package = Package([mod.hugr, mod2.hugr])

    # Binary compression roundtrip
    for format in [EnvelopeFormat.JSON]:
        for compression in [None, 0]:
            encoded = package.to_bytes(EnvelopeConfig(format=format, zstd=compression))
            decoded = Package.from_bytes(encoded)
            assert decoded == package

    # String roundtrip
    encoded = package.to_str(EnvelopeConfig.TEXT)
    decoded = Package.from_str(encoded)
    assert decoded == package
