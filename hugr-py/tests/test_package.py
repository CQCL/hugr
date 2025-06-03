from hugr import tys
from hugr.build import Dfg
from hugr.build.function import Module
from hugr.ext import Extension, FixedHugr, OpDef, OpDefSig, Version
from hugr.package import (
    FuncDeclPointer,
    FuncDefnPointer,
    ModulePointer,
    Package,
    PackagePointer,
)

from .conftest import validate


def test_package():
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
    validate(package)

    p = PackagePointer(package)
    assert p.package == package

    m = ModulePointer(package, 1)
    assert m.module == mod2.hugr

    f = FuncDeclPointer(package, 1, f_id_decl)
    assert f.func_decl == mod2.hugr[f_id_decl].op

    f = FuncDefnPointer(package, 0, f_id.to_node())

    assert f.func_defn == mod.hugr[f_id.to_node()].op

    main = m.to_executable_package()
    assert main.entry_point_node == f_main.to_node()


def test_lower_func():
    hugr = Dfg(tys.Qubit)
    hugr.set_outputs(hugr.input_node[0])

    ext = Extension("dummy", Version(0, 1, 0))
    ext.add_op_def(
        OpDef(
            "dummy_op",
            OpDefSig(tys.FunctionType.endo([tys.Qubit])),
            lower_funcs=[FixedHugr([], hugr.hugr)],
        )
    )

    pkg = Package([hugr.hugr], [ext])

    validate(pkg)
