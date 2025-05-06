from dataclasses import dataclass

import pytest

from hugr import ext, ops, tys
from hugr.build.dfg import Dfg
from hugr.hugr import Hugr, Node
from hugr.ops import AsExtOp, Custom, ExtOp
from hugr.package import Package
from hugr.std.float import FLOAT_T
from hugr.std.float import FLOAT_TYPES_EXTENSION as FLOAT_EXT
from hugr.std.int import INT_OPS_EXTENSION, INT_TYPES_EXTENSION, DivMod, int_t
from hugr.std.logic import EXTENSION as LOGIC_EXT
from hugr.std.logic import Not

from .conftest import CX, QUANTUM_EXT, H, Measure, Rz, validate

STRINGLY_EXT = ext.Extension("my_extension", ext.Version(0, 0, 0))
_STRINGLY_DEF = STRINGLY_EXT.add_op_def(
    ext.OpDef(
        "StringlyOp",
        signature=ext.OpDefSig(
            tys.PolyFuncType([tys.StringParam()], tys.FunctionType.endo([]))
        ),
    )
)


@dataclass
class StringlyOp(AsExtOp):
    tag: str

    def op_def(self) -> ext.OpDef:
        return STRINGLY_EXT.get_op("StringlyOp")

    def type_args(self) -> list[tys.TypeArg]:
        return [tys.StringArg(self.tag)]

    def cached_signature(self) -> tys.FunctionType | None:
        return tys.FunctionType.endo([])

    @classmethod
    def from_ext(cls, custom: ops.ExtOp) -> "StringlyOp":
        match custom:
            case ops.ExtOp(
                _op_def=_STRINGLY_DEF,
                args=[tys.StringArg(tag)],
            ):
                return cls(tag=tag)
            case _:
                msg = f"Invalid custom op: {custom}"
                raise AsExtOp.InvalidExtOp(msg)


def test_stringly_typed():
    dfg = Dfg()
    n = dfg.add(StringlyOp("world")())
    dfg.set_outputs()
    assert dfg.hugr[n].op == StringlyOp("world")
    validate(Package([dfg.hugr], [STRINGLY_EXT]))

    new_h = Hugr._from_serial(dfg.hugr._to_serial())

    assert isinstance(new_h[n].op, Custom)

    registry = ext.ExtensionRegistry()
    new_h.resolve_extensions(registry)

    # doesn't resolve without extension
    assert isinstance(new_h[n].op, Custom)

    registry.add_extension(STRINGLY_EXT)
    new_h.resolve_extensions(registry)

    assert isinstance(new_h[n].op, ExtOp)


def test_registry():
    reg = ext.ExtensionRegistry()
    reg.add_extension(LOGIC_EXT)
    assert reg.get_extension(LOGIC_EXT.name).name == LOGIC_EXT.name
    assert len(reg.extensions) == 1
    with pytest.raises(ext.ExtensionRegistry.ExtensionExists):
        reg.add_extension(LOGIC_EXT)

    with pytest.raises(ext.ExtensionRegistry.ExtensionNotFound):
        reg.get_extension("not_found")


@pytest.fixture
def registry() -> ext.ExtensionRegistry:
    reg = ext.ExtensionRegistry()
    reg.add_extension(LOGIC_EXT)
    reg.add_extension(QUANTUM_EXT)
    reg.add_extension(STRINGLY_EXT)
    reg.add_extension(INT_TYPES_EXTENSION)
    reg.add_extension(INT_OPS_EXTENSION)
    reg.add_extension(FLOAT_EXT)

    return reg


@pytest.mark.parametrize(
    "as_ext",
    [Not, DivMod, H, CX, Measure, Rz, StringlyOp("hello")],
)
def test_custom_op(as_ext: AsExtOp, registry: ext.ExtensionRegistry):
    ext_op = as_ext.ext_op

    assert ExtOp.from_ext(ext_op) == ext_op

    assert type(as_ext).from_ext(ext_op) == as_ext
    custom = as_ext._to_serial(Node(0)).deserialize()
    assert isinstance(custom, Custom)
    # ExtOp compared to Custom via `to_custom`
    assert custom.resolve(registry) == ext_op
    assert ext_op == as_ext
    assert as_ext == ext_op


def test_custom_bad_eq():
    assert Not != DivMod

    bad_custom_sig = Custom("Not", extension=LOGIC_EXT.name)  # empty signature

    assert Not != bad_custom_sig

    bad_custom_args = Custom(
        "Not",
        extension=LOGIC_EXT.name,
        signature=tys.FunctionType.endo([tys.Bool]),
        args=[tys.Bool.type_arg()],
    )

    assert Not != bad_custom_args


_LIST_T = STRINGLY_EXT.add_type_def(
    ext.TypeDef(
        "List",
        description="A list of elements.",
        params=[tys.TypeTypeParam(tys.TypeBound.Any)],
        bound=ext.FromParamsBound([0]),
    )
)

_BOOL_LIST_T = _LIST_T.instantiate([tys.Bool.type_arg()])


@pytest.mark.parametrize(
    "ext_t",
    [FLOAT_T, int_t(5), _BOOL_LIST_T],
)
def test_custom_type(ext_t: tys.ExtType, registry: ext.ExtensionRegistry):
    opaque = ext_t._to_serial().deserialize()
    assert isinstance(opaque, tys.Opaque)
    assert opaque.resolve(registry) == ext_t

    assert opaque.resolve(ext.ExtensionRegistry()) == opaque

    f_t = tys.FunctionType.endo([ext_t])
    f_t_opaque = f_t._to_serial().deserialize()
    assert isinstance(f_t_opaque.input[0], tys.Opaque)

    assert f_t_opaque.resolve(registry) == f_t
