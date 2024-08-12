from dataclasses import dataclass

import pytest

from hugr import ext, ops, tys
from hugr.dfg import Dfg
from hugr.hugr import Hugr
from hugr.node_port import Node
from hugr.ops import AsCustomOp, Custom, ExtOp
from hugr.std.float import EXTENSION as FLOAT_EXT
from hugr.std.int import OPS_EXTENSION, TYPES_EXTENSION, DivMod
from hugr.std.logic import EXTENSION as LOGIC_EXT
from hugr.std.logic import Not

from .conftest import CX, H, Measure, Rz, validate
from .conftest import EXTENSION as QUANTUM_EXT

STRINGLY_EXT = ext.Extension("my_extension", ext.Version(0, 0, 0))
STRINGLY_EXT.add_op_def(
    ext.OpDef(
        "StringlyOp",
        signature=ext.OpDefSig(
            tys.PolyFuncType([tys.StringParam()], tys.FunctionType.endo([]))
        ),
    )
)


@dataclass
class StringlyOp(AsCustomOp):
    tag: str

    def to_custom(self) -> Custom:
        return ops.ExtOp(
            STRINGLY_EXT.get_op("StringlyOp"),
            tys.FunctionType.endo([]),
            [tys.StringArg(self.tag)],
        ).to_custom()

    @classmethod
    def from_custom(cls, custom: Custom) -> "StringlyOp":
        match custom:
            case Custom(
                name="StringlyOp",
                extension="my_extension",
                args=[tys.StringArg(tag)],
            ):
                return cls(tag=tag)
            case _:
                msg = f"Invalid custom op: {custom}"
                raise AsCustomOp.InvalidCustomOp(msg)


def test_stringly_typed():
    dfg = Dfg()
    n = dfg.add(StringlyOp("world")())
    dfg.set_outputs()
    assert dfg.hugr[n].op == StringlyOp("world")
    validate(dfg.hugr)

    new_h = Hugr.from_serial(dfg.hugr.to_serial())

    assert isinstance(new_h[n].op, Custom)

    registry = ext.ExtensionRegistry()
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


@pytest.fixture()
def registry() -> ext.ExtensionRegistry:
    reg = ext.ExtensionRegistry()
    reg.add_extension(LOGIC_EXT)
    reg.add_extension(QUANTUM_EXT)
    reg.add_extension(STRINGLY_EXT)
    reg.add_extension(TYPES_EXTENSION)
    reg.add_extension(OPS_EXTENSION)
    reg.add_extension(FLOAT_EXT)

    return reg


@pytest.mark.parametrize(
    "as_custom",
    [Not, DivMod, H, CX, Measure, Rz, StringlyOp("hello")],
)
def test_custom(as_custom: AsCustomOp, registry: ext.ExtensionRegistry):
    custom = as_custom.to_custom()

    assert custom.to_custom() == custom
    assert Custom.from_custom(custom) == custom

    assert type(as_custom).from_custom(custom) == as_custom
    # ExtOp compared to Custom via `to_custom`
    assert as_custom.to_serial(Node(0)).deserialize().resolve(registry) == custom
    assert custom == as_custom
    assert as_custom == custom


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
