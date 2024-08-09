from dataclasses import dataclass

import pytest

from hugr import tys
from hugr.dfg import Dfg
from hugr.ops import AsCustomOp, Custom
from hugr.std.int import DivMod
from hugr.std.logic import EXTENSION, Not

from .conftest import CX, H, Measure, Rz, validate


@dataclass
class StringlyOp(AsCustomOp):
    tag: str

    def to_custom(self) -> Custom:
        return Custom(
            "StringlyOp",
            extension="my_extension",
            signature=tys.FunctionType.endo([]),
            args=[tys.StringArg(self.tag)],
        )

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


@pytest.mark.parametrize(
    "as_custom",
    [Not, DivMod, H, CX, Measure, Rz, StringlyOp("hello")],
)
def test_custom(as_custom: AsCustomOp):
    custom = as_custom.to_custom()

    assert custom.to_custom() == custom
    assert Custom.from_custom(custom) == custom

    assert type(as_custom).from_custom(custom) == as_custom
    # TODO extension resolution needed for this equality
    # assert as_custom.to_serial(Node(0)).deserialize() == custom
    assert custom == as_custom
    assert as_custom == custom


def test_custom_bad_eq():
    assert Not != DivMod

    bad_custom_sig = Custom("Not", extension=EXTENSION.name)  # empty signature

    assert Not != bad_custom_sig

    bad_custom_args = Custom(
        "Not",
        extension=EXTENSION.name,
        signature=tys.FunctionType.endo([tys.Bool]),
        args=[tys.Bool.type_arg()],
    )

    assert Not != bad_custom_args
