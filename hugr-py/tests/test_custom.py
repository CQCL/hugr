import pytest

from hugr import tys
from hugr.node_port import Node
from hugr.ops import AsCustomOp, Custom
from hugr.std.int import DivMod
from hugr.std.logic import EXTENSION_ID, Not

from .conftest import CX, H, Measure, Rz


@pytest.mark.parametrize(
    "as_custom",
    [Not, DivMod, H, CX, Measure, Rz],
)
def test_custom(as_custom: AsCustomOp):
    custom = as_custom.to_custom()

    assert custom.to_custom() == custom
    assert Custom.from_custom(custom) == custom

    assert type(as_custom).from_custom(custom) == as_custom
    assert as_custom.to_serial(Node(0)).deserialize() == custom
    assert custom == as_custom
    assert as_custom == custom


def test_custom_bad_eq():
    assert Not != DivMod

    bad_custom_sig = Custom("Not", extension=EXTENSION_ID)  # empty signature

    assert Not != bad_custom_sig

    bad_custom_args = Custom(
        "Not",
        extension=EXTENSION_ID,
        signature=tys.FunctionType.endo([tys.Bool]),
        args=[tys.Bool.type_arg()],
    )

    assert Not != bad_custom_args