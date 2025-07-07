import pytest

from hugr.build.dfg import Dfg
from hugr.std.int import IntVal, int_t
from hugr.std.prelude import STRING_T, StringVal

from .conftest import validate


def test_string_val():
    val = StringVal("test")
    ext_val = val.to_value()
    assert ext_val.name == "ConstString"
    assert ext_val.typ == STRING_T
    assert ext_val.val == "test"

    dfg = Dfg()
    v = dfg.load(val)
    dfg.set_outputs(v)

    validate(dfg.hugr)


@pytest.mark.parametrize(
    ("log_width", "v", "unsigned"),
    [
        (5, 1, 1),
        (4, 0, 0),
        (6, 42, 42),
        (2, -1, 15),
        (1, -2, 2),
        (3, -23, 233),
        (3, -256, None),
        (2, 16, None),
    ],
)
def test_int_val(log_width: int, v: int, unsigned: int | None):
    val = IntVal(v, log_width)
    if unsigned is None:
        with pytest.raises(
            ValueError,
            match=f"Value {v} out of range for {1<<log_width}-bit signed integer.",
        ):
            val.to_value()
        return
    ext_val = val.to_value()

    assert ext_val.name == "ConstInt"
    assert ext_val.typ == int_t(log_width)
    assert ext_val.val == {"log_width": log_width, "value": unsigned}

    dfg = Dfg()
    o = dfg.load(val)
    dfg.set_outputs(o)

    validate(dfg.hugr)
