from hugr.build.dfg import Dfg
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
