from hugr.std.prelude import STRING_T, StringVal


def test_string_val():
    ext_val = StringVal("test").to_value()
    assert ext_val.name == "ConstString"
    assert ext_val.typ == STRING_T
    assert ext_val.val == {"value": "test"}
