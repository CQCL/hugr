from semver import Version

from hugr.serialization.extension import (
    ExplicitBound,
    Extension,
    OpDef,
    TypeDef,
    TypeDefBound,
)
from hugr.serialization.tys import (
    FunctionType,
    PolyFuncType,
    Type,
    TypeBound,
    TypeParam,
    TypeTypeParam,
    Variable,
)

EXAMPLE = r"""
{
    "version": "0.1.0",
    "name": "ext",
    "extension_reqs": [],
    "types": {
        "foo": {
            "extension": "ext",
            "name": "foo",
            "params": [
                {
                    "tp": "Type",
                    "b": "C"
                }
            ],
            "description": "foo",
            "bound": {
                "b": "Explicit",
                "bound": "C"
            }
        }
    },
    "values": {},
    "operations": {
        "New": {
            "extension": "ext",
            "name": "New",
            "description": "new",
            "signature": {
                "params": [
                    {
                        "tp": "Type",
                        "b": "C"
                    }
                ],
                "body": {
                    "input": [
                        {
                            "t": "V",
                            "i": 0,
                            "b": "C"
                        }
                    ],
                    "output": [],
                    "extension_reqs": []
                }
            },
            "lower_funcs": []
        }
    }
}
"""


def test_deserialize():
    param = TypeParam(root=TypeTypeParam(b=TypeBound.Copyable))

    bound = TypeDefBound(root=ExplicitBound(bound=TypeBound.Copyable))
    type_def = TypeDef(
        extension="ext", name="foo", description="foo", params=[param], bound=bound
    )
    body = FunctionType(
        input=[Type(root=Variable(b=TypeBound.Copyable, i=0))], output=[]
    )
    op_def = OpDef(
        extension="ext",
        name="New",
        description="new",
        signature=PolyFuncType(
            params=[param],
            body=body,
        ),
        lower_funcs=[],
    )
    ext = Extension(
        version=Version(0, 1, 0),
        name="ext",
        extension_reqs=set(),
        types={"foo": type_def},
        values={},
        operations={"New": op_def},
    )

    ext_load = Extension.model_validate_json(EXAMPLE)
    assert ext == ext_load

    dumped_json = ext.model_dump_json()

    assert Extension.model_validate_json(dumped_json) == ext
