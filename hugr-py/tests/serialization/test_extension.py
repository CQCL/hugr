from pydantic_extra_types.semantic_version import SemanticVersion

from hugr._serialization.extension import (
    ExplicitBound,
    Extension,
    OpDef,
    Package,
    TypeDef,
    TypeDefBound,
)
from hugr._serialization.ops import Module, OpType
from hugr._serialization.serial_hugr import SerialHugr, serialization_version
from hugr._serialization.tys import (
    FunctionType,
    PolyFuncType,
    Type,
    TypeBound,
    TypeParam,
    TypeTypeParam,
    Variable,
)
from hugr.envelope import EnvelopeConfig
from tests.conftest import validate

EXAMPLE = r"""
{
    "version": "0.1.0",
    "name": "ext",
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
                    "output": []
                }
            },
            "lower_funcs": []
        }
    }
}
"""


def test_extension():
    assert serialization_version() == Extension.get_version()
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
        version=SemanticVersion(0, 1, 0),
        name="ext",
        types={"foo": type_def},
        operations={"New": op_def},
    )

    ext_load = Extension.model_validate_json(EXAMPLE)
    assert ext == ext_load

    dumped_json = ext.model_dump_json()

    assert Extension.model_validate_json(dumped_json) == ext
    hugr_ext = ext.deserialize()
    assert hugr_ext.from_json(hugr_ext.to_json()) == hugr_ext
    from hugr.package import Package as HugrPackage

    validate(HugrPackage([], [hugr_ext]))


def test_package():
    assert serialization_version() == Package.get_version()

    ext = Extension(
        version=SemanticVersion(0, 1, 0),
        name="ext",
        types={},
        operations={},
    )
    ext_load = Extension.model_validate_json(EXAMPLE)

    package = Package(
        extensions=[ext, ext_load],
        modules=[SerialHugr(nodes=[OpType(root=Module(parent=0))], edges=[])],
    )

    package_load = Package.model_validate_json(package.model_dump_json())
    assert package == package_load

    hugr_package = package.deserialize()
    assert (
        hugr_package.from_bytes(hugr_package.to_bytes(EnvelopeConfig.TEXT))
        == hugr_package
    )

    validate(package.deserialize())
