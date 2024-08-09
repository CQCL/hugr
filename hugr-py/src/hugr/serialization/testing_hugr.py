from pydantic import ConfigDict

from .extension import OpDef
from .ops import OpType, Value
from .ops import classes as ops_classes
from .serial_hugr import VersionField
from .tys import ConfiguredBaseModel, PolyFuncType, SumType, Type, model_rebuild


class TestingHugr(ConfiguredBaseModel):
    """A serializable representation of a Hugr Type, SumType, PolyFuncType,
    Value, OpType. Intended for testing only.
    """

    version: str = VersionField
    typ: Type | None = None
    sum_type: SumType | None = None
    poly_func_type: PolyFuncType | None = None
    value: Value | None = None
    optype: OpType | None = None
    op_def: OpDef | None = None

    @classmethod
    def get_version(cls) -> str:
        """Return the version of the schema."""
        return cls().version

    @classmethod
    def _pydantic_rebuild(cls, config: ConfigDict | None = None, **kwargs):
        config = config or ConfigDict()
        my_classes = dict(ops_classes)
        my_classes[cls.__name__] = cls
        model_rebuild(my_classes, config=config, **kwargs)
