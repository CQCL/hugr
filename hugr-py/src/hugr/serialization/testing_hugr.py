from pydantic import ConfigDict
from typing import Literal
from .tys import Type, SumType, PolyFuncType, ConfiguredBaseModel, model_rebuild
from .ops import Value, OpType, classes


class TestingHugr(ConfiguredBaseModel):
    """A serializable representation of a Hugr Type, SumType, PolyFuncType,
    Value, OpType. Intended for testing only."""

    version: Literal["v1"] = "v1"
    typ: Type | None = None
    sum_type: SumType | None = None
    poly_func_type: PolyFuncType | None = None
    value: Value | None = None
    optype: OpType | None = None

    @classmethod
    def get_version(cls) -> str:
        """Return the version of the schema."""
        return cls().version

    @classmethod
    def _pydantic_rebuild(cls, config: ConfigDict = ConfigDict(), **kwargs):
        model_rebuild([(cls.__name__, cls)] + classes, config=config, **kwargs)

    class Config:
        title = "HugrTesting"
