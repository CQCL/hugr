from typing import Literal
from .tys import Type, SumType, PolyFuncType, ConfiguredBaseModel
from .ops import Value, OpType


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

    class Config:
        title = "HugrTesting"
