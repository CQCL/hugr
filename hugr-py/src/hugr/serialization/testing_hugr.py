from typing import Literal, Optional
from pydantic import BaseModel
from .tys import Type, SumType
from .ops import Value


class TestingHugr(BaseModel):
    """A serializable representation of a Hugr Type, SumType, or Value. Intended for testing only."""

    version: Literal["v1"] = "v1"
    typ: Optional[Type] = None
    sum_type: Optional[SumType] = None
    value: Optional[Value] = None

    @classmethod
    def get_version(cls) -> str:
        """Return the version of the schema."""
        return cls().version

    class Config:
        title = "HugrTesting"
