from typing import Literal
from pydantic import BaseModel
from .tys import Type, USize, SumType
from .ops import Value


class TestingHugr(BaseModel):
    """A serializable representation of a Hugr Type, SumType, or Value. Intended for testing only."""

    typ: Type | SumType | Value
    version: Literal["v2"] = "v2"

    @classmethod
    def get_version(cls) -> str:
        """Return the version of the schema."""
        return cls(typ=Type(USize())).version

    class Config:
        title = "HugrTesting"
        json_schema_extra = {
            "required": ["typ"],
        }
