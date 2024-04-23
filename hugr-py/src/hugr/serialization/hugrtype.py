from typing import Any, Literal
from pydantic import BaseModel, Field
from .tys import Type, USize, SumTypeBase


class HugrType(BaseModel):
    """A serializable representation of a Hugr Type. Intended for testing only."""

    typ: Type | SumTypeBase
    version: Literal["v1"] = "v1"

    @classmethod
    def get_version(cls) -> str:
        """Return the version of the schema."""
        return cls(typ=Type(USize())).version

    class Config:
        title = "HugrType"
        json_schema_extra = {
            "required": ["typ"],
        }
