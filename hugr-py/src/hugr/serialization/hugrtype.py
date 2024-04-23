from typing import Any, Literal
from pydantic import BaseModel, Field
from .tys import Type, USize

class HugrType(BaseModel):
    """A serializable representation of a Hugr Type. Intended for testing only."""

    typ: Type
    v: Literal["v1"] = "v1"

    @classmethod
    def get_version(cls) -> str:
        """Return the version of the schema."""
        return cls(typ=Type(USize())).v

    class Config:
        title = "HugrType"
        json_schema_extra = {
            "required": ["typ"],
        }
