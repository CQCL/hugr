from typing import Any, Literal

from pydantic import BaseModel, Field

from .ops import NodeID, OpType
import hugr

Port = tuple[NodeID, int | None]  # (node, offset)
Edge = tuple[Port, Port]


class SerialHugr(BaseModel):
    """A serializable representation of a Hugr."""

    version: Literal["v1"] = "v1"
    nodes: list[OpType]
    edges: list[Edge]
    encoder: str | None = Field(
        default=None, description="The name of the encoder used to generate the Hugr."
    )

    def to_json(self) -> str:
        """Return a JSON representation of the Hugr."""
        self.encoder = f"hugr-py v{hugr.__version__}"
        return self.model_dump_json()

    @classmethod
    def load_json(cls, json: dict[Any, Any]) -> "SerialHugr":
        """Decode a JSON-encoded Hugr."""
        return cls(**json)

    @classmethod
    def get_version(cls) -> str:
        """Return the version of the schema."""
        return cls(nodes=[], edges=[]).version

    class Config:
        title = "Hugr"
        json_schema_extra = {
            "required": ["version", "nodes", "edges"],
        }
