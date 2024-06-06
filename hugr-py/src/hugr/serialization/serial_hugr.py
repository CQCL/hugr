from typing import Any, Literal

from pydantic import Field, ConfigDict

from .ops import NodeID, OpType, classes as ops_classes
from .tys import model_rebuild, ConfiguredBaseModel
import hugr

Port = tuple[NodeID, int | None]  # (node, offset)
Edge = tuple[Port, Port]


class SerialHugr(ConfiguredBaseModel):
    """A serializable representation of a Hugr."""

    version: Literal["v1"] = "v1"
    nodes: list[OpType]
    edges: list[Edge]
    metadata: list[dict[str, Any] | None] | None = None
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

    @classmethod
    def _pydantic_rebuild(cls, config: ConfigDict = ConfigDict(), **kwargs):
        my_classes = dict(ops_classes)
        my_classes[cls.__name__] = cls
        model_rebuild(my_classes, config=config, **kwargs)

    model_config = ConfigDict(
        title="Hugr",
        json_schema_extra={
            "required": ["version", "nodes", "edges"],
        },
    )
