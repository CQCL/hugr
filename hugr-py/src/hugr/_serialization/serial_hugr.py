from typing import Any

from pydantic import ConfigDict, Field

from hugr.hugr.node_port import NodeIdx, PortOffset

from .ops import OpType
from .ops import classes as ops_classes
from .tys import ConfiguredBaseModel, model_rebuild

Port = tuple[NodeIdx, PortOffset | None]
Edge = tuple[Port, Port]


def serialization_version() -> str:
    """Return the current version of the serialization schema."""
    return "live"


VersionField = Field(
    default_factory=serialization_version,
    title="Version",
    description="Serialisation Schema Version",
    frozen=True,
)


class SerialHugr(ConfiguredBaseModel):
    """A serializable representation of a Hugr."""

    version: str = VersionField
    nodes: list[OpType]
    edges: list[Edge]
    metadata: list[dict[str, Any] | None] | None = None
    encoder: str | None = Field(
        default=None, description="The name of the encoder used to generate the Hugr."
    )
    entrypoint: NodeIdx | None = None

    def to_json(self) -> str:
        """Return a JSON representation of the Hugr."""
        from hugr import __version__ as hugr_version

        self.encoder = f"hugr-py v{hugr_version}"
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
    def _pydantic_rebuild(cls, config: ConfigDict | None = None, **kwargs):
        config = config or ConfigDict()
        my_classes = dict(ops_classes)
        my_classes[cls.__name__] = cls
        model_rebuild(my_classes, config=config, **kwargs)

    model_config = ConfigDict(
        title="Hugr",
        json_schema_extra={
            "required": ["version", "nodes", "edges"],
        },
    )
