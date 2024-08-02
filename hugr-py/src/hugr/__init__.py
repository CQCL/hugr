"""`hugr` is a Python package for the Quantinuum HUGR common
representation.
"""

from .hugr import Hugr
from .node_port import Direction, InPort, Node, OutPort, Wire
from .ops import Op
from .tys import Kind, Type

__all__ = [
    "Hugr",
    "Node",
    "OutPort",
    "InPort",
    "Direction",
    "Op",
    "Kind",
    "Type",
    "Wire",
    "get_serialisation_version",
]

# This is updated by our release-please workflow, triggered by this
# annotation: x-release-please-version
__version__ = "0.5.0"


def get_serialization_version() -> str:
    """Return the current version of the serialization schema."""
    return "live"
