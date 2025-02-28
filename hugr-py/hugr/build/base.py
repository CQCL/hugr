"""Base classes for HUGR builders."""

from __future__ import annotations

from typing import (
    Protocol,
    cast,
)

from hugr.hugr.base import Hugr, OpVar
from hugr.hugr.node_port import (
    Node,
    ToNode,
)


class ParentBuilder(ToNode, Protocol[OpVar]):
    """Abstract interface implemented by builders of nodes that contain child HUGRs."""

    #: The child HUGR.
    hugr: Hugr[OpVar]
    # Unique parent node.
    parent_node: Node

    def to_node(self) -> Node:
        return self.parent_node

    @property
    def parent_op(self) -> OpVar:
        """The parent node's operation."""
        return cast(OpVar, self.hugr[self.parent_node].op)
