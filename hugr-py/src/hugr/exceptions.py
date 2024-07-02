"""HUGR builder exceptions."""

from dataclasses import dataclass


@dataclass
class NoSiblingAncestor(Exception):
    """No sibling ancestor of target for valid inter-graph edge."""

    src: int
    tgt: int

    @property
    def msg(self):
        return (
            f"Source {self.src} has no sibling ancestor of target {self.tgt},"
            " so cannot wire up."
        )


@dataclass
class NotInSameCfg(Exception):
    """Source and target nodes are not in the same CFG."""

    src: int
    tgt: int

    @property
    def msg(self):
        return (
            f"Source {self.src} is not in the same CFG as target {self.tgt},"
            " so cannot wire up."
        )


@dataclass
class MismatchedExit(Exception):
    """Edge to exit block signature mismatch."""

    src: int

    @property
    def msg(self):
        return (
            f"Exit branch from node {self.src} does not match existing exit block type."
        )


class ParentBeforeChild(Exception):
    """Parent added before child."""

    msg: str = "Parent node must be added before child node."
