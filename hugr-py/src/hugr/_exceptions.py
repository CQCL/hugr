from dataclasses import dataclass


@dataclass
class NoSiblingAncestor(Exception):
    src: int
    tgt: int

    @property
    def msg(self):
        return f"Source {self.src} has no sibling ancestor of target {self.tgt}, so cannot wire up."


@dataclass
class NotInSameCfg(Exception):
    src: int
    tgt: int

    @property
    def msg(self):
        return f"Source {self.src} is not in the same CFG as target {self.tgt}, so cannot wire up."


class ParentBeforeChild(Exception):
    msg: str = "Parent node must be added before child node."
