from dataclasses import dataclass


@dataclass
class NoSiblingAncestor(Exception):
    src: int
    tgt: int

    @property
    def msg(self):
        return f"Source {self.src} has no sibling ancestor of target {self.tgt}, so cannot wire up."
