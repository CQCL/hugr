from collections.abc import Hashable, ItemsView, MutableMapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar


L = TypeVar("L", bound=Hashable)
R = TypeVar("R", bound=Hashable)


@dataclass()
class BiMap(MutableMapping, Generic[L, R]):
    fwd: dict[L, R] = field(default_factory=dict)
    bck: dict[R, L] = field(default_factory=dict)

    def __getitem__(self, key: L) -> R:
        return self.fwd[key]

    def __setitem__(self, key: L, value: R) -> None:
        self.insert_left(key, value)

    def __delitem__(self, key: L) -> None:
        self.delete_left(key)

    def __iter__(self):
        return iter(self.fwd)

    def __len__(self) -> int:
        return len(self.fwd)

    def items(self) -> ItemsView[L, R]:
        return self.fwd.items()

    def get_left(self, key: R) -> L | None:
        return self.bck.get(key)

    def get_right(self, key: L) -> R | None:
        return self.fwd.get(key)

    def insert_left(self, key: L, value: R) -> None:
        self.fwd[key] = value
        self.bck[value] = key

    def insert_right(self, key: R, value: L) -> None:
        self.bck[key] = value
        self.fwd[value] = key

    def delete_left(self, key: L) -> None:
        del self.bck[self.fwd[key]]
        del self.fwd[key]

    def delete_right(self, key: R) -> None:
        del self.fwd[self.bck[key]]
        del self.bck[key]
