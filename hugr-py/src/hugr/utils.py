"""Shared utility classes and functions."""

from collections.abc import Hashable, ItemsView, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

L = TypeVar("L", bound=Hashable)
R = TypeVar("R", bound=Hashable)


class NotBijection(Exception):
    """Initial map is not a bijection."""


@dataclass()
class BiMap(MutableMapping, Generic[L, R]):
    """Bidirectional map backed by two dictionaries, between left types `L` and
    right types `R`.
    """

    fwd: dict[L, R] = field(default_factory=dict)
    bck: dict[R, L] = field(default_factory=dict)

    def __init__(self, fwd: Mapping[L, R] | None = None) -> None:
        """Initialize a bidirectional map.

        Args:
            fwd: Left to right mapping. Defaults to empty.

        Raises:
            NotBijection: If the initial map is not a bijection.
        """
        fwd = fwd or {}
        if len(fwd) != len(set(fwd.values())):
            raise NotBijection
        self.fwd = dict(fwd)
        self.bck = {v: k for k, v in fwd.items()}

    def __getitem__(self, key: L) -> R:
        """Get the right value for a left key.

        Args:
            key: Left key.

        Raises:
            KeyError: If the key is not found.

        Example:
            >>> bm = BiMap({"a": 1})
            >>> bm["a"]
            1
        """
        return self.fwd[key]

    def __setitem__(self, key: L, value: R) -> None:
        """See :meth:`insert_left`."""
        self.insert_left(key, value)

    def __delitem__(self, key: L) -> None:
        """See :meth:`delete_left`."""
        self.delete_left(key)

    def __iter__(self):
        return iter(self.fwd)

    def __len__(self) -> int:
        return len(self.fwd)

    def items(self) -> ItemsView[L, R]:
        """Iterator over left, right pairs.

        Example:
            >>> bm = BiMap({"a": 1, "b": 2})
            >>> list(bm.items())
            [('a', 1), ('b', 2)]
        """
        return self.fwd.items()

    def get_left(self, key: R) -> L | None:
        """Get a left value using a right key.

        Example:
            >>> bm = BiMap({"a": 1})
            >>> bm.get_left(1)
            'a'
            >>> bm.get_left(2)
        """
        return self.bck.get(key)

    def get_right(self, key: L) -> R | None:
        """Get a right value using a left key.

        Example:
            >>> bm = BiMap({"a": 1})
            >>> bm.get_right("a")
            1
            >>> bm.get_right("b")
        """
        return self.fwd.get(key)

    def insert_left(self, key: L, value: R) -> None:
        """Insert a left key and right value.
        If the key or value already exist, the existing key-value pair is replaced.

        Args:
            key: Left key.
            value: Right value.

        Example:
            >>> bm = BiMap()
            >>> bm.insert_left("a", 1)
            >>> bm["a"]
            1
        """
        if (existing_key := self.bck.get(value)) is not None:
            del self.fwd[existing_key]
        if (existing_value := self.fwd.get(key)) is not None:
            del self.bck[existing_value]
        self.fwd[key] = value
        self.bck[value] = key

    def insert_right(self, key: R, value: L) -> None:
        """Insert a right key and left value.
        If the key or value already exist, the existing key-value pair is replaced.

        Args:
            key: Right key.
            value: Left value.

        Example:
            >>> bm = BiMap()
            >>> bm.insert_right(1, "a")
            >>> bm["a"]
            1
        """
        self.insert_left(value, key)

    def delete_left(self, key: L) -> None:
        """Delete a left key and its right value.

        Args:
            key: Left key.

        Raises:
            KeyError: If the key is not found.

        Example:
            >>> bm = BiMap({"a": 1})
            >>> bm.delete_left("a")
            >>> bm
            BiMap({})
        """
        del self.bck[self.fwd[key]]
        del self.fwd[key]

    def delete_right(self, key: R) -> None:
        """Delete a right key and its left value.

        Args:
            key: Right key.

        Raises:
            KeyError: If the key is not found.

        Example:
            >>> bm = BiMap({"a": 1})
            >>> bm.delete_right(1)
            >>> bm
            BiMap({})
        """
        del self.fwd[self.bck[key]]
        del self.bck[key]

    def __repr__(self) -> str:
        return f"BiMap({self.fwd})"


S = TypeVar("S", covariant=True)


class SerCollection(Protocol[S]):
    """Protocol for serializable objects."""

    def _to_serial_root(self) -> S:
        """Convert to serializable root model."""
        ...  # pragma: no cover


class DeserCollection(Protocol[S]):
    """Protocol for deserializable objects."""

    def deserialize(self) -> S:
        """Deserialize from model."""
        ...  # pragma: no cover


def ser_it(it: Iterable[SerCollection[S]]) -> list[S]:
    """Serialize an iterable of serializable objects."""
    return [v._to_serial_root() for v in it]


def deser_it(it: Iterable[DeserCollection[S]]) -> list[S]:
    """Deserialize an iterable of deserializable objects."""
    return [v.deserialize() for v in it]


T = TypeVar("T")


def comma_sep_str(items: Iterable[T]) -> str:
    """Join items with commas and str."""
    return ", ".join(map(str, items))


def comma_sep_repr(items: Iterable[T]) -> str:
    """Join items with commas and repr."""
    return ", ".join(map(repr, items))
