"""Node and port classes for Hugr graphs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    overload,
)

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Iterator


class Direction(Enum):
    """Enum over port directions, INCOMING and OUTGOING."""

    INCOMING = 0
    OUTGOING = 1


NodeIdx = int
PortOffset = int


@dataclass(frozen=True, eq=True, order=True)
class _Port:
    node: Node
    offset: PortOffset
    direction: ClassVar[Direction]


@dataclass(frozen=True, eq=True, order=True)
class InPort(_Port):
    """Incoming port, defined by the `node` it belongs to and the port `offset`."""

    direction: ClassVar[Direction] = Direction.INCOMING

    def __repr__(self) -> str:
        return f"InPort({self.node}, {self.offset})"


class Wire(Protocol):
    """Protocol for objects that can provide a dataflow output port."""

    def out_port(self) -> OutPort:
        """OutPort corresponding to this :class:`Wire`."""
        ...  # pragma: no cover


@dataclass(frozen=True, eq=True, order=True)
class OutPort(_Port, Wire):
    """Outgoing port, defined by the `node` it belongs to and the port `offset`."""

    direction: ClassVar[Direction] = Direction.OUTGOING

    def out_port(self) -> OutPort:
        return self

    def __repr__(self) -> str:
        return f"OutPort({self.node}, {self.offset})"


class ToNode(Wire, Protocol):
    """Protocol by any object that can be treated as a :class:`Node`."""

    def to_node(self) -> Node:
        """Convert to a :class:`Node`."""
        ...  # pragma: no cover

    @overload
    def __getitem__(self, index: PortOffset) -> OutPort: ...
    @overload
    def __getitem__(self, index: slice) -> Iterator[OutPort]: ...
    @overload
    def __getitem__(self, index: tuple[PortOffset, ...]) -> Iterator[OutPort]: ...

    def __getitem__(
        self, index: PortOffset | slice | tuple[PortOffset, ...]
    ) -> OutPort | Iterator[OutPort]:
        return self.to_node()._index(index)

    def out_port(self) -> OutPort:
        return OutPort(self.to_node(), 0)

    def outputs(self) -> Iterator[OutPort]:
        """Returns an iterator over the output ports of this node."""
        return self[:]

    def __iter__(self) -> Iterator[OutPort]:
        return self.outputs()

    def inp(self, offset: PortOffset) -> InPort:
        """Generate an input port for this node.

        Args:
            offset: port offset.

        Returns:
            Incoming port for this node.

        Examples:
            >>> Node(0).inp(1)
            InPort(Node(0), 1)
        """
        return InPort(self.to_node(), offset)

    def out(self, offset: PortOffset) -> OutPort:
        """Generate an output port for this node.

        Args:
            offset: port offset.

        Returns:
            Outgoing port for this node.

        Examples:
            >>> Node(0).out(1)
            OutPort(Node(0), 1)
        """
        return OutPort(self.to_node(), offset)

    def port(self, offset: PortOffset, direction: Direction) -> InPort | OutPort:
        """Generate a port in `direction` for this node with `offset`.

        Examples:
            >>> Node(0).port(1, Direction.INCOMING)
            InPort(Node(0), 1)
            >>> Node(0).port(1, Direction.OUTGOING)
            OutPort(Node(0), 1)
        """
        if direction == Direction.INCOMING:
            return self.inp(offset)
        else:
            return self.out(offset)

    @property
    def metadata(self) -> dict[str, object]:
        """Metadata associated with this node."""
        return self.to_node()._metadata


@dataclass(eq=True, order=True)
class Node(ToNode):
    """Node in hierarchical :class:`Hugr <hugr.hugr.Hugr>` graph,
    with globally unique index.
    """

    idx: NodeIdx
    _metadata: dict[str, object] = field(
        repr=False, compare=False, default_factory=dict
    )
    _num_out_ports: int | None = field(default=None, compare=False, repr=False)

    def _index(
        self, index: PortOffset | slice | tuple[PortOffset, ...]
    ) -> OutPort | Iterator[OutPort]:
        match index:
            case PortOffset(index):
                index = self._normalize_index(index)
                return self.out(index)
            case slice():
                start = index.start or 0
                stop = index.stop if index.stop is not None else self._num_out_ports
                if stop is None:
                    msg = (
                        f"{self} does not have a fixed number of output ports. "
                        "Iterating over all output ports is not supported."
                    )
                    raise ValueError(msg)

                start = self._normalize_index(start, allow_overflow=True)
                stop = self._normalize_index(stop, allow_overflow=True)
                step = index.step or 1

                return (self[i] for i in range(start, stop, step))
            case tuple(xs):
                return (self[i] for i in xs)

    def _normalize_index(self, index: int, allow_overflow: bool = False) -> int:
        """Given an index passed to `__getitem__`, normalize it to be within the
        range of output ports.

        Args:
            index: index to normalize.
            allow_overflow: whether to allow indices beyond the number of outputs.
                If True, indices over `self._num_out_ports` will be truncated.

        Returns:
            Normalized index.

        Raises:
            IndexError: if the index is out of range.
        """
        msg = f"Index {index} out of range"

        if self._num_out_ports is not None:
            if index >= self._num_out_ports and not allow_overflow:
                raise IndexError(msg)
            if index < -self._num_out_ports:
                raise IndexError(msg)
        else:
            if index < 0:
                raise IndexError(msg)

        if index >= 0 and self._num_out_ports is not None:
            return min(index, self._num_out_ports)
        elif index >= 0:
            return index
        else:
            assert self._num_out_ports is not None
            return self._num_out_ports + index

    def to_node(self) -> Node:
        return self

    def __repr__(self) -> str:
        return f"Node({self.idx})"

    def __hash__(self) -> int:
        return hash(self.idx)


P = TypeVar("P", InPort, OutPort)


@dataclass(frozen=True, eq=True, order=True)
class _SubPort(Generic[P]):
    port: P
    sub_offset: int = 0

    def next_sub_offset(self) -> Self:
        return replace(self, sub_offset=self.sub_offset + 1)
