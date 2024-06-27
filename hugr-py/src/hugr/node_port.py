from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    ClassVar,
    Iterator,
    Protocol,
    overload,
    TypeVar,
    Generic,
)
from typing_extensions import Self


class Direction(Enum):
    INCOMING = 0
    OUTGOING = 1


@dataclass(frozen=True, eq=True, order=True)
class _Port:
    node: Node
    offset: int
    direction: ClassVar[Direction]


@dataclass(frozen=True, eq=True, order=True)
class InPort(_Port):
    direction: ClassVar[Direction] = Direction.INCOMING


class Wire(Protocol):
    def out_port(self) -> OutPort: ...


@dataclass(frozen=True, eq=True, order=True)
class OutPort(_Port, Wire):
    direction: ClassVar[Direction] = Direction.OUTGOING

    def out_port(self) -> OutPort:
        return self


class ToNode(Wire, Protocol):
    def to_node(self) -> Node: ...

    @overload
    def __getitem__(self, index: int) -> OutPort: ...
    @overload
    def __getitem__(self, index: slice) -> Iterator[OutPort]: ...
    @overload
    def __getitem__(self, index: tuple[int, ...]) -> Iterator[OutPort]: ...

    def __getitem__(
        self, index: int | slice | tuple[int, ...]
    ) -> OutPort | Iterator[OutPort]:
        return self.to_node()._index(index)

    def out_port(self) -> "OutPort":
        return OutPort(self.to_node(), 0)

    def inp(self, offset: int) -> InPort:
        return InPort(self.to_node(), offset)

    def out(self, offset: int) -> OutPort:
        return OutPort(self.to_node(), offset)

    def port(self, offset: int, direction: Direction) -> InPort | OutPort:
        if direction == Direction.INCOMING:
            return self.inp(offset)
        else:
            return self.out(offset)


@dataclass(frozen=True, eq=True, order=True)
class Node(ToNode):
    idx: int
    _num_out_ports: int | None = field(default=None, compare=False)

    def _index(
        self, index: int | slice | tuple[int, ...]
    ) -> OutPort | Iterator[OutPort]:
        match index:
            case int(index):
                if self._num_out_ports is not None:
                    if index >= self._num_out_ports:
                        raise IndexError("Index out of range")
                return self.out(index)
            case slice():
                start = index.start or 0
                stop = index.stop or self._num_out_ports
                if stop is None:
                    raise ValueError(
                        "Stop must be specified when number of outputs unknown"
                    )
                step = index.step or 1
                return (self[i] for i in range(start, stop, step))
            case tuple(xs):
                return (self[i] for i in xs)

    def to_node(self) -> Node:
        return self


P = TypeVar("P", InPort, OutPort)


@dataclass(frozen=True, eq=True, order=True)
class _SubPort(Generic[P]):
    port: P
    sub_offset: int = 0

    def next_sub_offset(self) -> Self:
        return replace(self, sub_offset=self.sub_offset + 1)
