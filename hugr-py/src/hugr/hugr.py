from dataclasses import dataclass

from collections.abc import Collection, MutableMapping, Hashable
from typing import Iterable, Sequence, Protocol, Generic, TypeVar

from hugr.serialization.serial_hugr import SerialHugr
from hugr.serialization.ops import OpType as Op, DataflowOp
from hugr.serialization.tys import Type, Qubit


L = TypeVar("L", bound=Hashable)
R = TypeVar("R", bound=Hashable)


@dataclass(init=False)
class BiMap(MutableMapping, Generic[L, R]):
    fwd: dict[L, R]
    bck: dict[R, L]

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


class ToPort(Protocol):
    def to_port(self) -> "OutPort": ...


@dataclass(frozen=True, eq=True, order=True)
class Node(ToPort):
    idx: int

    def to_port(self) -> "OutPort":
        return OutPort(self, 0)


@dataclass(frozen=True, eq=True, order=True)
class Port:
    node: Node
    offset: int


@dataclass(frozen=True, eq=True, order=True)
class InPort(Port):
    pass


@dataclass(frozen=True, eq=True, order=True)
class OutPort(Port, ToPort):
    def to_port(self) -> "OutPort":
        return self


@dataclass()
class NodeData:
    weight: Op
    in_ports: set[InPort]
    out_ports: set[OutPort]


@dataclass(init=False)
class Hugr:
    root: Node
    nodes: dict[Node, NodeData]
    links: BiMap[OutPort, InPort]

    def add_node(self, op: Op) -> Node:
        node = Node(len(self.nodes))
        self.nodes[node] = NodeData(op, set(), set())
        return node

    def add_link(self, src: OutPort, dst: InPort) -> None:
        self.links.insert_left(src, dst)
        self.nodes[dst.node].in_ports.add(dst)
        self.nodes[src.node].out_ports.add(src)

    def in_ports(self, node: Node) -> Collection[InPort]:
        return self.nodes[node].in_ports

    def out_ports(self, node: Node) -> Collection[OutPort]:
        return self.nodes[node].out_ports

    def to_serial(self) -> SerialHugr:
        return SerialHugr(
            version="v1",
            # non contiguous indices will be erased
            nodes=[node.weight for _, node in sorted(self.nodes.items())],
            edges=[
                ((src.node, src.offset), (dst.node, dst.offset))
                for src, dst in self.links.items()
            ],
        )

    @classmethod
    def from_serial(cls, serial: SerialHugr) -> "Hugr":
        raise NotImplementedError


@dataclass()
class Dfg(Hugr):
    input_node: Node
    output_node: Node

    def __init__(self, input_types: Sequence[Type]) -> None:
        self.root = Node(0)

    def inputs(self) -> list[OutPort]:
        return []

    def add_op(self, op: DataflowOp, ports: Iterable[ToPort]) -> Node:
        node = Node(len(self.nodes))
        # self.nodes[node] = NodeData(op, list(in_ports), [])
        return node

    def set_outputs(self, ports: Iterable[ToPort]) -> None:
        pass


if __name__ == "__main__":
    h = Dfg([Type(Qubit())] * 2)
