from dataclasses import dataclass, field

from collections.abc import Collection, MutableMapping, Hashable, Mapping
from typing import Iterable, Sequence, Protocol, Generic, TypeVar

from hugr.serialization.serial_hugr import SerialHugr
from hugr.serialization.ops import OpType as Op
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


@dataclass(frozen=True, eq=True, order=True)
class Port:
    node: "Node"
    offset: int


@dataclass(frozen=True, eq=True, order=True)
class InPort(Port):
    pass


class ToPort(Protocol):
    def to_port(self) -> "OutPort": ...


@dataclass(frozen=True, eq=True, order=True)
class OutPort(Port, ToPort):
    def to_port(self) -> "OutPort":
        return self


@dataclass(frozen=True, eq=True, order=True)
class Node(ToPort):
    idx: int

    def to_port(self) -> "OutPort":
        return OutPort(self, 0)

    def in_port(self, offset: int) -> InPort:
        return InPort(self, offset)

    def out_port(self, offset: int) -> OutPort:
        return OutPort(self, offset)


@dataclass()
class NodeData:
    weight: Op
    _in_ports: set[int]
    _out_ports: set[int]


@dataclass(init=False)
class Hugr(Mapping):
    root: Node
    nodes: list[NodeData | None]
    links: BiMap[OutPort, InPort]
    _free_nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, key: Node) -> NodeData:
        try:
            n = self.nodes[key.idx]
        except IndexError:
            n = None
        if n is None:
            raise KeyError(key)
        return n

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes) - len(self._free_nodes)

    def add_node(self, op: Op) -> Node:
        # TODO add in_ports and out_ports
        node_data = NodeData(op, set(), set())

        if self._free_nodes:
            node = self._free_nodes.pop()
            self.nodes[node.idx] = node_data
        else:
            node = Node(len(self.nodes))
            self.nodes.append(node_data)
        return node

    def delete_node(self, node: Node) -> None:
        for offset in self[node]._in_ports:
            self.links.delete_right(node.in_port(offset))
        for offset in self[node]._out_ports:
            self.links.delete_left(node.out_port(offset))
        self.nodes[node.idx] = None
        self._free_nodes.append(node)

    def add_link(self, src: OutPort, dst: InPort) -> None:
        self.links.insert_left(src, dst)
        self[dst.node]._in_ports.add(dst.offset)
        self[src.node]._out_ports.add(src.offset)

    def in_ports(self, node: Node) -> Collection[InPort]:
        return [node.in_port(o) for o in self[node]._in_ports]

    def out_ports(self, node: Node) -> Collection[OutPort]:
        return [node.out_port(o) for o in self[node]._out_ports]

    def to_serial(self) -> SerialHugr:
        return SerialHugr(
            version="v1",
            # non contiguous indices will be erased
            nodes=[node.weight for node in self.nodes if node is not None],
            edges=[
                ((src.node, src.offset), (dst.node, dst.offset))
                for src, dst in self.links.items()
            ],
        )

    @classmethod
    def from_serial(cls, serial: SerialHugr) -> "Hugr":
        raise NotImplementedError


@dataclass()
class Dfg:
    hugr: Hugr
    input_node: Node
    output_node: Node

    def __init__(self, input_types: Sequence[Type]) -> None:
        self.root = Node(0)

    def inputs(self) -> list[OutPort]:
        return []

    def add_op(self, op: Op, ports: Iterable[ToPort]) -> Node:
        # TODO wire up ports
        return self.hugr.add_node(op)

    def set_outputs(self, ports: Iterable[ToPort]) -> None:
        pass


if __name__ == "__main__":
    h = Dfg([Type(Qubit())] * 2)
