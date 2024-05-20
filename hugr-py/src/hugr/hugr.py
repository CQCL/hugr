from dataclasses import dataclass, field

from collections.abc import Collection, ItemsView, MutableMapping, Hashable, Mapping
from typing import Iterable, Sequence, Protocol, Generic, TypeVar

from hugr.serialization.serial_hugr import SerialHugr
from hugr.serialization.ops import NodeID, OpType as SerialOp, Module
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

    def inp(self, offset: int) -> InPort:
        return InPort(self, offset)

    def out(self, offset: int) -> OutPort:
        return OutPort(self, offset)


class Op(Protocol):
    def to_serial(self, parent: NodeID) -> SerialOp: ...


@dataclass(init=False)
class DummyOp(Op):
    input_extensions: set[str] | None = None

    def to_serial(self, parent: NodeID) -> SerialOp:
        return SerialOp(root=Module(parent=-1))


@dataclass()
class NodeData:
    op: Op
    parent: Node | None
    _in_ports: set[int]
    _out_ports: set[int]


@dataclass(init=False)
class Hugr(Mapping):
    root: Node
    nodes: list[NodeData | None]
    _links: BiMap[OutPort, InPort]
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

    def add_node(self, op: Op, parent: Node | None = None) -> Node:
        parent = parent or self.root
        # TODO add in_ports and out_ports
        node_data = NodeData(op, parent, set(), set())

        if self._free_nodes:
            node = self._free_nodes.pop()
            self.nodes[node.idx] = node_data
        else:
            node = Node(len(self.nodes))
            self.nodes.append(node_data)
        return node

    def delete_node(self, node: Node) -> None:
        for offset in self[node]._in_ports:
            self._links.delete_right(node.inp(offset))
        for offset in self[node]._out_ports:
            self._links.delete_left(node.out(offset))
        self.nodes[node.idx] = None
        self._free_nodes.append(node)

    def add_link(self, src: OutPort, dst: InPort) -> None:
        self._links.insert_left(src, dst)
        self[dst.node]._in_ports.add(dst.offset)
        self[src.node]._out_ports.add(src.offset)

    def delete_link(self, src: OutPort, dst: InPort) -> None:
        self._links.delete_left(src)
        self[dst.node]._in_ports.remove(dst.offset)
        self[src.node]._out_ports.remove(src.offset)

    def in_ports(self, node: Node) -> Collection[InPort]:
        return [node.inp(o) for o in self[node]._in_ports]

    def out_ports(self, node: Node) -> Collection[OutPort]:
        return [node.out(o) for o in self[node]._out_ports]

    def insert_hugr(self, hugr: "Hugr", parent: Node | None = None) -> dict[Node, Node]:
        mapping: dict[Node, Node] = {}
        for idx, node_data in enumerate(self.nodes):
            if node_data is not None:
                mapping[Node(idx)] = self.add_node(node_data.op, parent)
        for src, dst in hugr._links.items():
            self.add_link(
                mapping[src.node].out(src.offset), mapping[dst.node].inp(dst.offset)
            )
        return mapping

    def to_serial(self) -> SerialHugr:
        node_it = (node for node in self.nodes if node is not None)
        return SerialHugr(
            version="v1",
            # non contiguous indices will be erased
            nodes=[
                node.op.to_serial(node.parent.idx if node.parent else idx)
                for idx, node in enumerate(node_it)
            ],
            edges=[
                ((src.node.idx, src.offset), (dst.node.idx, dst.offset))
                for src, dst in self._links.items()
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

    def insert_nested(self, dfg: "Dfg", ports: Iterable[ToPort]) -> Node:
        mapping = self.hugr.insert_hugr(dfg.hugr, self.hugr.root)
        # TODO wire up ports
        return mapping[dfg.hugr.root]

    def add_nested(self, ports: Iterable[ToPort]) -> "Dfg":
        dfg = Dfg.__new__(Dfg)
        dfg.hugr = self.hugr
        # I/O nodes

        return dfg

    def set_outputs(self, ports: Iterable[ToPort]) -> None:
        pass


if __name__ == "__main__":
    h = Dfg([Type(Qubit())] * 2)

    a, b = h.inputs()
    x = h.add_op(DummyOp(), [a, b])

    y = h.add_op(DummyOp(), [x, x])

    h.set_outputs([y])
