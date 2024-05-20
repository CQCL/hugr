from dataclasses import dataclass

from collections.abc import Collection, Mapping
from typing import Iterable, Sequence, Protocol, Generic, TypeVar

from hugr.serialization.serial_hugr import SerialHugr
from hugr.serialization.ops import BaseOp, NodeID, OpType as SerialOp
import hugr.serialization.ops as sops
from hugr.serialization.tys import Type
from hugr.utils import BiMap


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


T = TypeVar("T", bound=BaseOp)


@dataclass()
class DummyOp(Op, Generic[T]):
    _serial_op: T

    def to_serial(self, parent: NodeID) -> SerialOp:
        self._serial_op.parent = parent
        return SerialOp(root=self._serial_op)  # type: ignore


@dataclass()
class NodeData:
    op: Op
    parent: Node | None
    _in_ports: set[int]
    _out_ports: set[int]


class Hugr(Mapping):
    root: Node
    _nodes: list[NodeData | None]
    _links: BiMap[OutPort, InPort]
    _free_nodes: list[Node]

    def __init__(self, root_op: Op) -> None:
        self.root = Node(0)
        self._nodes = [NodeData(root_op, None, set(), set())]
        self._links = BiMap()
        self._free_nodes = []

    def __getitem__(self, key: Node) -> NodeData:
        try:
            n = self._nodes[key.idx]
        except IndexError:
            n = None
        if n is None:
            raise KeyError(key)
        return n

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes) - len(self._free_nodes)

    def add_node(self, op: Op, parent: Node | None = None) -> Node:
        parent = parent or self.root
        # TODO add in_ports and out_ports
        node_data = NodeData(op, parent, set(), set())

        if self._free_nodes:
            node = self._free_nodes.pop()
            self._nodes[node.idx] = node_data
        else:
            node = Node(len(self._nodes))
            self._nodes.append(node_data)
        return node

    def delete_node(self, node: Node) -> None:
        for offset in self[node]._in_ports:
            self._links.delete_right(node.inp(offset))
        for offset in self[node]._out_ports:
            self._links.delete_left(node.out(offset))
        self._nodes[node.idx] = None
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
        for idx, node_data in enumerate(self._nodes):
            if node_data is not None:
                mapping[Node(idx)] = self.add_node(node_data.op, parent)
        for src, dst in hugr._links.items():
            self.add_link(
                mapping[src.node].out(src.offset), mapping[dst.node].inp(dst.offset)
            )
        return mapping

    def to_serial(self) -> SerialHugr:
        node_it = (node for node in self._nodes if node is not None)
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
    _n_input: int

    def __init__(self, input_types: Sequence[Type]) -> None:
        self._n_input = len(input_types)
        input_types = list(input_types)
        root_op = DummyOp(sops.DFG(parent=-1))
        root_op._serial_op.signature.input = input_types
        # TODO don't assume endo output
        root_op._serial_op.signature.output = input_types
        self.hugr = Hugr(root_op)
        self.input_node = self.hugr.add_node(
            DummyOp(sops.Input(parent=0, types=input_types))
        )
        self.output_node = self.hugr.add_node(
            DummyOp(sops.Output(parent=0, types=input_types))
        )

    def inputs(self) -> list[OutPort]:
        return [self.input_node.out(i) for i in range(self._n_input)]

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
        for i, p in enumerate(ports):
            self.hugr.add_link(p.to_port(), self.output_node.inp(i))
