from dataclasses import dataclass, replace

from collections.abc import Collection, Mapping
from typing import Iterable, Sequence, Protocol, Generic, TypeVar, overload

from hugr.serialization.serial_hugr import SerialHugr
from hugr.serialization.ops import BaseOp, OpType as SerialOp
import hugr.serialization.ops as sops
from hugr.serialization.tys import Type
from hugr.utils import BiMap


@dataclass(frozen=True, eq=True, order=True)
class Port:
    node: "Node"
    offset: int
    sub_offset: int = 0


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

    @overload
    def __getitem__(self, index: int) -> OutPort: ...
    @overload
    def __getitem__(self, index: slice) -> Iterable[OutPort]: ...
    @overload
    def __getitem__(self, index: tuple[int, ...]) -> list[OutPort]: ...

    def __getitem__(
        self, index: int | slice | tuple[int, ...]
    ) -> OutPort | Iterable[OutPort]:
        match index:
            case int(index):
                return self.out(index)
            case slice():
                start = index.start or 0
                stop = index.stop
                if stop is None:
                    raise ValueError("Stop must be specified")
                step = index.step or 1
                return (self[i] for i in range(start, stop, step))
            case tuple(xs):
                return [self[i] for i in xs]

    def to_port(self) -> "OutPort":
        return OutPort(self, 0)

    def inp(self, offset: int) -> InPort:
        return InPort(self, offset)

    def out(self, offset: int) -> OutPort:
        return OutPort(self, offset)


class Op(Protocol):
    def to_serial(self, node: Node, hugr: "Hugr") -> SerialOp: ...


T = TypeVar("T", bound=BaseOp)


@dataclass()
class DummyOp(Op, Generic[T]):
    _serial_op: T

    def to_serial(self, node: Node, hugr: "Hugr") -> SerialOp:
        return SerialOp(root=self._serial_op)  # type: ignore


@dataclass()
class NodeData:
    op: Op
    parent: Node | None
    # TODO children field?

    def to_serial(self, node: Node, hugr: "Hugr") -> SerialOp:
        o = self.op.to_serial(node, hugr)
        o.root.parent = self.parent.idx if self.parent else node.idx

        return o


P = TypeVar("P", InPort, OutPort)


def _unused_sub_offset(port: P, links: BiMap[OutPort, InPort]) -> P:
    d: dict[OutPort, InPort] | dict[InPort, OutPort]
    match port:
        case OutPort(_):
            d = links.fwd
        case InPort(_):
            d = links.bck
    while port in d:
        port = replace(port, sub_offset=port.sub_offset + 1)
    return port


@dataclass()
class Hugr(Mapping[Node, NodeData]):
    root: Node
    _nodes: list[NodeData | None]
    _links: BiMap[OutPort, InPort]
    _free_nodes: list[Node]

    def __init__(self, root_op: Op) -> None:
        self._free_nodes = []
        self._links = BiMap()
        self._nodes = []
        self.root = self.add_node(root_op)

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

    def add_node(
        self,
        op: Op,
        parent: Node | None = None,
    ) -> Node:
        node_data = NodeData(op, parent)

        if self._free_nodes:
            node = self._free_nodes.pop()
            self._nodes[node.idx] = node_data
        else:
            node = Node(len(self._nodes))
            self._nodes.append(node_data)
        return node

    def delete_node(self, node: Node) -> None:
        for offset in range(self.num_in_ports(node)):
            self._links.delete_right(node.inp(offset))
        for offset in range(self.num_out_ports(node)):
            self._links.delete_left(node.out(offset))
        self._nodes[node.idx] = None
        self._free_nodes.append(node)

    def add_link(self, src: OutPort, dst: InPort, ty: Type | None = None) -> None:
        src = _unused_sub_offset(src, self._links)
        dst = _unused_sub_offset(dst, self._links)
        if self._links.get_left(dst) is not None:
            dst = replace(dst, sub_offset=dst.sub_offset + 1)
        self._links.insert_left(src, dst)

    def delete_link(self, src: OutPort, dst: InPort) -> None:
        self._links.delete_left(src)

    def num_in_ports(self, node: Node) -> int:
        return len(self.in_ports(node))

    def num_out_ports(self, node: Node) -> int:
        return len(self.out_ports(node))

    def in_ports(self, node: Node) -> Collection[InPort]:
        # can be optimised by caching number of ports
        # or by guaranteeing that all ports are contiguous
        return [p for p in self._links.bck if p.node == node]

    def out_ports(self, node: Node) -> Collection[OutPort]:
        return [p for p in self._links.fwd if p.node == node]

    def insert_hugr(self, hugr: "Hugr", parent: Node | None = None) -> dict[Node, Node]:
        mapping: dict[Node, Node] = {}

        for idx, node_data in enumerate(hugr._nodes):
            if node_data is not None:
                mapping[Node(idx)] = self.add_node(node_data.op, node_data.parent)

        for new_node in mapping.values():
            # update mapped parent
            node_data = self[new_node]
            node_data.parent = mapping[node_data.parent] if node_data.parent else parent

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
            nodes=[node.to_serial(Node(idx), self) for idx, node in enumerate(node_it)],
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

    def __init__(
        self, input_types: Sequence[Type], output_types: Sequence[Type]
    ) -> None:
        input_types = list(input_types)
        output_types = list(output_types)
        root_op = DummyOp(sops.DFG(parent=-1))
        root_op._serial_op.signature.input = input_types
        root_op._serial_op.signature.output = output_types
        self.hugr = Hugr(root_op)
        self.input_node = self.hugr.add_node(
            DummyOp(sops.Input(parent=0, types=input_types)), self.hugr.root
        )
        self.output_node = self.hugr.add_node(
            DummyOp(sops.Output(parent=0, types=output_types)), self.hugr.root
        )

    @classmethod
    def endo(cls, types: Sequence[Type]) -> "Dfg":
        return Dfg(types, types)

    def _input_op(self) -> DummyOp[sops.Input]:
        dop = self.hugr[self.input_node].op
        assert isinstance(dop, DummyOp)
        assert isinstance(dop._serial_op, sops.Input)
        return dop

    def inputs(self) -> list[OutPort]:
        return [
            self.input_node.out(i)
            for i in range(len(self._input_op()._serial_op.types))
        ]

    def add_op(self, op: Op, ports: Iterable[ToPort]) -> Node:
        new_n = self.hugr.add_node(op, self.hugr.root)
        self._wire_up(new_n, ports)
        return new_n

    def insert_nested(self, dfg: "Dfg", ports: Iterable[ToPort]) -> Node:
        mapping = self.hugr.insert_hugr(dfg.hugr, self.hugr.root)
        self._wire_up(mapping[dfg.hugr.root], ports)
        return mapping[dfg.hugr.root]

    def add_nested(self, ports: Iterable[ToPort]) -> "Dfg":
        dfg = Dfg.__new__(Dfg)
        dfg.hugr = self.hugr
        # I/O nodes

        return dfg

    def set_outputs(self, ports: Iterable[ToPort]) -> None:
        self._wire_up(self.output_node, ports)

    def make_tuple(self, ports: Iterable[ToPort], tys: Sequence[Type]) -> Node:
        ports = list(ports)
        assert len(tys) == len(ports), "Number of types must match number of ports"
        return self.add_op(DummyOp(sops.MakeTuple(parent=0, tys=list(tys))), ports)

    def split_tuple(self, port: ToPort, tys: Sequence[Type]) -> list[OutPort]:
        tys = list(tys)
        n = self.add_op(DummyOp(sops.UnpackTuple(parent=0, tys=tys)), [port])

        return [n.out(i) for i in range(len(tys))]

    def _wire_up(self, node: Node, ports: Iterable[ToPort]):
        for i, p in enumerate(ports):
            src = p.to_port()
            self.hugr.add_link(src, node.inp(i))
