from dataclasses import dataclass, field, replace

from collections.abc import Mapping
from enum import Enum
from typing import (
    Iterable,
    Sequence,
    Protocol,
    Generic,
    TypeVar,
    cast,
    overload,
    ClassVar,
)

from typing_extensions import Self

from hugr.serialization.serial_hugr import SerialHugr
from hugr.serialization.ops import BaseOp, OpType as SerialOp
import hugr.serialization.ops as sops
from hugr.serialization.tys import Type
from hugr.utils import BiMap


class Direction(Enum):
    INCOMING = 0
    OUTGOING = 1


@dataclass(frozen=True, eq=True, order=True)
class _Port:
    node: "Node"
    offset: int
    _sub_offset: int = 0

    def next_sub_offset(self) -> Self:
        return replace(self, _sub_offset=self._sub_offset + 1)


@dataclass(frozen=True, eq=True, order=True)
class InPort(_Port):
    direction: ClassVar[Direction] = Direction.INCOMING


class ToPort(Protocol):
    def to_port(self) -> "OutPort": ...


@dataclass(frozen=True, eq=True, order=True)
class OutPort(_Port, ToPort):
    direction: ClassVar[Direction] = Direction.OUTGOING

    def to_port(self) -> "OutPort":
        return self


@dataclass(frozen=True, eq=True, order=True)
class Node(ToPort):
    idx: int
    _num_out_ports: int | None = field(default=None, compare=False)

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
                return [self[i] for i in xs]

    def to_port(self) -> "OutPort":
        return OutPort(self, 0)

    def inp(self, offset: int) -> InPort:
        return InPort(self, offset)

    def out(self, offset: int) -> OutPort:
        return OutPort(self, offset)

    def port(self, offset: int, direction: Direction) -> InPort | OutPort:
        if direction == Direction.INCOMING:
            return self.inp(offset)
        else:
            return self.out(offset)


class Op(Protocol):
    def to_serial(self, node: Node, hugr: "Hugr") -> SerialOp: ...


T = TypeVar("T", bound=BaseOp)


@dataclass()
class DummyOp(Op, Generic[T]):
    _serial_op: T

    def to_serial(self, node: Node, hugr: "Hugr") -> SerialOp:
        return SerialOp(root=self._serial_op)  # type: ignore


class Command(Protocol):
    def op(self) -> Op: ...
    def incoming(self) -> Iterable[ToPort]: ...
    def num_out(self) -> int | None:
        return None


@dataclass()
class NodeData:
    op: Op
    parent: Node | None
    _num_inps: int = 0
    _num_outs: int = 0
    # TODO children field?

    def to_serial(self, node: Node, hugr: "Hugr") -> SerialOp:
        o = self.op.to_serial(node, hugr)
        o.root.parent = self.parent.idx if self.parent else node.idx

        return o


P = TypeVar("P", InPort, OutPort)
K = TypeVar("K", InPort, OutPort)


def _unused_sub_offset(port: P, links: BiMap[OutPort, InPort]) -> P:
    d: dict[OutPort, InPort] | dict[InPort, OutPort]
    match port:
        case OutPort(_):
            d = links.fwd
        case InPort(_):
            d = links.bck
    while port in d:
        port = port.next_sub_offset()
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
        return self.num_nodes()

    def add_node(
        self,
        op: Op,
        parent: Node | None = None,
        num_outs: int | None = None,
    ) -> Node:
        node_data = NodeData(op, parent)

        if self._free_nodes:
            node = self._free_nodes.pop()
            self._nodes[node.idx] = node_data
        else:
            node = Node(len(self._nodes))
            self._nodes.append(node_data)
        return replace(node, _num_out_ports=num_outs)

    def delete_node(self, node: Node) -> NodeData | None:
        for offset in range(self.num_in_ports(node)):
            self._links.delete_right(node.inp(offset))
        for offset in range(self.num_out_ports(node)):
            self._links.delete_left(node.out(offset))

        weight, self._nodes[node.idx] = self._nodes[node.idx], None
        self._free_nodes.append(node)
        return weight

    def add_link(self, src: OutPort, dst: InPort) -> None:
        src = _unused_sub_offset(src, self._links)
        dst = _unused_sub_offset(dst, self._links)
        if self._links.get_left(dst) is not None:
            dst = replace(dst, _sub_offset=dst._sub_offset + 1)
        self._links.insert_left(src, dst)

        self[src.node]._num_outs = max(self[src.node]._num_outs, src.offset + 1)
        self[dst.node]._num_inps = max(self[dst.node]._num_inps, dst.offset + 1)

    def delete_link(self, src: OutPort, dst: InPort) -> None:
        # TODO make sure sub-offset is handled correctly
        self._links.delete_left(src)

    def num_nodes(self) -> int:
        return len(self._nodes) - len(self._free_nodes)

    def num_ports(self, node: Node, direction: Direction) -> int:
        return (
            self.num_in_ports(node)
            if direction == Direction.INCOMING
            else self.num_out_ports(node)
        )

    def num_in_ports(self, node: Node) -> int:
        return self[node]._num_inps

    def num_out_ports(self, node: Node) -> int:
        return self[node]._num_outs

    def _linked_ports(self, port: P, links: dict[P, K]) -> Iterable[K]:
        port = replace(port, _sub_offset=0)
        while port in links:
            # sub offset not used in API
            yield replace(links[port], _sub_offset=0)
            port = port.next_sub_offset()

    # TODO: single linked port

    def _node_links(self, node: Node, links: dict[P, K]) -> Iterable[tuple[P, list[K]]]:
        try:
            direction = next(iter(links.keys())).direction
        except StopIteration:
            return
        # iterate over known offsets
        for offset in range(self.num_ports(node, direction)):
            port = cast(P, node.port(offset, direction))
            yield port, list(self._linked_ports(port, links))

    def outgoing_links(self, node: Node) -> Iterable[tuple[OutPort, list[InPort]]]:
        return self._node_links(node, self._links.fwd)

    def incoming_links(self, node: Node) -> Iterable[tuple[InPort, list[OutPort]]]:
        return self._node_links(node, self._links.bck)

    def num_incoming(self, node: Node) -> int:
        # connecetd links
        return sum(1 for _ in self.incoming_links(node))

    def num_outgoing(self, node: Node) -> int:
        # connecetd links
        return sum(1 for _ in self.outgoing_links(node))

    # TODO: num_links and _linked_ports

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
    root: Node
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
        self.root = self.hugr.root
        self.input_node = self.hugr.add_node(
            DummyOp(sops.Input(parent=0, types=input_types)),
            self.root,
            len(input_types),
        )
        self.output_node = self.hugr.add_node(
            DummyOp(sops.Output(parent=0, types=output_types)), self.root
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

    def add_op(self, op: Op, /, *args: ToPort, num_outs: int | None = None) -> Node:
        new_n = self.hugr.add_node(op, self.root, num_outs=num_outs)
        self._wire_up(new_n, args)
        return new_n

    def add(self, com: Command) -> Node:
        return self.add_op(com.op(), *com.incoming(), num_outs=com.num_out())

    def insert_nested(self, dfg: "Dfg", *args: ToPort) -> Node:
        mapping = self.hugr.insert_hugr(dfg.hugr, self.root)
        self._wire_up(mapping[dfg.root], args)
        return mapping[dfg.root]

    def add_nested(
        self,
        input_types: Sequence[Type],
        output_types: Sequence[Type],
        ports: Iterable[ToPort],
    ) -> "Dfg":
        dfg = Dfg(input_types, output_types)
        mapping = self.hugr.insert_hugr(dfg.hugr, self.root)
        self._wire_up(mapping[dfg.root], ports)
        dfg.hugr = self.hugr
        dfg.input_node = mapping[dfg.input_node]
        dfg.output_node = mapping[dfg.output_node]
        dfg.root = mapping[dfg.root]

        return dfg

    def set_outputs(self, *args: ToPort) -> None:
        self._wire_up(self.output_node, args)

    def make_tuple(self, tys: Sequence[Type], *args: ToPort) -> Node:
        ports = list(args)
        assert len(tys) == len(ports), "Number of types must match number of ports"
        return self.add_op(DummyOp(sops.MakeTuple(parent=0, tys=list(tys))), *args)

    def split_tuple(self, tys: Sequence[Type], port: ToPort) -> list[OutPort]:
        tys = list(tys)
        n = self.add_op(DummyOp(sops.UnpackTuple(parent=0, tys=tys)), port)

        return [n.out(i) for i in range(len(tys))]

    def _wire_up(self, node: Node, ports: Iterable[ToPort]):
        for i, p in enumerate(ports):
            src = p.to_port()
            self.hugr.add_link(src, node.inp(i))
