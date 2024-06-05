from __future__ import annotations
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from enum import Enum
from typing import (
    Iterable,
    Iterator,
    Sequence,
    Protocol,
    Generic,
    TypeVar,
    cast,
    overload,
    ClassVar,
    TYPE_CHECKING,
)

from typing_extensions import Self

from hugr.serialization.serial_hugr import SerialHugr
from hugr.serialization.ops import OpType as SerialOp
from hugr.serialization.tys import Type
from hugr._ops import Op
from hugr.utils import BiMap

if TYPE_CHECKING:
    from ._dfg import Dfg


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


@dataclass(frozen=True, eq=True, order=True)
class Node(Wire):
    idx: int
    _num_out_ports: int | None = field(default=None, compare=False)

    @overload
    def __getitem__(self, index: int) -> OutPort: ...
    @overload
    def __getitem__(self, index: slice) -> Iterator[OutPort]: ...
    @overload
    def __getitem__(self, index: tuple[int, ...]) -> Iterator[OutPort]: ...

    def __getitem__(
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

    def out_port(self) -> "OutPort":
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


@dataclass()
class NodeData:
    op: Op
    parent: Node | None
    _num_inps: int = 0
    _num_outs: int = 0
    # TODO children field?

    def to_serial(self, node: Node, hugr: Hugr) -> SerialOp:
        o = self.op.to_serial(node, self.parent if self.parent else node, hugr)

        return SerialOp(root=o)  # type: ignore[arg-type]


P = TypeVar("P", InPort, OutPort)
K = TypeVar("K", InPort, OutPort)


@dataclass(frozen=True, eq=True, order=True)
class _SubPort(Generic[P]):
    port: P
    sub_offset: int = 0

    def next_sub_offset(self) -> Self:
        return replace(self, sub_offset=self.sub_offset + 1)


_SO = _SubPort[OutPort]
_SI = _SubPort[InPort]


@dataclass()
class Hugr(Mapping[Node, NodeData]):
    root: Node
    _nodes: list[NodeData | None]
    _links: BiMap[_SO, _SI]
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
            self._links.delete_right(_SubPort(node.inp(offset)))
        for offset in range(self.num_out_ports(node)):
            self._links.delete_left(_SubPort(node.out(offset)))

        weight, self._nodes[node.idx] = self._nodes[node.idx], None
        self._free_nodes.append(node)
        return weight

    def _unused_sub_offset(self, port: P) -> _SubPort[P]:
        d: dict[_SO, _SI] | dict[_SI, _SO]
        match port:
            case OutPort(_):
                d = self._links.fwd
            case InPort(_):
                d = self._links.bck
        sub_port = _SubPort(port)
        while sub_port in d:
            sub_port = sub_port.next_sub_offset()
        return sub_port

    def add_link(self, src: OutPort, dst: InPort) -> None:
        src_sub = self._unused_sub_offset(src)
        dst_sub = self._unused_sub_offset(dst)
        # if self._links.get_left(dst_sub) is not None:
        #     dst = replace(dst, _sub_offset=dst._sub_offset + 1)
        self._links.insert_left(src_sub, dst_sub)

        self[src.node]._num_outs = max(self[src.node]._num_outs, src.offset + 1)
        self[dst.node]._num_inps = max(self[dst.node]._num_inps, dst.offset + 1)

    def delete_link(self, src: OutPort, dst: InPort) -> None:
        try:
            sub_offset = next(
                i for i, inp in enumerate(self.linked_ports(src)) if inp == dst
            )
            self._links.delete_left(_SubPort(src, sub_offset))
        except StopIteration:
            return
        # TODO make sure sub-offset is handled correctly

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

    def _linked_ports(
        self, port: P, links: dict[_SubPort[P], _SubPort[K]]
    ) -> Iterable[K]:
        sub_port = _SubPort(port)
        while sub_port in links:
            # sub offset not used in API
            yield links[sub_port].port
            sub_port = sub_port.next_sub_offset()

    @overload
    def linked_ports(self, port: OutPort) -> Iterable[InPort]: ...
    @overload
    def linked_ports(self, port: InPort) -> Iterable[OutPort]: ...
    def linked_ports(self, port: OutPort | InPort):
        match port:
            case OutPort(_):
                return self._linked_ports(port, self._links.fwd)
            case InPort(_):
                return self._linked_ports(port, self._links.bck)

    # TODO: single linked port

    def _node_links(
        self, node: Node, links: dict[_SubPort[P], _SubPort[K]]
    ) -> Iterable[tuple[P, list[K]]]:
        try:
            direction = next(iter(links.keys())).port.direction
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

    def insert_hugr(self, hugr: Hugr, parent: Node | None = None) -> dict[Node, Node]:
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
                mapping[src.port.node].out(src.port.offset),
                mapping[dst.port.node].inp(dst.port.offset),
            )
        return mapping

    def add_dfg(self, input_types: Sequence[Type], output_types: Sequence[Type]) -> Dfg:
        from ._dfg import Dfg

        dfg = Dfg(input_types, output_types)
        mapping = self.insert_hugr(dfg.hugr, self.root)
        dfg.hugr = self
        dfg.input_node = mapping[dfg.input_node]
        dfg.output_node = mapping[dfg.output_node]
        dfg.root = mapping[dfg.root]
        return dfg

    def to_serial(self) -> SerialHugr:
        node_it = (node for node in self._nodes if node is not None)
        return SerialHugr(
            version="v1",
            # non contiguous indices will be erased
            nodes=[node.to_serial(Node(idx), self) for idx, node in enumerate(node_it)],
            edges=[
                (
                    (src.port.node.idx, src.port.offset),
                    (dst.port.node.idx, dst.port.offset),
                )
                for src, dst in self._links.items()
            ],
        )

    @classmethod
    def from_serial(cls, serial: SerialHugr) -> Hugr:
        assert serial.nodes, "Empty Hugr is invalid"

        hugr = Hugr.__new__(Hugr)
        hugr._nodes = []
        hugr._links = BiMap()
        hugr._free_nodes = []
        hugr.root = Node(0)
        for idx, serial_node in enumerate(serial.nodes):
            parent: Node | None = Node(serial_node.root.parent)
            if serial_node.root.parent == idx:
                hugr.root = Node(idx)
                parent = None
            serial_node.root.parent = -1
            hugr._nodes.append(NodeData(serial_node.root.deserialize(), parent))

        for (src_node, src_offset), (dst_node, dst_offset) in serial.edges:
            if src_offset is None or dst_offset is None:
                continue
            hugr.add_link(
                Node(src_node).out(src_offset), Node(dst_node).inp(dst_offset)
            )

        return hugr
