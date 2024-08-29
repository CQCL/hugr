"""Core data structures for HUGR."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
    overload,
)

from hugr._serialization.ops import OpType as SerialOp
from hugr._serialization.serial_hugr import SerialHugr
from hugr.exceptions import ParentBeforeChild
from hugr.ops import Call, Const, Custom, DataflowOp, Module, Op
from hugr.tys import Kind, Type, ValueKind
from hugr.utils import BiMap
from hugr.val import Value

from .node_port import (
    Direction,
    InPort,
    Node,
    NodeIdx,
    OutPort,
    PortOffset,
    ToNode,
    _SubPort,
)

if TYPE_CHECKING:
    import graphviz as gv  # type: ignore[import-untyped]

    from hugr import ext
    from hugr.val import Value

    from .render import RenderConfig


@dataclass()
class NodeData:
    """Node weights in HUGR graph. Defined by an operation and parent node."""

    #: The operation of the node.
    op: Op
    #: The parent node, or None for the root node.
    parent: Node | None
    _num_inps: int = field(default=0, repr=False)
    _num_outs: int = field(default=0, repr=False)
    children: list[Node] = field(default_factory=list, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _to_serial(self, node: Node) -> SerialOp:
        o = self.op._to_serial(self.parent if self.parent else node)

        return SerialOp(root=o)  # type: ignore[arg-type]


_SO = _SubPort[OutPort]
_SI = _SubPort[InPort]

P = TypeVar("P", InPort, OutPort)
K = TypeVar("K", InPort, OutPort)
OpVar = TypeVar("OpVar", bound=Op)
OpVarCov = TypeVar("OpVarCov", bound=Op, covariant=True)


@dataclass()
class Hugr(Mapping[Node, NodeData], Generic[OpVarCov]):
    """The core HUGR datastructure.

    Args:
        root_op: The operation for the root node. Defaults to a Module.

    Examples:
        >>> h = Hugr()
        >>> h.root_op()
        Module()
        >>> h[h.root].op
        Module()
    """

    #: Root node of the HUGR.
    root: Node
    # List of nodes, with None for deleted nodes.
    _nodes: list[NodeData | None]
    # Bidirectional map of links between ports.
    _links: BiMap[_SO, _SI]
    # List of free node indices, populated when nodes are deleted.
    _free_nodes: list[Node]

    def __init__(self, root_op: OpVarCov | None = None) -> None:
        self._free_nodes = []
        self._links = BiMap()
        self._nodes = []
        self.root = self._add_node(root_op or Module(), None, 0)

    def __getitem__(self, key: ToNode) -> NodeData:
        key = key.to_node()
        try:
            n = self._nodes[key.idx]
        except IndexError:
            n = None
        if n is None:
            raise KeyError(key)
        return n

    def __iter__(self) -> Iterator[Node]:
        return (
            Node(idx, data.metadata)
            for idx, data in enumerate(self._nodes)
            if data is not None
        )

    def __len__(self) -> int:
        return self.num_nodes()

    def _get_typed_op(self, node: ToNode, cl: type[OpVar]) -> OpVar:
        op = self[node].op
        assert isinstance(op, cl)
        return op

    def nodes(self) -> Iterable[tuple[Node, NodeData]]:
        """Iterator over nodes of the hugr and their data."""
        return self.items()

    def links(self) -> Iterator[tuple[OutPort, InPort]]:
        """Iterator over all the links in the HUGR.

        Returns:
            Iterator of pairs of outgoing port and the incoming ports.
        """
        return ((src.port, tgt.port) for src, tgt in self._links.items())

    def children(self, node: ToNode | None = None) -> list[Node]:
        """The child nodes of a given `node`.

        Args:
            node: Parent node. Defaults to the HUGR root.

        Returns:
            List of child nodes.

        Examples:
            >>> h = Hugr()
            >>> n = h.add_node(ops.Const(val.TRUE))
            >>> h.children(h.root)
            [Node(1)]
        """
        node = node or self.root
        return self[node].children

    def _add_node(
        self,
        op: Op,
        parent: ToNode | None = None,
        num_outs: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        parent = parent.to_node() if parent else None
        node_data = NodeData(op, parent, metadata=metadata or {})

        if self._free_nodes:
            node = self._free_nodes.pop()
            self._nodes[node.idx] = node_data
        else:
            node = Node(len(self._nodes), {})
            self._nodes.append(node_data)
        node = replace(node, _num_out_ports=num_outs, _metadata=node_data.metadata)
        if parent:
            self[parent].children.append(node)
        return node

    def _update_node_outs(self, node: Node, num_outs: int | None) -> Node:
        """Update the number of outgoing ports for a node.

        Returns:
            The updated node.
        """
        self[node]._num_outs = num_outs or 0
        node = replace(node, _num_out_ports=num_outs)
        parent = self[node].parent
        if parent is not None:
            pos = self[parent].children.index(node)
            self[parent].children[pos] = node
        return node

    def add_node(
        self,
        op: Op,
        parent: ToNode | None = None,
        num_outs: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        """Add a node to the HUGR.

        Args:
            op: Operation of the node.
            parent: Parent node of added node. Defaults to HUGR root if None.
            num_outs: Number of output ports expected for this node. Defaults to None.
            metadata: A dictionary of metadata to associate with the node.
                Defaults to None.

        Returns:
            Handle to the added node.
        """
        parent = parent or self.root
        return self._add_node(op, parent, num_outs, metadata)

    def add_const(
        self,
        value: Value,
        parent: ToNode | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        """Add a constant node to the HUGR.

        Args:
            value: Value of the constant.
            parent: Parent node of added node. Defaults to HUGR root if None.
            metadata: A dictionary of metadata to associate with the node.
                Defaults to None.

        Returns:
            Handle to the added node.

        Examples:
            >>> h = Hugr()
            >>> n = h.add_const(val.TRUE)
            >>> h[n].op
            Const(TRUE)
        """
        return self.add_node(Const(value), parent, metadata=metadata)

    def delete_node(self, node: ToNode) -> NodeData | None:
        """Delete a node from the HUGR.

        Args:
            node: Node to delete.

        Returns:
            The deleted node data, or None if the node was not found.

        Examples:
            >>> h = Hugr()
            >>> n = h.add_const(val.TRUE)
            >>> deleted = h.delete_node(n)
            >>> deleted.op
            Const(TRUE)
            >>> len(h)
            1
        """
        node = node.to_node()
        parent = self[node].parent
        if parent:
            self[parent].children.remove(node)
        for offset in range(self.num_in_ports(node)):
            self._links.delete_right(_SubPort(node.inp(offset)))
        for offset in range(self.num_out_ports(node)):
            self._links.delete_left(_SubPort(node.out(offset)))

        weight, self._nodes[node.idx] = self._nodes[node.idx], None

        # Free up the metadata dictionary
        node = replace(node, _metadata={})

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
        """Add a link (edge) between two nodes to the HUGR,
          from an outgoing port to an incoming port.

        Args:
            src: Source port.
            dst: Destination port.

        Examples:
            >>> df = dfg.Dfg(tys.Bool)
            >>> df.hugr.add_link(df.input_node.out(0), df.output_node.inp(0))
            >>> list(df.hugr.linked_ports(df.input_node[0]))
            [InPort(Node(2), 0)]
        """
        src_sub = self._unused_sub_offset(src)
        dst_sub = self._unused_sub_offset(dst)
        # if self._links.get_left(dst_sub) is not None:
        #     dst = replace(dst, _sub_offset=dst._sub_offset + 1)
        self._links.insert_left(src_sub, dst_sub)

        self[src.node]._num_outs = max(self[src.node]._num_outs, src.offset + 1)
        self[dst.node]._num_inps = max(self[dst.node]._num_inps, dst.offset + 1)

    def delete_link(self, src: OutPort, dst: InPort) -> None:
        """Delete a link (edge) between two nodes from the HUGR.

        Args:
            src: Source port.
            dst: Destination port.
        """
        try:
            sub_offset = next(
                i for i, inp in enumerate(self.linked_ports(src)) if inp == dst
            )
            self._links.delete_left(_SubPort(src, sub_offset))
        except StopIteration:
            return
        # TODO make sure sub-offset is handled correctly

    def root_op(self) -> OpVarCov:
        """The operation of the root node.

        Examples:
            >>> h = Hugr()
            >>> h.root_op()
            Module()
        """
        return cast(OpVarCov, self[self.root].op)

    def num_nodes(self) -> int:
        """The number of nodes in the HUGR.

        Examples:
            >>> h = Hugr()
            >>> n = h.add_const(val.TRUE)
            >>> h.num_nodes()
            2
        """
        return len(self._nodes) - len(self._free_nodes)

    def num_ports(self, node: ToNode, direction: Direction) -> int:
        """The number of ports of a node in a given direction.
        Not necessarily the number of connected ports - if port `i` is
        connected, then all ports `0..i` are assumed to exist.

        Args:
            node: Node to query.
            direction: Direction of ports to count.

        Examples:
            >>> h = Hugr()
            >>> n1 = h.add_const(val.TRUE)
            >>> n2 = h.add_const(val.FALSE)
            >>> h.add_link(n1.out(0), n2.inp(2)) # not a valid link!
            >>> h.num_ports(n1, Direction.OUTGOING)
            1
            >>> h.num_ports(n2, Direction.INCOMING)
            3
        """
        return (
            self.num_in_ports(node)
            if direction == Direction.INCOMING
            else self.num_out_ports(node)
        )

    def num_in_ports(self, node: ToNode) -> int:
        """The number of incoming ports of a node. See :meth:`num_ports`."""
        return self[node]._num_inps

    def num_out_ports(self, node: ToNode) -> int:
        """The number of outgoing ports of a node. See :meth:`num_ports`."""
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
        """Return an iterable of In(Out)Ports linked to given Out(In)Port.

        Args:
            port: Given port.

        Returns:
            Iterator over linked ports.

        Examples:
            >>> df = dfg.Dfg(tys.Bool)
            >>> df.set_outputs(df.input_node[0])
            >>> list(df.hugr.linked_ports(df.input_node[0]))
            [InPort(Node(2), 0)]

        """
        match port:
            case OutPort(_):
                return self._linked_ports(port, self._links.fwd)
            case InPort(_):
                return self._linked_ports(port, self._links.bck)

    # TODO: single linked port

    def outgoing_order_links(self, node: ToNode) -> Iterable[Node]:
        """Iterator over nodes connected by an outgoing state order link from a
        given node.

        Args:
            node: Source node of state order link.

        Examples:
            >>> df = dfg.Dfg()
            >>> df.add_state_order(df.input_node, df.output_node)
            >>> list(df.hugr.outgoing_order_links(df.input_node))
            [Node(2)]
        """
        return (p.node for p in self.linked_ports(node.out(-1)))

    def incoming_order_links(self, node: ToNode) -> Iterable[Node]:
        """Iterator over nodes connected by an incoming state order link to a
        given node.

        Args:
            node: Destination node of state order link.

        Examples:
            >>> df = dfg.Dfg()
            >>> df.add_state_order(df.input_node, df.output_node)
            >>> list(df.hugr.incoming_order_links(df.output_node))
            [Node(1)]
        """
        return (p.node for p in self.linked_ports(node.inp(-1)))

    def _node_links(
        self, node: ToNode, links: dict[_SubPort[P], _SubPort[K]]
    ) -> Iterable[tuple[P, list[K]]]:
        try:
            direction = next(iter(links.keys())).port.direction
        except StopIteration:
            return
        # iterate over known offsets
        for offset in range(self.num_ports(node, direction)):
            port = cast(P, node.port(offset, direction))
            yield port, list(self._linked_ports(port, links))

    def outgoing_links(self, node: ToNode) -> Iterable[tuple[OutPort, list[InPort]]]:
        """Iterator over outgoing links from a given node.

        Args:
            node: Node to query.

        Returns:
            Iterator of pairs of outgoing port and the incoming ports connected
            to that port.

        Examples:
            >>> df = dfg.Dfg()
            >>> df.hugr.add_link(df.input_node.out(0), df.output_node.inp(0))
            >>> df.hugr.add_link(df.input_node.out(0), df.output_node.inp(1))
            >>> list(df.hugr.outgoing_links(df.input_node))
            [(OutPort(Node(1), 0), [InPort(Node(2), 0), InPort(Node(2), 1)])]
        """
        return self._node_links(node, self._links.fwd)

    def incoming_links(self, node: ToNode) -> Iterable[tuple[InPort, list[OutPort]]]:
        """Iterator over incoming links to a given node.

        Args:
            node: Node to query.

        Returns:
            Iterator of pairs of incoming port and the outgoing ports connected
            to that port.

        Examples:
            >>> df = dfg.Dfg()
            >>> df.hugr.add_link(df.input_node.out(0), df.output_node.inp(0))
            >>> df.hugr.add_link(df.input_node.out(0), df.output_node.inp(1))
            >>> list(df.hugr.incoming_links(df.output_node))
            [(InPort(Node(2), 0), [OutPort(Node(1), 0)]), (InPort(Node(2), 1), [OutPort(Node(1), 0)])]
        """  # noqa: E501
        return self._node_links(node, self._links.bck)

    def num_incoming(self, node: Node) -> int:
        """The number of incoming links to a `node`.

        Examples:
            >>> df = dfg.Dfg()
            >>> df.hugr.add_link(df.input_node.out(0), df.output_node.inp(0))
            >>> df.hugr.num_incoming(df.output_node)
            1
        """
        return sum(1 for _ in self.incoming_links(node))

    def num_outgoing(self, node: ToNode) -> int:
        """The number of outgoing links from a `node`.

        Examples:
            >>> df = dfg.Dfg()
            >>> df.hugr.add_link(df.input_node.out(0), df.output_node.inp(0))
            >>> df.hugr.num_outgoing(df.input_node)
            1
        """
        return sum(1 for _ in self.outgoing_links(node))

    # TODO: num_links and _linked_ports

    def port_kind(self, port: InPort | OutPort) -> Kind:
        """The kind of a `port`.

        Examples:
            >>> df = dfg.Dfg(tys.Bool)
            >>> df.hugr.port_kind(df.input_node.out(0))
            ValueKind(Bool)
        """
        return self[port.node].op.port_kind(port)

    def port_type(self, port: InPort | OutPort) -> Type | None:
        """The type of a `port`, if the kind is
        :class:`ValueKind <hugr.tys.ValueKind>`, else None.

        Examples:
            >>> df = dfg.Dfg(tys.Bool)
            >>> df.hugr.port_type(df.input_node.out(0))
            Bool
        """
        op = self[port.node].op
        if isinstance(op, DataflowOp):
            return op.port_type(port)
        if isinstance(op, Call) and isinstance(port, OutPort):
            kind = self.port_kind(port)
            if isinstance(kind, ValueKind):
                return kind.ty
        return None

    def insert_hugr(self, hugr: Hugr, parent: ToNode | None = None) -> dict[Node, Node]:
        """Insert a HUGR into this HUGR.

        Args:
            hugr: HUGR to insert.
            parent: Parent for root of inserted HUGR. Defaults to None.

        Returns:
            Mapping from node indices in inserted HUGR to their new indices
            in this HUGR.

        Examples:
            >>> d = dfg.Dfg()
            >>> h = Hugr()
            >>> h.insert_hugr(d.hugr)
            {Node(0): Node(1), Node(1): Node(2), Node(2): Node(3)}
        """
        mapping: dict[Node, Node] = {}

        for node, node_data in hugr.nodes():
            # relies on parents being inserted before any children
            try:
                node_parent = mapping[node_data.parent] if node_data.parent else parent
            except KeyError as e:
                raise ParentBeforeChild from e
            mapping[node] = self.add_node(
                node_data.op,
                node_parent,
                num_outs=node_data._num_outs,
                metadata=node_data.metadata,
            )

        for src, dst in hugr._links.items():
            self.add_link(
                mapping[src.port.node].out(src.port.offset),
                mapping[dst.port.node].inp(dst.port.offset),
            )
        return mapping

    def _to_serial(self) -> SerialHugr:
        """Serialize the HUGR."""
        node_it = (node for node in self._nodes if node is not None)

        def _serialize_link(
            link: tuple[_SO, _SI],
        ) -> tuple[tuple[NodeIdx, PortOffset], tuple[NodeIdx, PortOffset]]:
            src, dst = link
            s, d = self._constrain_offset(src.port), self._constrain_offset(dst.port)
            return (src.port.node.idx, s), (dst.port.node.idx, d)

        return SerialHugr(
            # non contiguous indices will be erased
            nodes=[node._to_serial(Node(idx, {})) for idx, node in enumerate(node_it)],
            edges=[_serialize_link(link) for link in self._links.items()],
            metadata=[node.metadata if node.metadata else None for node in node_it],
        )

    def _constrain_offset(self, p: P) -> PortOffset:
        # negative offsets are used to refer to the last port
        if p.offset < 0:
            match p.direction:
                case Direction.INCOMING:
                    current = self.num_incoming(p.node)
                case Direction.OUTGOING:
                    current = self.num_outgoing(p.node)
            offset = current + p.offset + 1
        else:
            offset = p.offset

        return offset

    def resolve_extensions(self, registry: ext.ExtensionRegistry) -> Hugr:
        """Resolve extension types and operations in the HUGR by matching them to
        extensions in the registry.
        """
        for node in self:
            op = self[node].op
            if isinstance(op, Custom):
                self[node].op = op.resolve(registry)
        return self

    @classmethod
    def _from_serial(cls, serial: SerialHugr) -> Hugr:
        """Load a HUGR from a serialized form."""
        assert serial.nodes, "Empty Hugr is invalid"

        hugr = Hugr.__new__(Hugr)
        hugr._nodes = []
        hugr._links = BiMap()
        hugr._free_nodes = []
        hugr.root = Node(0)

        def get_meta(idx: int) -> dict[str, Any]:
            if not serial.metadata:
                return {}
            if idx < len(serial.metadata):
                return serial.metadata[idx] or {}
            return {}

        for idx, serial_node in enumerate(serial.nodes):
            node_meta = get_meta(idx)
            parent: Node | None = Node(serial_node.root.parent)
            if serial_node.root.parent == idx:
                hugr.root = Node(idx, _metadata=node_meta)
                parent = None

            serial_node.root.parent = -1
            n = hugr._add_node(
                serial_node.root.deserialize(), parent, metadata=node_meta
            )
            assert n.idx == idx, "Nodes should be added contiguously"

        for (src_node, src_offset), (dst_node, dst_offset) in serial.edges:
            if src_offset is None or dst_offset is None:
                continue
            hugr.add_link(
                Node(src_node, _metadata=get_meta(src_node)).out(src_offset),
                Node(dst_node, _metadata=get_meta(dst_node)).inp(dst_offset),
            )

        return hugr

    def to_json(self) -> str:
        """Serialize the HUGR to a JSON string."""
        return self._to_serial().to_json()

    @classmethod
    def load_json(cls, json_str: str) -> Hugr:
        """Deserialize a JSON string into a HUGR."""
        json_dict = json.loads(json_str)
        serial = SerialHugr.load_json(json_dict)
        return cls._from_serial(serial)

    def render_dot(self, config: RenderConfig | None = None) -> gv.Digraph:
        """Render the HUGR to a graphviz Digraph.

        Args:
            config: Render configuration.

        Returns:
            The graphviz Digraph.
        """
        from .render import DotRenderer

        return DotRenderer(config).render(self)

    def store_dot(
        self, filename: str, format: str = "svg", config: RenderConfig | None = None
    ) -> None:
        """Render the HUGR to a graphviz dot file.

        Args:
            filename: The file to render to.
            format: The format used for rendering ('pdf', 'png', etc.).
                Defaults to SVG.
            config: Render configuration.
        """
        from .render import DotRenderer

        DotRenderer(config).store(self, filename=filename, format=format)
