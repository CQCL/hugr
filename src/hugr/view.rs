#![allow(unused)]
//! A Trait for "read-only" HUGRs.

use std::iter::FusedIterator;
use std::ops::Deref;

use context_iterators::{ContextIterator, IntoContextIterator, MapCtx, MapWithCtx, WithCtx};
use delegate::delegate;
use itertools::{Itertools, MapInto};
use portgraph::dot::{DotFormat, EdgeStyle, NodeStyle, PortStyle};
use portgraph::{multiportgraph, LinkView, MultiPortGraph, PortView};

use super::{Hugr, NodeMetadata};
use super::{Node, Port};
use crate::ops::{OpName, OpType};
use crate::types::EdgeKind;
use crate::Direction;

/// A trait for inspecting HUGRs.
/// For end users we intend this to be superseded by region-specific APIs.
pub trait HugrView: sealed::HugrInternals {
    /// An Iterator over the nodes in a Hugr(View)
    type Nodes<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// An Iterator over (some or all) ports of a node
    type NodePorts<'a>: Iterator<Item = Port>
    where
        Self: 'a;

    /// An Iterator over the children of a node
    type Children<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// An Iterator over (some or all) the nodes neighbouring a node
    type Neighbours<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// Iterator over the children of a node
    type PortLinks<'a>: Iterator<Item = (Node, Port)>
    where
        Self: 'a;

    /// Return index of HUGR root node.
    fn root(&self) -> Node;

    /// Return the type of the HUGR root node.
    fn root_type(&self) -> &OpType {
        self.get_optype(self.root())
    }

    /// Returns the parent of a node.
    fn get_parent(&self, node: Node) -> Option<Node>;

    /// Returns the operation type of a node.
    fn get_optype(&self, node: Node) -> &OpType;

    /// Returns the metadata associated with a node.
    fn get_metadata(&self, node: Node) -> &NodeMetadata;

    /// Returns the number of nodes in the hugr.
    fn node_count(&self) -> usize;

    /// Returns the number of edges in the hugr.
    fn edge_count(&self) -> usize;

    /// Iterates over the nodes in the port graph.
    fn nodes(&self) -> Self::Nodes<'_>;

    /// Iterator over ports of node in a given direction.
    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_>;

    /// Iterator over output ports of node.
    /// Shorthand for [`node_ports`][HugrView::node_ports]`(node, Direction::Outgoing)`.
    #[inline]
    fn node_outputs(&self, node: Node) -> Self::NodePorts<'_> {
        self.node_ports(node, Direction::Outgoing)
    }

    /// Iterator over inputs ports of node.
    /// Shorthand for [`node_ports`][HugrView::node_ports]`(node, Direction::Incoming)`.
    #[inline]
    fn node_inputs(&self, node: Node) -> Self::NodePorts<'_> {
        self.node_ports(node, Direction::Incoming)
    }

    /// Iterator over both the input and output ports of node.
    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_>;

    /// Iterator over the nodes and ports connected to a port.
    fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_>;

    /// Returns whether a port is connected.
    fn is_linked(&self, node: Node, port: Port) -> bool {
        self.linked_ports(node, port).next().is_some()
    }

    /// Number of ports in node for a given direction.
    fn num_ports(&self, node: Node, dir: Direction) -> usize;

    /// Number of inputs to a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Incoming)`.
    #[inline]
    fn num_inputs(&self, node: Node) -> usize {
        self.num_ports(node, Direction::Incoming)
    }

    /// Number of outputs from a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Outgoing)`.
    #[inline]
    fn num_outputs(&self, node: Node) -> usize {
        self.num_ports(node, Direction::Outgoing)
    }

    /// Return iterator over children of node.
    fn children(&self, node: Node) -> Self::Children<'_>;

    /// Iterates over neighbour nodes in the given direction.
    /// May contain duplicates if the graph has multiple links between nodes.
    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_>;

    /// Iterates over the input neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Incoming)`.
    #[inline]
    fn input_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.neighbours(node, Direction::Incoming)
    }

    /// Iterates over the output neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Outgoing)`.
    #[inline]
    fn output_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.neighbours(node, Direction::Outgoing)
    }

    /// Iterates over the input and output neighbours of the `node` in sequence.
    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_>;

    /// Return dot string showing underlying graph and hierarchy side by side.
    fn dot_string(&self) -> String {
        let hugr = self.base_hugr();
        let graph = self.portgraph();
        graph
            .dot_format()
            .with_hierarchy(&hugr.hierarchy)
            .with_node_style(|n| {
                NodeStyle::Box(format!(
                    "({ni}) {name}",
                    ni = n.index(),
                    name = self.get_optype(n.into()).name()
                ))
            })
            .with_port_style(|port| {
                let node = graph.port_node(port).unwrap();
                let optype = self.get_optype(node.into());
                let offset = graph.port_offset(port).unwrap();
                match optype.port_kind(offset).unwrap() {
                    EdgeKind::Static(ty) => {
                        PortStyle::new(html_escape::encode_text(&format!("{}", ty)))
                    }
                    EdgeKind::Value(ty) => {
                        PortStyle::new(html_escape::encode_text(&format!("{}", ty)))
                    }
                    EdgeKind::StateOrder => match graph.port_links(port).count() > 0 {
                        true => PortStyle::text("", false),
                        false => PortStyle::Hidden,
                    },
                    _ => PortStyle::text("", true),
                }
            })
            .with_edge_style(|src, tgt| {
                let src_node = graph.port_node(src).unwrap();
                let src_optype = self.get_optype(src_node.into());
                let src_offset = graph.port_offset(src).unwrap();
                let tgt_node = graph.port_node(tgt).unwrap();

                if hugr.hierarchy.parent(src_node) != hugr.hierarchy.parent(tgt_node) {
                    EdgeStyle::Dashed
                } else if src_optype.port_kind(src_offset) == Some(EdgeKind::StateOrder) {
                    EdgeStyle::Dotted
                } else {
                    EdgeStyle::Solid
                }
            })
            .finish()
    }
}

impl HugrView for Hugr {
    /// An Iterator over the nodes in a Hugr(View)
    type Nodes<'a> = MapInto<multiportgraph::Nodes<'a>, Node> where Self: 'a;

    /// An Iterator over (some or all) ports of a node
    type NodePorts<'a> = MapInto<portgraph::portgraph::NodePortOffsets, Port> where Self: 'a;

    /// An Iterator over the children of a node
    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node> where Self: 'a;

    /// An Iterator over (some or all) the nodes neighbouring a node
    type Neighbours<'a> = MapInto<multiportgraph::Neighbours<'a>, Node> where Self: 'a;

    /// Iterator over the children of a node
    type PortLinks<'a> = MapWithCtx<multiportgraph::PortLinks<'a>, &'a Hugr, (Node, Port)>
    where
        Self: 'a;

    #[inline]
    fn root(&self) -> Node {
        self.root.into()
    }

    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        self.hierarchy.parent(node.index).map(Into::into)
    }

    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        self.op_types.get(node.index)
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.graph.link_count()
    }

    #[inline]
    fn nodes(&self) -> Self::Nodes<'_> {
        self.graph.nodes_iter().map_into()
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        self.graph.port_offsets(node.index, dir).map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        self.graph.all_port_offsets(node.index).map_into()
    }

    #[inline]
    fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_> {
        let port = self.graph.port_index(node.index, port.offset).unwrap();
        self.graph
            .port_links(port)
            .with_context(self)
            .map_with_context(|(_, link), hugr| {
                let port = link.port();
                let node = hugr.graph.port_node(port).unwrap();
                let offset = hugr.graph.port_offset(port).unwrap();
                (node.into(), offset.into())
            })
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(node.index, dir)
    }

    #[inline]
    fn children(&self, node: Node) -> Self::Children<'_> {
        self.hierarchy.children(node.index).map_into()
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_> {
        self.graph.neighbours(node.index, dir).map_into()
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.graph.all_neighbours(node.index).map_into()
    }

    #[inline]
    fn get_metadata(&self, node: Node) -> &NodeMetadata {
        self.metadata.get(node.index)
    }
}

impl<T: HugrView + sealed::HugrInternals> HugrView for &mut T {
    type Nodes<'a> = T::Nodes<'a>
    where
        Self: 'a;

    type NodePorts<'a> = T::NodePorts<'a>
    where
        Self: 'a;

    type Children<'a> = T::Children<'a>
    where
        Self: 'a;

    type Neighbours<'a> = T::Neighbours<'a>
    where
        Self: 'a;

    type PortLinks<'a> = T::PortLinks<'a>
    where
        Self: 'a;

    delegate! {
        to (**self) {
            fn root(&self) -> Node;
            fn get_parent(&self, node: Node) -> Option<Node>;
            fn get_optype(&self, node: Node) -> &OpType;
            fn get_metadata(&self, node: Node) -> &NodeMetadata;
            fn node_count(&self) -> usize;
            fn edge_count(&self) -> usize;
            fn nodes(&self) -> Self::Nodes<'_>;
            fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_>;
            fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_>;
            fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_>;
            fn num_ports(&self, node: Node, dir: Direction) -> usize;
            fn children(&self, node: Node) -> Self::Children<'_>;
            fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_>;
            fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_>;
        }
    }
}

pub(crate) mod sealed {
    use super::*;

    /// Trait for accessing the internals of a Hugr(View).
    ///
    /// Specifically, this trait provides access to the underlying portgraph
    /// view.
    pub trait HugrInternals {
        /// The underlying portgraph view type.
        type Portgraph: LinkView;

        /// Returns a reference to the underlying portgraph.
        fn portgraph(&self) -> &Self::Portgraph;

        /// Returns the Hugr at the base of a chain of views.
        fn base_hugr(&self) -> &Hugr;
    }

    impl HugrInternals for Hugr {
        type Portgraph = MultiPortGraph;

        #[inline]
        fn portgraph(&self) -> &Self::Portgraph {
            &self.graph
        }

        fn base_hugr(&self) -> &Hugr {
            self
        }
    }

    impl<T: HugrInternals> HugrInternals for &mut T {
        type Portgraph = T::Portgraph;
        fn portgraph(&self) -> &Self::Portgraph {
            (**self).portgraph()
        }

        fn base_hugr(&self) -> &Hugr {
            (**self).base_hugr()
        }
    }

    impl<T: HugrInternals> HugrInternals for &T {
        type Portgraph = T::Portgraph;
        fn portgraph(&self) -> &Self::Portgraph {
            (**self).portgraph()
        }

        fn base_hugr(&self) -> &Hugr {
            (**self).base_hugr()
        }
    }
}
