//! SiblingGraph: view onto a sibling subgraph of the HUGR.

use std::iter;

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{LinkView, MultiPortGraph, PortIndex, PortView};

use crate::ops::handle::NodeHandle;
use crate::ops::OpTrait;
use crate::{Direction, Hugr, Node, Port};

use super::{sealed::HugrInternals, HierarchyView, HugrView};

type FlatRegionGraph<'g> = portgraph::view::FlatRegion<'g, &'g MultiPortGraph>;

/// View of a HUGR sibling graph.
///
/// Includes only the root node and its direct children, but no deeper descendants.
/// Uniquely, the root node has no parent.
///
/// See [`DescendantsGraph`] for a view that includes all descendants of the root.
///
/// Implements the [`HierarchyView`] trait, as well as [`HugrView`], it can be
/// used interchangeably with [`DescendantsGraph`].
///
/// [`DescendantsGraph`]: super::DescendantsGraph
#[derive(Clone)]
pub struct SiblingGraph<'g, Root = Node> {
    /// The chosen root node.
    root: Node,

    /// The filtered portgraph encoding the adjacency structure of the HUGR.
    graph: FlatRegionGraph<'g>,

    /// View onto the underlying Hugr which this graph filters
    hugr: &'g Hugr,

    /// The operation type of the root node.
    _phantom: std::marker::PhantomData<Root>,
}

impl<'g, Root> HugrView for SiblingGraph<'g, Root>
where
    Root: NodeHandle,
{
    type RootHandle = Root;

    type Nodes<'a> = iter::Chain<iter::Once<Node>, MapInto<portgraph::hierarchy::Children<'a>, Node>>
    where
        Self: 'a;

    type NodePorts<'a> = MapInto<<FlatRegionGraph<'g> as PortView>::NodePortOffsets<'a>, Port>
    where
        Self: 'a;

    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>
    where
        Self: 'a;

    type Neighbours<'a> = MapInto<<FlatRegionGraph<'g> as LinkView>::Neighbours<'a>, Node>
    where
        Self: 'a;

    type PortLinks<'a> = MapWithCtx<
        <FlatRegionGraph<'g> as LinkView>::PortLinks<'a>,
        &'a Self,
        (Node, Port),
    > where
        Self: 'a;

    type NodeConnections<'a> = MapWithCtx<
        <FlatRegionGraph<'g> as LinkView>::NodeConnections<'a>,
        &'a Self,
       [Port; 2],
    > where
        Self: 'a;

    #[inline]
    fn contains_node(&self, node: Node) -> bool {
        self.graph.contains_node(node.index)
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.base_hugr().hierarchy.child_count(self.root.index) + 1
    }

    #[inline]
    fn edge_count(&self) -> usize {
        // Faster implementation than filtering all the nodes in the internal graph.
        self.nodes()
            .map(|n| self.output_neighbours(n).count())
            .sum()
    }

    #[inline]
    fn nodes(&self) -> Self::Nodes<'_> {
        // Faster implementation than filtering all the nodes in the internal graph.
        let children = self
            .base_hugr()
            .hierarchy
            .children(self.root.index)
            .map_into();
        iter::once(self.root).chain(children)
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        self.graph.port_offsets(node.index, dir).map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        self.graph.all_port_offsets(node.index).map_into()
    }

    fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_> {
        let port = self.graph.port_index(node.index, port.offset).unwrap();
        self.graph
            .port_links(port)
            .with_context(self)
            .map_with_context(|(_, link), region| {
                let port: PortIndex = link.into();
                let node = region.graph.port_node(port).unwrap();
                let offset = region.graph.port_offset(port).unwrap();
                (node.into(), offset.into())
            })
    }

    fn node_connections(&self, node: Node, other: Node) -> Self::NodeConnections<'_> {
        self.graph
            .get_connections(node.index, other.index)
            .with_context(self)
            .map_with_context(|(p1, p2), hugr| {
                [p1, p2].map(|link| {
                    let offset = hugr.graph.port_offset(link).unwrap();
                    offset.into()
                })
            })
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(node.index, dir)
    }

    #[inline]
    fn children(&self, node: Node) -> Self::Children<'_> {
        match node == self.root {
            true => self.base_hugr().hierarchy.children(node.index).map_into(),
            false => portgraph::hierarchy::Children::default().map_into(),
        }
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
    fn get_io(&self, node: Node) -> Option<[Node; 2]> {
        if node == self.root() {
            self.base_hugr().get_io(node)
        } else {
            None
        }
    }
}

impl<'a, Root> HierarchyView<'a> for SiblingGraph<'a, Root>
where
    Root: NodeHandle,
{
    fn new(hugr: &'a impl HugrView, root: Node) -> Self {
        let root_tag = hugr.get_optype(root).tag();
        if !Root::TAG.is_superset(root_tag) {
            // TODO: Return an error
            panic!("Root node must have the correct operation type tag.")
        }
        let hugr = hugr.base_hugr();
        Self {
            root,
            graph: FlatRegionGraph::new_flat_region(&hugr.graph, &hugr.hierarchy, root.index),
            hugr,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, Root> HugrInternals for SiblingGraph<'g, Root>
where
    Root: NodeHandle,
{
    type Portgraph<'p> = &'p FlatRegionGraph<'g> where Self: 'p;

    #[inline]
    fn portgraph(&self) -> Self::Portgraph<'_> {
        &self.graph
    }

    #[inline]
    fn base_hugr(&self) -> &Hugr {
        self.hugr
    }

    #[inline]
    fn root_node(&self) -> Node {
        self.root
    }
}

#[cfg(test)]
mod test {
    use super::super::descendants::test::make_module_hgr;
    use super::*;

    #[test]
    fn flat_region() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, def, inner) = make_module_hgr()?;

        let region: SiblingGraph = SiblingGraph::new(&hugr, def);

        assert_eq!(region.node_count(), 5);
        assert!(region
            .nodes()
            .all(|n| n == def || hugr.get_parent(n) == Some(def)));
        assert_eq!(region.children(inner).count(), 0);

        Ok(())
    }
}
