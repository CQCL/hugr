//! Hierarchical views for HUGR.
//!
//! Views into subgraphs of HUGRs that are based on the hierarchical relationship
//! of the HUGR nodes. Such a subgraph includes a root node and some of its
//! descendants. The root node is the only node in the view that has no parent.
//!
//! There are currently 2 hierarchical views:
//!  - [`SiblingGraph`]: A view of the subgraph induced by the children
//!   of the root node.
//! - [`DescendantsGraph`]: A view of the subgraph induced by all the
//!   descendants of the root node.
//!
//! Both views implement the [`HierarchyView`] trait, so they can be used
//! interchangeably. They implement [`HugrView`] as well as petgraph's _visit_
//! traits.

pub mod petgraph;

use std::iter;

use ::petgraph::visit as pv;
use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{LinkView, PortIndex, PortView};

use crate::ops::handle::NodeHandle;
use crate::ops::OpTrait;
use crate::{hugr::NodeType, hugr::OpType, Direction, Hugr, Node, Port};

use super::{sealed::HugrInternals, HugrView, NodeMetadata};

type FlatRegionGraph<'g, Base> =
    portgraph::view::FlatRegion<'g, <Base as HugrInternals>::Portgraph>;

/// View of a HUGR sibling graph.
///
/// Includes only the root node and its direct children.
///
/// For a view that includes all the descendants of the root, see [`DescendantsGraph`].
pub struct SiblingGraph<'g, Root = Node, Base = Hugr>
where
    Base: HugrInternals,
{
    /// The chosen root node.
    root: Node,

    /// The filtered portgraph encoding the adjacency structure of the HUGR.
    graph: FlatRegionGraph<'g, Base>,

    /// The rest of the HUGR.
    hugr: &'g Base,

    /// The operation type of the root node.
    _phantom: std::marker::PhantomData<Root>,
}

impl<'g, Root, Base> Clone for SiblingGraph<'g, Root, Base>
where
    Root: NodeHandle,
    Base: HugrInternals + HugrView + Clone,
{
    fn clone(&self) -> Self {
        SiblingGraph::new(self.hugr, self.root)
    }
}

impl<'g, Root, Base> HugrView for SiblingGraph<'g, Root, Base>
where
    Root: NodeHandle,
    Base: HugrInternals + HugrView,
{
    type RootHandle = Root;

    type Nodes<'a> = iter::Chain<iter::Once<Node>, MapInto<portgraph::hierarchy::Children<'a>, Node>>
    where
        Self: 'a;

    type NodePorts<'a> = MapInto<<FlatRegionGraph<'g, Base> as PortView>::NodePortOffsets<'a>, Port>
    where
        Self: 'a;

    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>
    where
        Self: 'a;

    type Neighbours<'a> = MapInto<<FlatRegionGraph<'g, Base> as LinkView>::Neighbours<'a>, Node>
    where
        Self: 'a;

    type PortLinks<'a> = MapWithCtx<
        <FlatRegionGraph<'g, Base> as LinkView>::PortLinks<'a>,
        &'a Self,
        (Node, Port),
    > where
        Self: 'a;

    #[inline]
    fn root(&self) -> Node {
        self.root
    }

    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        self.hugr.get_parent(node).filter(|&n| n == self.root)
    }

    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        self.hugr.get_optype(node)
    }

    #[inline]
    fn get_nodetype(&self, node: Node) -> &NodeType {
        self.hugr.get_nodetype(node)
    }

    #[inline]
    fn get_metadata(&self, node: Node) -> &NodeMetadata {
        self.hugr.get_metadata(node)
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

type RegionGraph<'g, Base> = portgraph::view::Region<'g, <Base as HugrInternals>::Portgraph>;

/// View of a HUGR descendants graph.
///
/// Includes the root node and all its descendants nodes at any depth.
///
/// For a view that includes only the direct children of the root, see
/// [`SiblingGraph`]. Prefer using [`SiblingGraph`] over this type when
/// possible, as it is more efficient.
pub struct DescendantsGraph<'g, Root = Node, Base = Hugr>
where
    Base: HugrInternals,
{
    /// The chosen root node.
    root: Node,

    /// The graph encoding the adjacency structure of the HUGR.
    graph: RegionGraph<'g, Base>,

    /// The node hierarchy.
    hugr: &'g Base,

    /// The operation handle of the root node.
    _phantom: std::marker::PhantomData<Root>,
}

impl<'g, Root, Base: Clone> Clone for DescendantsGraph<'g, Root, Base>
where
    Root: NodeHandle,
    Base: HugrInternals + HugrView,
{
    fn clone(&self) -> Self {
        DescendantsGraph::new(self.hugr, self.root)
    }
}

impl<'g, Root, Base> HugrView for DescendantsGraph<'g, Root, Base>
where
    Root: NodeHandle,
    Base: HugrInternals + HugrView,
{
    type RootHandle = Root;

    type Nodes<'a> = MapInto<<RegionGraph<'g, Base> as PortView>::Nodes<'a>, Node>
    where
        Self: 'a;

    type NodePorts<'a> = MapInto<<RegionGraph<'g, Base> as PortView>::NodePortOffsets<'a>, Port>
    where
        Self: 'a;

    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>
    where
        Self: 'a;

    type Neighbours<'a> = MapInto<<RegionGraph<'g, Base> as LinkView>::Neighbours<'a>, Node>
    where
        Self: 'a;

    type PortLinks<'a> = MapWithCtx<
        <RegionGraph<'g, Base> as LinkView>::PortLinks<'a>,
        &'a Self,
        (Node, Port),
    > where
        Self: 'a;

    #[inline]
    fn root(&self) -> Node {
        self.root
    }

    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        self.hugr
            .get_parent(node)
            .filter(|&parent| self.graph.contains_node(parent.index))
            .map(Into::into)
    }

    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        self.hugr.get_optype(node)
    }

    #[inline]
    fn get_nodetype(&self, node: Node) -> &NodeType {
        self.hugr.get_nodetype(node)
    }

    #[inline]
    fn get_metadata(&self, node: Node) -> &NodeMetadata {
        self.hugr.get_metadata(node)
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

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(node.index, dir)
    }

    #[inline]
    fn children(&self, node: Node) -> Self::Children<'_> {
        match self.graph.contains_node(node.index) {
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
        self.base_hugr().get_io(node)
    }
}

/// A common trait for views of a HUGR hierarchical subgraph.
pub trait HierarchyView<'a>:
    HugrView
    + pv::GraphBase<NodeId = Node>
    + pv::GraphProp
    + pv::NodeCount
    + pv::NodeIndexable
    + pv::EdgeCount
    + pv::Visitable
    + pv::GetAdjacencyMatrix
    + pv::Visitable
where
    for<'g> &'g Self: pv::IntoNeighborsDirected + pv::IntoNodeIdentifiers,
{
    /// The base from which the subgraph is derived.
    type Base;

    /// Create a hierarchical view of a HUGR given a root node.
    fn new(hugr: &'a Self::Base, root: Node) -> Self;
}

impl<'a, Root, Base> HierarchyView<'a> for SiblingGraph<'a, Root, Base>
where
    Root: NodeHandle,
    Base: HugrInternals + HugrView,
{
    type Base = Base;

    fn new(hugr: &'a Base, root: Node) -> Self {
        let root_tag = hugr.get_optype(root).tag();
        if !Root::TAG.is_superset(root_tag) {
            // TODO: Return an error
            panic!("Root node must have the correct operation type tag.")
        }
        Self {
            root,
            graph: FlatRegionGraph::<Base>::new_flat_region(
                hugr.portgraph(),
                &hugr.base_hugr().hierarchy,
                root.index,
            ),
            hugr,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, Root, Base> HierarchyView<'a> for DescendantsGraph<'a, Root, Base>
where
    Root: NodeHandle,
    Base: HugrInternals + HugrView,
{
    type Base = Base;

    fn new(hugr: &'a Base, root: Node) -> Self {
        let root_tag = hugr.get_optype(root).tag();
        if !Root::TAG.is_superset(root_tag) {
            // TODO: Return an error
            panic!("Root node must have the correct operation type tag.")
        }
        Self {
            root,
            graph: RegionGraph::<Base>::new_region(
                hugr.portgraph(),
                &hugr.base_hugr().hierarchy,
                root.index,
            ),
            hugr,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'g, Root, Base> super::sealed::HugrInternals for SiblingGraph<'g, Root, Base>
where
    Root: NodeHandle,
    Base: HugrInternals,
{
    type Portgraph = FlatRegionGraph<'g, Base>;

    #[inline]
    fn portgraph(&self) -> &Self::Portgraph {
        &self.graph
    }

    #[inline]
    fn base_hugr(&self) -> &Hugr {
        self.hugr.base_hugr()
    }
}

impl<'g, Root, Base> super::sealed::HugrInternals for DescendantsGraph<'g, Root, Base>
where
    Root: NodeHandle,
    Base: HugrInternals,
{
    type Portgraph = RegionGraph<'g, Base>;

    #[inline]
    fn portgraph(&self) -> &Self::Portgraph {
        &self.graph
    }

    #[inline]
    fn base_hugr(&self) -> &Hugr {
        self.hugr.base_hugr()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        ops::handle::NodeHandle,
        std_extensions::quantum::test::h_gate,
        type_row,
        types::{AbstractSignature, Type},
    };

    use super::*;

    const NAT: Type = crate::extension::prelude::USIZE_T;
    const QB: Type = crate::extension::prelude::QB_T;

    /// Make a module hugr with a fn definition containing an inner dfg node.
    ///
    /// Returns the hugr, the fn node id, and the nested dgf node id.
    fn make_module_hgr() -> Result<(Hugr, Node, Node), Box<dyn std::error::Error>> {
        let mut module_builder = ModuleBuilder::new();

        let (f_id, inner_id) = {
            let mut func_builder = module_builder.define_function(
                "main",
                AbstractSignature::new_df(type_row![NAT, QB], type_row![NAT, QB]).pure(),
            )?;

            let [int, qb] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb])?;

            let inner_id = {
                let inner_builder = func_builder.dfg_builder(
                    AbstractSignature::new_df(type_row![NAT], type_row![NAT]),
                    None,
                    [int],
                )?;
                let w = inner_builder.input_wires();
                inner_builder.finish_with_outputs(w)
            }?;

            let f_id =
                func_builder.finish_with_outputs(inner_id.outputs().chain(q_out.outputs()))?;
            (f_id, inner_id)
        };
        let hugr = module_builder.finish_hugr()?;
        Ok((hugr, f_id.handle().node(), inner_id.handle().node()))
    }

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

    #[test]
    fn full_region() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, def, inner) = make_module_hgr()?;

        let region: DescendantsGraph = DescendantsGraph::new(&hugr, def);

        assert_eq!(region.node_count(), 7);
        assert!(region.nodes().all(|n| n == def
            || hugr.get_parent(n) == Some(def)
            || hugr.get_parent(n) == Some(inner)));
        assert_eq!(region.children(inner).count(), 2);

        Ok(())
    }
}
