//! Region-level views of a HUGR.
//!
//! A region is a subgraph of a HUGR that includes a root node and some of its
//! descendants. The root node is the only node in the region that has no parent
//! in the region. Non-local edges between nodes inside and outside the region
//! are ignored.
//!
//! [`FlatRegionView`] includes only the root node and its direct children,
//! while [`RegionView`] includes all the descendants of the root.
//!
//! Both views implement the [`Region`] trait, so they can be used
//! interchangeably. They implement [`HugrView`] as well as petgraph's _visit_
//! traits.

pub mod petgraph;

use std::iter;

use ::petgraph::visit as pv;
use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{LinkView, PortIndex, PortView};

use crate::{ops::OpType, Direction, Hugr, Node, Port};

use super::view::sealed::HugrInternals;
use super::{HugrView, NodeMetadata};

type FlatRegionGraph<'g, Base> =
    portgraph::view::FlatRegion<'g, <Base as HugrInternals>::Portgraph>;

/// Single region view of a HUGR. Includes only the root node and its direct children.
///
/// For a view that includes all the descendants of the root, see [`RegionView`].
pub struct FlatRegionView<'g, Base = Hugr>
where
    Base: HugrInternals,
{
    /// The chosen root node.
    root: Node,

    /// The filtered portgraph encoding the adjacency structure of the HUGR.
    graph: FlatRegionGraph<'g, Base>,

    /// The rest of the HUGR.
    hugr: &'g Base,
}

impl<'g, Base> Clone for FlatRegionView<'g, Base>
where
    Base: HugrInternals + HugrView + Clone,
{
    fn clone(&self) -> Self {
        FlatRegionView::new(self.hugr, self.root)
    }
}

impl<'g, Base> HugrView for FlatRegionView<'g, Base>
where
    Base: HugrInternals + HugrView,
{
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
}

type RegionGraph<'g, Base> = portgraph::view::Region<'g, <Base as HugrInternals>::Portgraph>;

/// Single region view of a HUGR. Includes only the root node and its
/// descendants.
///
/// For a view that includes only the direct children of the root, see
/// [`FlatRegionView`]. Prefer using [`FlatRegionView`] over this type when
/// possible, as it is more efficient.
pub struct RegionView<'g, Base>
where
    Base: HugrInternals,
{
    /// The chosen root node.
    root: Node,

    /// The graph encoding the adjacency structure of the HUGR.
    graph: RegionGraph<'g, Base>,

    /// The node hierarchy.
    hugr: &'g Base,
}

impl<'g, Base: Clone> Clone for RegionView<'g, Base>
where
    Base: HugrInternals + HugrView,
{
    fn clone(&self) -> Self {
        RegionView::new(self.hugr, self.root)
    }
}

impl<'g, Base> HugrView for RegionView<'g, Base>
where
    Base: HugrInternals + HugrView,
{
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
}

/// A common trait for views of a hugr region.
pub trait Region<'a>:
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
    /// The base from which the region is derived.
    type Base;

    /// Create a region view of a HUGR given a root node.
    fn new(hugr: &'a Self::Base, root: Node) -> Self;
}

impl<'a, Base> Region<'a> for FlatRegionView<'a, Base>
where
    Base: HugrInternals + HugrView,
{
    type Base = Base;

    fn new(hugr: &'a Base, root: Node) -> Self {
        Self {
            root,
            graph: FlatRegionGraph::<Base>::new_flat_region(
                hugr.portgraph(),
                &hugr.base_hugr().hierarchy,
                root.index,
            ),
            hugr,
        }
    }
}

impl<'a, Base> Region<'a> for RegionView<'a, Base>
where
    Base: HugrInternals + HugrView,
{
    type Base = Base;

    fn new(hugr: &'a Base, root: Node) -> Self {
        Self {
            root,
            graph: RegionGraph::<Base>::new_region(
                hugr.portgraph(),
                &hugr.base_hugr().hierarchy,
                root.index,
            ),
            hugr,
        }
    }
}

impl<'g, Base> super::view::sealed::HugrInternals for FlatRegionView<'g, Base>
where
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

impl<'g, Base> super::view::sealed::HugrInternals for RegionView<'g, Base>
where
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
        ops::{handle::NodeHandle, LeafOp},
        type_row,
        types::{ClassicType, LinearType, Signature, SimpleType},
    };

    use super::*;

    const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());
    const QB: SimpleType = SimpleType::Linear(LinearType::Qubit);

    /// Make a module hugr with a fn definition containing an inner dfg node.
    ///
    /// Returns the hugr, the fn node id, and the nested dgf node id.
    fn make_module_hgr() -> Result<(Hugr, Node, Node), Box<dyn std::error::Error>> {
        let mut module_builder = ModuleBuilder::new();

        let (f_id, inner_id) = {
            let mut func_builder = module_builder.define_function(
                "main",
                Signature::new_df(type_row![NAT, QB], type_row![NAT, QB]),
            )?;

            let [int, qb] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(LeafOp::H, vec![qb])?;

            let inner_id = {
                let inner_builder = func_builder
                    .dfg_builder(Signature::new_df(type_row![NAT], type_row![NAT]), [int])?;
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

        let region = FlatRegionView::new(&hugr, def);

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

        let region = RegionView::new(&hugr, def);

        assert_eq!(region.node_count(), 7);
        assert!(region.nodes().all(|n| n == def
            || hugr.get_parent(n) == Some(def)
            || hugr.get_parent(n) == Some(inner)));
        assert_eq!(region.children(inner).count(), 2);

        Ok(())
    }
}
