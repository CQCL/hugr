//! Region-level views of a HUGR.

pub mod petgraph;

use std::iter;

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{Hierarchy, LinkView, MultiPortGraph, PortView, UnmanagedDenseMap};

use crate::{ops::OpType, Direction, Hugr, Node, Port};

use super::{HugrView, NodeMetadata};

type FlatRegionGraph<'g> = portgraph::view::FlatRegion<'g, MultiPortGraph>;

/// Single region view of a HUGR. Includes only the root node and its direct children.
///
/// For a view that includes all the descendants of the root, see [`RegionView`].
#[derive(Clone, Debug)]
pub struct FlatRegionView<'g> {
    /// The chosen root node.
    root: Node,

    /// The graph encoding the adjacency structure of the HUGR.
    graph: FlatRegionGraph<'g>,

    /// The node hierarchy.
    hierarchy: &'g Hierarchy,

    /// Operation types for each node.
    op_types: &'g UnmanagedDenseMap<portgraph::NodeIndex, OpType>,

    /// Metadata types for each node.
    metadata: &'g UnmanagedDenseMap<portgraph::NodeIndex, NodeMetadata>,
}

impl<'g> FlatRegionView<'g> {
    /// Create a new region view of a HUGR containing only a root and its direct
    /// children.
    pub fn new(hugr: &'g Hugr, root: Node) -> Self {
        let Hugr {
            graph,
            hierarchy,
            op_types,
            metadata,
            ..
        } = hugr;
        Self {
            root,
            graph: FlatRegionGraph::new_flat_region(graph, hierarchy, root.index),
            hierarchy,
            op_types,
            metadata,
        }
    }
}

impl<'g> HugrView for FlatRegionView<'g> {
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

    #[inline]
    fn root(&self) -> Node {
        self.root
    }

    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        self.hierarchy
            .parent(node.index)
            .map(Into::into)
            .filter(|&n| n == self.root)
    }

    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        self.op_types.get(node.index)
    }

    #[inline]
    fn get_metadata(&self, node: Node) -> &NodeMetadata {
        self.metadata.get(node.index)
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.hierarchy.child_count(self.root.index) + 1
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
        let children = self.hierarchy.children(self.root.index).map_into();
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
                let port = link.port();
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
            true => self.hierarchy.children(node.index).map_into(),
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

type RegionGraph<'g> = portgraph::view::Region<'g, MultiPortGraph>;

/// Single region view of a HUGR. Includes only the root node and its
/// descendants.
///
/// For a view that includes only the direct children of the root, see
/// [`FlatRegionView`]. Prefer using [`FlatRegionView`] over this type when
/// possible, as it is more efficient.
#[derive(Clone, Debug)]
pub struct RegionView<'g> {
    /// The chosen root node.
    root: Node,

    /// The graph encoding the adjacency structure of the HUGR.
    graph: RegionGraph<'g>,

    /// The node hierarchy.
    hierarchy: &'g Hierarchy,

    /// Operation types for each node.
    op_types: &'g UnmanagedDenseMap<portgraph::NodeIndex, OpType>,

    /// Metadata types for each node.
    metadata: &'g UnmanagedDenseMap<portgraph::NodeIndex, NodeMetadata>,
}

impl<'g> RegionView<'g> {
    /// Create a new region view of a HUGR containing only a root node and its
    /// descendants.
    pub fn new(hugr: &'g Hugr, root: Node) -> Self {
        let Hugr {
            graph,
            hierarchy,
            op_types,
            metadata,
            ..
        } = hugr;
        Self {
            root,
            graph: RegionGraph::new_region(graph, hierarchy, root.index),
            hierarchy,
            op_types,
            metadata,
        }
    }
}

impl<'g> HugrView for RegionView<'g> {
    type Nodes<'a> = MapInto<<RegionGraph<'g> as PortView>::Nodes<'a>, Node>
    where
        Self: 'a;

    type NodePorts<'a> = MapInto<<RegionGraph<'g> as PortView>::NodePortOffsets<'a>, Port>
    where
        Self: 'a;

    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>
    where
        Self: 'a;

    type Neighbours<'a> = MapInto<<RegionGraph<'g> as LinkView>::Neighbours<'a>, Node>
    where
        Self: 'a;

    type PortLinks<'a> = MapWithCtx<
        <RegionGraph<'g> as LinkView>::PortLinks<'a>,
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
        self.hierarchy
            .parent(node.index)
            .filter(|&parent| self.graph.contains_node(parent))
            .map(Into::into)
    }

    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        self.op_types.get(node.index)
    }

    #[inline]
    fn get_metadata(&self, node: Node) -> &NodeMetadata {
        self.metadata.get(node.index)
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
                let port = link.port();
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
            true => self.hierarchy.children(node.index).map_into(),
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

#[cfg(test)]
mod test {
    use crate::{
        builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        ops::{handle::NodeHandle, LeafOp},
        type_row,
        types::{ClassicType, Signature, SimpleType},
    };

    use super::*;

    const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());
    const QB: SimpleType = SimpleType::Qubit;

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

impl<'g> super::view::sealed::HugrInternals for FlatRegionView<'g> {
    type Portgraph = FlatRegionGraph<'g>;

    fn as_portgraph(&self) -> &Self::Portgraph {
        &self.graph
    }
}

impl<'g> super::view::sealed::HugrInternals for RegionView<'g> {
    type Portgraph = RegionGraph<'g>;

    fn as_portgraph(&self) -> &Self::Portgraph {
        &self.graph
    }
}
