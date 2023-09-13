//! SiblingGraph: view onto a sibling subgraph of the HUGR.

use std::iter;

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{multiportgraph, LinkView, MultiPortGraph, PortIndex, PortView};

use crate::hugr::hugrmut::sealed::HugrMutInternals;
use crate::hugr::{HugrError, HugrMut};
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

    /// The underlying Hugr onto which this view is a filter
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
    fn try_new(hugr: &'a impl HugrView, root: Node) -> Result<Self, HugrError> {
        hugr.valid_node(root)?;
        if !Root::TAG.is_superset(hugr.get_optype(root).tag()) {
            return Err(HugrError::InvalidNode(root));
        }
        let hugr = hugr.base_hugr();
        Ok(Self {
            root,
            graph: FlatRegionGraph::new_flat_region(&hugr.graph, &hugr.hierarchy, root.index),
            hugr,
            _phantom: std::marker::PhantomData,
        })
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

/// Mutable view onto a HUGR sibling graph.
///
/// Includes only the root node and its direct children, but no deeper descendants.
/// Uniquely, the root node has no parent.
///
/// [HugrView] methods may be slower than for an immutable [SiblingGraph] as the latter
/// may cache information about the graph connectivity, whereas (in order
/// to ease mutation) this does not.
pub struct SiblingMut<'g, Root = Node> {
    /// The chosen root node.
    root: Node,

    /// The rest of the HUGR.
    hugr: &'g mut Hugr,

    /// The operation type of the root node.
    _phantom: std::marker::PhantomData<Root>,
}

impl<'g, Root: NodeHandle> SiblingMut<'g, Root> {
    /// Create a new SiblingMut from a base.
    /// Equivalent to [HierarchyView::try_new] but takes a *mutable* reference.
    pub fn try_new(hugr: &'g mut impl HugrMut, root: Node) -> Result<Self, HugrError> {
        hugr.valid_node(root)?;
        if !Root::TAG.is_superset(hugr.get_optype(root).tag()) {
            return Err(HugrError::InvalidNode(root));
        }
        Ok(Self {
            hugr: hugr.hugr_mut(),
            root,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<'g, Root: NodeHandle> HugrInternals for SiblingMut<'g, Root> {
    type Portgraph<'p> = FlatRegionGraph<'p> where 'g: 'p, Root: 'p;

    fn portgraph(&self) -> Self::Portgraph<'_> {
        FlatRegionGraph::new_flat_region(
            &self.base_hugr().graph,
            &self.base_hugr().hierarchy,
            self.root.index,
        )
    }

    fn base_hugr(&self) -> &Hugr {
        self.hugr
    }

    fn root_node(&self) -> Node {
        self.root
    }
}

impl<'g, Root: NodeHandle> HugrView for SiblingMut<'g, Root> {
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

    type Neighbours<'a> = MapInto<multiportgraph::Neighbours<'a>, Node>
    where
        Self: 'a;

    type PortLinks<'a> = MapWithCtx<multiportgraph::PortLinks<'a>, &'a Hugr, (Node, Port)>
    where
        Self: 'a;

    type NodeConnections<'a> = MapWithCtx<multiportgraph::NodeConnections<'a>,&'a Hugr, [Port; 2]> where Self: 'a;

    fn contains_node(&self, node: Node) -> bool {
        // Don't call self.get_parent(). That requires valid_node(node)
        // which infinitely-recurses back here.
        node == self.root || self.base_hugr().get_parent(node) == Some(self.root)
    }

    fn node_count(&self) -> usize {
        // Same as SiblingGraph.
        self.base_hugr().hierarchy.child_count(self.root.index) + 1
    }

    fn edge_count(&self) -> usize {
        // Same as SiblingGraph
        self.nodes()
            .map(|n| self.output_neighbours(n).count())
            .sum()
    }

    fn nodes(&self) -> Self::Nodes<'_> {
        // Same as SiblingGraph
        let children = self
            .base_hugr()
            .hierarchy
            .children(self.root.index)
            .map_into();
        iter::once(self.root).chain(children)
    }

    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        self.valid_node(node).unwrap(); // ?? Or return empty iterator?
        self.base_hugr().node_ports(node, dir)
    }

    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        self.valid_node(node).unwrap(); // ?? Or return empty iterator?
        self.base_hugr().all_node_ports(node)
    }

    fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_> {
        self.valid_node(node).unwrap(); // ?? Or return empty iterator?
        self.base_hugr().linked_ports(node, port)
    }

    fn node_connections(&self, node: Node, other: Node) -> Self::NodeConnections<'_> {
        self.valid_node(node).unwrap(); // ?? Or return empty iterator?
        self.valid_node(other).unwrap(); // ?? Is this even a requirement for non-local edges?
        self.base_hugr().node_connections(node, other)
    }

    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.valid_node(node).unwrap(); // ?? Or return 0?
        self.base_hugr().num_ports(node, dir)
    }

    fn children(&self, node: Node) -> Self::Children<'_> {
        match node == self.root {
            true => self.base_hugr().hierarchy.children(node.index).map_into(),
            false => portgraph::hierarchy::Children::default().map_into(),
        }
    }

    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_> {
        self.valid_node(node).unwrap(); // ?? Or return empty iterator?
        self.base_hugr().neighbours(node, dir) // Or self.hugr ?
    }

    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.valid_node(node).unwrap(); // ?? Or return empty iterator?
        self.base_hugr().all_neighbours(node)
    }

    fn get_io(&self, node: Node) -> Option<[Node; 2]> {
        self.valid_node(node).unwrap(); // ?? Or return None?
        self.base_hugr().get_io(node)
    }
}

impl<'g> HugrMutInternals for SiblingMut<'g> {
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.hugr
    }
}

impl<'g> HugrMut for SiblingMut<'g> {}

#[cfg(test)]
mod test {
    use crate::builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use crate::extension::PRELUDE_REGISTRY;
    use crate::ops::handle::{DfgID, FuncID, ModuleRootID};
    use crate::ops::{dataflow::IOTrait, Input, Output};
    use crate::type_row;
    use crate::types::{FunctionType, Type};

    use super::super::descendants::test::make_module_hgr;
    use super::*;

    #[test]
    fn flat_region() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, def, inner) = make_module_hgr()?;

        let region: SiblingGraph = SiblingGraph::try_new(&hugr, def)?;

        assert_eq!(region.node_count(), 5);
        assert!(region
            .nodes()
            .all(|n| n == def || hugr.get_parent(n) == Some(def)));
        assert_eq!(region.children(inner).count(), 0);

        Ok(())
    }

    const NAT: Type = crate::extension::prelude::USIZE_T;
    #[test]
    fn nested_flat() -> Result<(), Box<dyn std::error::Error>> {
        let mut module_builder = ModuleBuilder::new();
        let fty = FunctionType::new(type_row![NAT], type_row![NAT]);
        let mut fbuild = module_builder.define_function("main", fty.clone().pure())?;
        let dfg = fbuild.dfg_builder(fty, None, fbuild.input_wires())?;
        let ins = dfg.input_wires();
        let sub_dfg = dfg.finish_with_outputs(ins)?;
        let fun = fbuild.finish_with_outputs(sub_dfg.outputs())?;
        let h = module_builder.finish_hugr(&PRELUDE_REGISTRY)?;
        let sub_dfg = sub_dfg.node();
        // Can create a view from a child or grandchild of a hugr:
        let dfg_view: SiblingGraph<'_, DfgID> = SiblingGraph::try_new(&h, sub_dfg)?;
        let fun_view: SiblingGraph<'_, FuncID<true>> = SiblingGraph::try_new(&h, fun.node())?;
        assert_eq!(fun_view.children(sub_dfg).len(), 0);
        // And can create a view from a child of another SiblingGraph
        let nested_dfg_view: SiblingGraph<'_, DfgID> = SiblingGraph::try_new(&fun_view, sub_dfg)?;

        // Both ways work:
        let just_io = vec![
            Input::new(type_row![NAT]).into(),
            Output::new(type_row![NAT]).into(),
        ];
        for d in [dfg_view, nested_dfg_view] {
            assert_eq!(
                d.children(sub_dfg).map(|n| d.get_optype(n)).collect_vec(),
                just_io.iter().collect_vec()
            );
        }

        // But cannot create a view directly as a grandchild of another SiblingGraph
        let root_view: SiblingGraph<'_, ModuleRootID> =
            SiblingGraph::try_new(&h, h.root()).unwrap();
        assert_eq!(
            SiblingGraph::<'_, DfgID>::try_new(&root_view, sub_dfg.node()).err(),
            Some(HugrError::InvalidNode(sub_dfg.node()))
        );

        Ok(())
    }
}
