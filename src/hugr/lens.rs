//! Mutable lenses to hierarchical regions of a HUGR.
//!
//! Lenses are mutable references to a HUGR that allows arbitrary rewrites, but
//! have a restricted view interface rooted in a node.
//!
//! For read-only views of the hierarchical subgraphs, see [`::hugr::views`].

use std::sync::{OnceLock, RwLock};

use delegate::delegate;

use crate::ops::handle::NodeHandle;
use crate::ops::OpType;
use crate::{Direction, Hugr, HugrView, Node, Port};

use super::hugrmut::sealed::HugrMutInternals;
use super::views::sealed::HugrInternals;
use super::views::{HierarchyView, SiblingGraph};
use super::{NodeMetadata, NodeType, HugrMut};

/// Mutable lens to hierarchical regions of a HUGR.
///
/// Lenses are mutable references to a HUGR that allows arbitrary rewrites, but
/// have a restricted view interface rooted in a node.
pub struct HugrLens<'g, NodeHandle = Node> {
    /// The chosen root node.
    root: Node,

    /// The borrowed HUGR.
    hugr: &'g mut Hugr,

    /// The operation type of the root node.
    _phantom: std::marker::PhantomData<NodeHandle>,
}

impl<'g, Root> HugrLens<'g, Root>
where
    Root: NodeHandle,
{
    /// Creates a new mutable reference to a HUGR sibling graph.
    pub fn new(hugr: &'g mut Hugr, root: Node) -> Self {
        Self {
            root,
            hugr,
            view: Default::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a new mutable reference to a new node.
    #[inline]
    pub fn new_lens<NewRoot: NodeHandle>(&mut self, node: Node) -> HugrLens<'_, NewRoot> {
        HugrLens::new(self.hugr, node)
    }

    fn hugr(&self) -> &'g Hugr {
        self.hugr
    }

    /// TODO
    #[inline]
    fn new_view(&self) -> SiblingGraph<'g, Root> {
        SiblingGraph::new(self.hugr(), self.root)
    }

    /// Read-only view of the sibling graph rooted at the current node.
    #[inline]
    fn view(&self) -> &SiblingGraph<'g, Root> {
        self.view.read().unwrap().get_or_init(|| self.new_view())
    }
}

impl<'g, Root> HugrMut for HugrLens<'g, Root>
where
    Root: NodeHandle,
{
    
}

impl<'g, Root> HugrMutInternals for HugrLens<'g, Root>
where
    Root: NodeHandle,
{
    #[inline]
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.hugr
    }
}

impl<'g, Root> HugrInternals for HugrLens<'g, Root>
where
    Root: NodeHandle,
{
    type Portgraph<'p> = <SiblingGraph<'p, Root> as HugrInternals>::Portgraph<'p> where Self: 'p;

    #[inline(always)]
    fn portgraph(&self) -> Self::Portgraph<'_> {
        self.view().portgraph()
    }
    #[inline(always)]
    fn base_hugr(&self) -> &Hugr {
        self.view().base_hugr()
    }
    #[inline(always)]
    fn root_node(&self) -> Node {
        self.view().root_node()
    }
}

impl<'g, Root> HugrView for HugrLens<'g, Root>
where
    Root: NodeHandle,
{
    type RootHandle = Root;

    type Nodes<'a> = <SiblingGraph<'g, Root> as HugrView>::Nodes<'a> where Self: 'a;
    type NodePorts<'a> = <SiblingGraph<'g, Root> as HugrView>::NodePorts<'a> where Self: 'a;
    type Children<'a> = <SiblingGraph<'g, Root> as HugrView>::Children<'a> where Self: 'a;
    type Neighbours<'a> = <SiblingGraph<'g, Root> as HugrView>::Neighbours<'a> where Self: 'a;
    type PortLinks<'a> = <SiblingGraph<'g, Root> as HugrView>::PortLinks<'a> where Self: 'a;

    delegate! {
        to self.view() {
            fn get_parent(&self, node: Node) -> Option<Node>;
            fn get_optype(&self, node: Node) -> &OpType;
            fn get_nodetype(&self, node: Node) -> &NodeType;
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
            fn get_io(&self, node: Node) -> Option<[Node; 2]>;
        }
    }
}
