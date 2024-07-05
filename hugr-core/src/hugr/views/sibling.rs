//! SiblingGraph: view onto a sibling subgraph of the HUGR.

use std::iter;

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{LinkView, MultiPortGraph, PortIndex, PortView};

use crate::hugr::internal::HugrMutInternals;
use crate::hugr::{HugrError, HugrMut};
use crate::ops::handle::NodeHandle;
use crate::{Direction, Hugr, Node, Port};

use super::{check_tag, ExtractHugr, HierarchyView, HugrInternals, HugrView, RootTagged};

type FlatRegionGraph<'g> = portgraph::view::FlatRegion<'g, &'g MultiPortGraph>;

/// View of a HUGR sibling graph.
///
/// Includes only the root node and its direct children, but no deeper descendants.
/// However, the descendants can still be accessed by creating [`SiblingGraph`]s and/or
/// [`DescendantsGraph`]s from nodes in this view.
///
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

/// HugrView trait members common to both [SiblingGraph] and [SiblingMut],
/// i.e. that rely only on [HugrInternals::base_hugr]
macro_rules! impl_base_members {
    () => {

        type Nodes<'a> = iter::Chain<iter::Once<Node>, MapInto<portgraph::hierarchy::Children<'a>, Node>>
        where
            Self: 'a;

        type NodePorts<'a> = MapInto<<FlatRegionGraph<'g> as PortView>::NodePortOffsets<'a>, Port>
        where
            Self: 'a;

        type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>
        where
            Self: 'a;

        #[inline]
        fn node_count(&self) -> usize {
            self.base_hugr().hierarchy.child_count(self.root.pg_index()) + 1
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
                .children(self.root.pg_index())
                .map_into();
            iter::once(self.root).chain(children)
        }

        fn children(&self, node: Node) -> Self::Children<'_> {
            // Same as SiblingGraph
            match node == self.root {
                true => self.base_hugr().hierarchy.children(node.pg_index()).map_into(),
                false => portgraph::hierarchy::Children::default().map_into(),
            }
        }
    };
}

impl<'g, Root: NodeHandle> HugrView for SiblingGraph<'g, Root> {
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

    impl_base_members! {}

    #[inline]
    fn contains_node(&self, node: Node) -> bool {
        self.graph.contains_node(node.pg_index())
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        self.graph.port_offsets(node.pg_index(), dir).map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        self.graph.all_port_offsets(node.pg_index()).map_into()
    }

    fn linked_ports(&self, node: Node, port: impl Into<Port>) -> Self::PortLinks<'_> {
        let port = self
            .graph
            .port_index(node.pg_index(), port.into().pg_offset())
            .unwrap();
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
            .get_connections(node.pg_index(), other.pg_index())
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
        self.graph.num_ports(node.pg_index(), dir)
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_> {
        self.graph.neighbours(node.pg_index(), dir).map_into()
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.graph.all_neighbours(node.pg_index()).map_into()
    }
}
impl<'g, Root: NodeHandle> RootTagged for SiblingGraph<'g, Root> {
    type RootHandle = Root;
}

impl<'a, Root: NodeHandle> SiblingGraph<'a, Root> {
    fn new_unchecked(hugr: &'a impl HugrView, root: Node) -> Self {
        let hugr = hugr.base_hugr();
        Self {
            root,
            graph: FlatRegionGraph::new_flat_region(&hugr.graph, &hugr.hierarchy, root.pg_index()),
            hugr,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, Root> HierarchyView<'a> for SiblingGraph<'a, Root>
where
    Root: NodeHandle,
{
    fn try_new(hugr: &'a impl HugrView, root: Node) -> Result<Self, HugrError> {
        assert!(
            hugr.valid_node(root),
            "Cannot create a sibling graph from an invalid node {}.",
            root
        );
        check_tag::<Root>(hugr, root)?;
        Ok(Self::new_unchecked(hugr, root))
    }
}

impl<'g, Root: NodeHandle> ExtractHugr for SiblingGraph<'g, Root> {}

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
/// Like [SiblingGraph], includes only the root node and its direct children, but no
/// deeper descendants; but the descendants can still be accessed by creating nested
/// [SiblingMut] instances from nodes in the view.
///
/// Uniquely, the root node has no parent.
///
/// [HugrView] methods may be slower than for an immutable [SiblingGraph]
/// as the latter may cache information about the graph connectivity,
/// whereas (in order to ease mutation) this does not.
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
    pub fn try_new<Base: HugrMut>(hugr: &'g mut Base, root: Node) -> Result<Self, HugrError> {
        if root == hugr.root() && !Base::RootHandle::TAG.is_superset(Root::TAG) {
            return Err(HugrError::InvalidTag {
                required: Base::RootHandle::TAG,
                actual: Root::TAG,
            });
        }
        check_tag::<Root>(hugr, root)?;
        Ok(Self {
            hugr: hugr.hugr_mut(),
            root,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<'g, Root: NodeHandle> ExtractHugr for SiblingMut<'g, Root> {}

impl<'g, Root: NodeHandle> HugrInternals for SiblingMut<'g, Root> {
    type Portgraph<'p> = FlatRegionGraph<'p> where 'g: 'p, Root: 'p;

    fn portgraph(&self) -> Self::Portgraph<'_> {
        FlatRegionGraph::new_flat_region(
            &self.base_hugr().graph,
            &self.base_hugr().hierarchy,
            self.root.pg_index(),
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
    type Neighbours<'a> = <Vec<Node> as IntoIterator>::IntoIter
    where
        Self: 'a;

    type PortLinks<'a> = <Vec<(Node, Port)> as IntoIterator>::IntoIter
    where
        Self: 'a;

    type NodeConnections<'a> = <Vec<[Port; 2]> as IntoIterator>::IntoIter where Self: 'a;

    impl_base_members! {}

    fn contains_node(&self, node: Node) -> bool {
        // Don't call self.get_parent(). That requires valid_node(node)
        // which infinitely-recurses back here.
        node == self.root || self.base_hugr().get_parent(node) == Some(self.root)
    }

    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        match self.contains_node(node) {
            true => self.base_hugr().node_ports(node, dir),
            false => <FlatRegionGraph as PortView>::NodePortOffsets::default().map_into(),
        }
    }

    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        match self.contains_node(node) {
            true => self.base_hugr().all_node_ports(node),
            false => <FlatRegionGraph as PortView>::NodePortOffsets::default().map_into(),
        }
    }

    fn linked_ports(&self, node: Node, port: impl Into<Port>) -> Self::PortLinks<'_> {
        // Need to filter only to links inside the sibling graph
        SiblingGraph::<'_, Node>::new_unchecked(self.hugr, self.root)
            .linked_ports(node, port)
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn node_connections(&self, node: Node, other: Node) -> Self::NodeConnections<'_> {
        // Need to filter only to connections inside the sibling graph
        SiblingGraph::<'_, Node>::new_unchecked(self.hugr, self.root)
            .node_connections(node, other)
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        match self.contains_node(node) {
            true => self.base_hugr().num_ports(node, dir),
            false => 0,
        }
    }

    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_> {
        // Need to filter to neighbours in the Sibling Graph
        SiblingGraph::<'_, Node>::new_unchecked(self.hugr, self.root)
            .neighbours(node, dir)
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        SiblingGraph::<'_, Node>::new_unchecked(self.hugr, self.root)
            .all_neighbours(node)
            .collect::<Vec<_>>()
            .into_iter()
    }
}
impl<'g, Root: NodeHandle> RootTagged for SiblingMut<'g, Root> {
    type RootHandle = Root;
}
impl<'g, Root: NodeHandle> HugrMutInternals for SiblingMut<'g, Root> {
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.hugr
    }
}

impl<'g, Root: NodeHandle> HugrMut for SiblingMut<'g, Root> {}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::builder::test::simple_dfg_hugr;
    use crate::builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use crate::extension::PRELUDE_REGISTRY;
    use crate::ops::handle::{CfgID, DataflowParentID, DfgID, FuncID};
    use crate::ops::{dataflow::IOTrait, Input, OpTag, Output};
    use crate::ops::{OpTrait, OpType};
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
        let mut fbuild = module_builder.define_function("main", fty.clone())?;
        let dfg = fbuild.dfg_builder(fty, fbuild.input_wires())?;
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

        Ok(())
    }

    #[rstest]
    fn flat_mut(mut simple_dfg_hugr: Hugr) {
        simple_dfg_hugr.update_validate(&PRELUDE_REGISTRY).unwrap();
        let root = simple_dfg_hugr.root();
        let signature = simple_dfg_hugr.inner_function_type().unwrap().clone();

        let sib_mut = SiblingMut::<CfgID>::try_new(&mut simple_dfg_hugr, root);
        assert_eq!(
            sib_mut.err(),
            Some(HugrError::InvalidTag {
                required: OpTag::Cfg,
                actual: OpTag::Dfg
            })
        );

        let mut sib_mut = SiblingMut::<DfgID>::try_new(&mut simple_dfg_hugr, root).unwrap();
        let bad_nodetype: OpType = crate::ops::CFG { signature }.into();
        assert_eq!(
            sib_mut.replace_op(sib_mut.root(), bad_nodetype.clone()),
            Err(HugrError::InvalidTag {
                required: OpTag::Dfg,
                actual: OpTag::Cfg
            })
        );

        // In contrast, performing this on the Hugr (where the allowed root type is 'Any') is only detected by validation
        simple_dfg_hugr.replace_op(root, bad_nodetype).unwrap();
        assert!(simple_dfg_hugr.validate(&PRELUDE_REGISTRY).is_err());
    }

    #[rstest]
    fn sibling_mut_covariance(mut simple_dfg_hugr: Hugr) {
        let root = simple_dfg_hugr.root();
        let case_nodetype = crate::ops::Case {
            signature: simple_dfg_hugr.root_type().dataflow_signature().unwrap(),
        };
        let mut sib_mut = SiblingMut::<DfgID>::try_new(&mut simple_dfg_hugr, root).unwrap();
        // As expected, we cannot replace the root with a Case
        assert_eq!(
            sib_mut.replace_op(root, case_nodetype),
            Err(HugrError::InvalidTag {
                required: OpTag::Dfg,
                actual: OpTag::Case
            })
        );

        let nested_sib_mut = SiblingMut::<DataflowParentID>::try_new(&mut sib_mut, root);
        assert!(nested_sib_mut.is_err());
    }

    #[rstest]
    fn extract_hugr() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, _def, inner) = make_module_hgr()?;

        let region: SiblingGraph = SiblingGraph::try_new(&hugr, inner)?;
        let extracted = region.extract_hugr();
        extracted.validate(&PRELUDE_REGISTRY)?;

        let region: SiblingGraph = SiblingGraph::try_new(&hugr, inner)?;

        assert_eq!(region.node_count(), extracted.node_count());
        assert_eq!(region.root_type(), extracted.root_type());

        Ok(())
    }
}
