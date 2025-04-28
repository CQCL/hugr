//! SiblingGraph: view onto a sibling subgraph of the HUGR.

use std::iter;

use itertools::{Either, Itertools};
use portgraph::{LinkView, MultiPortGraph, PortView};

use crate::hugr::internal::HugrMutInternals;
use crate::hugr::{HugrError, HugrMut, NodeMetadataMap};
use crate::ops::handle::NodeHandle;
use crate::ops::OpTrait;
use crate::{Direction, Hugr, Node, Port};

use super::{check_tag, ExtractHugr, HierarchyView, HugrInternals, HugrView};

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
    // TODO: this can only be made generic once the call to base_hugr is removed
    // in try_new. See https://github.com/CQCL/hugr/issues/1926
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
        #[inline]
        fn root(&self) -> Self::Node {
            self.root
        }

        #[inline]
        fn node_count(&self) -> usize {
            self.hierarchy()
                .child_count(self.to_portgraph_node(self.root))
                + 1
        }

        #[inline]
        fn edge_count(&self) -> usize {
            // Faster implementation than filtering all the nodes in the internal graph.
            self.nodes()
                .map(|n| self.output_neighbours(n).count())
                .sum()
        }

        #[inline]
        fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone {
            // Faster implementation than filtering all the nodes in the internal graph.
            let children = self
                .hierarchy()
                .children(self.to_portgraph_node(self.root))
                .map(|n| self.from_portgraph_node(n));
            iter::once(self.root).chain(children)
        }

        fn children(
            &self,
            node: Self::Node,
        ) -> impl DoubleEndedIterator<Item = Self::Node> + Clone {
            // Same as SiblingGraph
            let children = match node == self.root {
                true => self.hierarchy().children(self.to_portgraph_node(node)),
                false => portgraph::hierarchy::Children::default(),
            };
            children.map(|n| self.from_portgraph_node(n))
        }

        fn get_optype(&self, node: Self::Node) -> &crate::ops::OpType {
            self.hugr.get_optype(node)
        }

        fn extensions(&self) -> &crate::extension::ExtensionRegistry {
            self.hugr.extensions()
        }
    };
}

impl<Root: NodeHandle> HugrView for SiblingGraph<'_, Root> {
    impl_base_members! {}

    #[inline]
    fn contains_node(&self, node: Node) -> bool {
        self.graph.contains_node(self.to_portgraph_node(node))
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> impl Iterator<Item = Port> + Clone {
        self.graph
            .port_offsets(self.to_portgraph_node(node), dir)
            .map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> impl Iterator<Item = Port> + Clone {
        self.graph
            .all_port_offsets(self.to_portgraph_node(node))
            .map_into()
    }

    fn linked_ports(
        &self,
        node: Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Node, Port)> + Clone {
        let port = self
            .graph
            .port_index(self.to_portgraph_node(node), port.into().pg_offset())
            .unwrap();
        self.graph.port_links(port).map(|(_, link)| {
            let node = self.graph.port_node(link).unwrap();
            let offset = self.graph.port_offset(link).unwrap();
            (self.from_portgraph_node(node), offset.into())
        })
    }

    fn node_connections(&self, node: Node, other: Node) -> impl Iterator<Item = [Port; 2]> + Clone {
        self.graph
            .get_connections(self.to_portgraph_node(node), self.to_portgraph_node(other))
            .map(|(p1, p2)| [p1, p2].map(|link| self.graph.port_offset(link).unwrap().into()))
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(self.to_portgraph_node(node), dir)
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> impl Iterator<Item = Node> + Clone {
        self.graph
            .neighbours(self.to_portgraph_node(node), dir)
            .map(|n| self.from_portgraph_node(n))
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone {
        self.graph
            .all_neighbours(self.to_portgraph_node(node))
            .map(|n| self.from_portgraph_node(n))
    }
}

impl<'a, Root: NodeHandle> SiblingGraph<'a, Root> {
    fn new_unchecked(hugr: &'a impl HugrView<Node = Node>, root: Node) -> Self {
        #[allow(deprecated)]
        let hugr = hugr.base_hugr();
        Self {
            root,
            graph: FlatRegionGraph::new_with_root(
                &hugr.graph,
                &hugr.hierarchy,
                hugr.to_portgraph_node(root),
            ),
            hugr,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, Root> HierarchyView<'a> for SiblingGraph<'a, Root>
where
    Root: NodeHandle,
{
    fn try_new(hugr: &'a impl HugrView<Node = Node>, root: Node) -> Result<Self, HugrError> {
        assert!(
            hugr.contains_node(root),
            "Cannot create a sibling graph from an invalid node {}.",
            root
        );
        check_tag::<Root, _>(hugr, root)?;
        Ok(Self::new_unchecked(hugr, root))
    }
}

impl<Root: NodeHandle> ExtractHugr for SiblingGraph<'_, Root> {}

impl<'g, Root: NodeHandle> HugrInternals for SiblingGraph<'g, Root>
where
    Root: NodeHandle,
{
    type Portgraph<'p>
        = &'p FlatRegionGraph<'g>
    where
        Self: 'p;
    type Node = Node;

    #[inline]
    fn portgraph(&self) -> Self::Portgraph<'_> {
        &self.graph
    }

    #[inline]
    fn hierarchy(&self) -> &portgraph::Hierarchy {
        self.hugr.hierarchy()
    }

    #[inline]
    fn base_hugr(&self) -> &Hugr {
        self.hugr
    }

    #[inline]
    fn to_portgraph_node(&self, node: impl NodeHandle<Self::Node>) -> portgraph::NodeIndex {
        self.hugr.to_portgraph_node(node)
    }

    #[inline]
    fn from_portgraph_node(&self, index: portgraph::NodeIndex) -> Node {
        self.hugr.from_portgraph_node(index)
    }

    #[inline]
    fn node_metadata_map(&self, node: Self::Node) -> &NodeMetadataMap {
        self.hugr.node_metadata_map(node)
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
pub struct SiblingMut<'g, H: HugrView, Root = Node> {
    /// The chosen root node.
    root: H::Node,

    /// The rest of the HUGR.
    hugr: &'g mut H,

    /// The operation type of the root node.
    _phantom: std::marker::PhantomData<Root>,
}

impl<'g, H: HugrMut, Root: NodeHandle<H::Node>> SiblingMut<'g, H, Root> {
    /// Create a new SiblingMut from a base.
    /// Equivalent to [HierarchyView::try_new] but takes a *mutable* reference.
    pub fn try_new(hugr: &'g mut H, root: H::Node) -> Result<Self, HugrError> {
        check_tag::<Root, _>(hugr, root)?;
        Ok(Self {
            hugr,
            root,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<H: HugrMut, Root: NodeHandle<H::Node>> ExtractHugr for SiblingMut<'_, H, Root> {}

impl<'g, H: HugrMut, Root: NodeHandle<H::Node>> HugrInternals for SiblingMut<'g, H, Root> {
    type Portgraph<'p>
        = FlatRegionGraph<'p>
    where
        'g: 'p,
        Root: 'p;
    type Node = H::Node;

    #[inline]
    fn portgraph(&self) -> Self::Portgraph<'_> {
        FlatRegionGraph::new_with_root(
            #[allow(deprecated)]
            &self.base_hugr().graph,
            self.hierarchy(),
            self.to_portgraph_node(self.root),
        )
    }

    #[inline]
    fn hierarchy(&self) -> &portgraph::Hierarchy {
        self.hugr.hierarchy()
    }

    #[inline]
    fn to_portgraph_node(&self, node: impl NodeHandle<Self::Node>) -> portgraph::NodeIndex {
        self.hugr.to_portgraph_node(node)
    }

    #[inline]
    fn from_portgraph_node(&self, index: portgraph::NodeIndex) -> Self::Node {
        self.hugr.from_portgraph_node(index)
    }

    #[inline]
    fn node_metadata_map(&self, node: Self::Node) -> &NodeMetadataMap {
        self.hugr.node_metadata_map(node)
    }

    #[inline]
    fn base_hugr(&self) -> &Hugr {
        #[allow(deprecated)]
        self.hugr.base_hugr()
    }
}

impl<H: HugrMut, Root: NodeHandle<H::Node>> HugrView for SiblingMut<'_, H, Root> {
    impl_base_members! {}

    fn contains_node(&self, node: H::Node) -> bool {
        // Don't call self.get_parent(). That requires valid_node(node)
        // which infinitely-recurses back here.
        node == self.root || self.hugr.get_parent(node) == Some(self.root)
    }

    fn node_ports(&self, node: Self::Node, dir: Direction) -> impl Iterator<Item = Port> + Clone {
        self.hugr.node_ports(node, dir)
    }

    fn all_node_ports(&self, node: Self::Node) -> impl Iterator<Item = Port> + Clone {
        self.hugr.all_node_ports(node)
    }

    fn linked_ports(
        &self,
        node: Self::Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Self::Node, Port)> + Clone {
        self.hugr
            .linked_ports(node, port)
            .filter(|(n, _)| self.contains_node(*n))
    }

    fn node_connections(
        &self,
        node: Self::Node,
        other: Self::Node,
    ) -> impl Iterator<Item = [Port; 2]> + Clone {
        match self.contains_node(node) && self.contains_node(other) {
            // The nodes are not in the sibling graph
            false => Either::Left(iter::empty()),
            // The nodes are in the sibling graph
            true => Either::Right(self.hugr.node_connections(node, other)),
        }
    }

    fn num_ports(&self, node: Self::Node, dir: Direction) -> usize {
        self.hugr.num_ports(node, dir)
    }

    fn neighbours(
        &self,
        node: Self::Node,
        dir: Direction,
    ) -> impl Iterator<Item = Self::Node> + Clone {
        self.hugr
            .neighbours(node, dir)
            .filter(|n| self.contains_node(*n))
    }

    fn all_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        self.hugr
            .all_neighbours(node)
            .filter(|n| self.contains_node(*n))
    }
}

impl<H: HugrMut, Root: NodeHandle<H::Node>> HugrMutInternals for SiblingMut<'_, H, Root> {
    fn replace_op(
        &mut self,
        node: Self::Node,
        op: impl Into<crate::ops::OpType>,
    ) -> crate::ops::OpType {
        let op = op.into();
        // Note: `SiblingMut` will be removed in a subsequent PR, so we just panic here for now.
        if node == self.root() && !Root::TAG.is_superset(op.tag()) {
            let err = HugrError::InvalidTag {
                required: Root::TAG,
                actual: op.tag(),
            };
            panic!("{err}");
        }
        self.hugr.replace_op(node, op)
    }

    delegate::delegate! {
        to (&mut *self.hugr) {
            fn set_root(&mut self, root: Self::Node);
            fn set_num_ports(&mut self, node: Self::Node, incoming: usize, outgoing: usize);
            fn add_ports(&mut self, node: Self::Node, direction: crate::Direction, amount: isize) -> std::ops::Range<usize>;
            fn insert_ports(&mut self, node: Self::Node, direction: crate::Direction, index: usize, amount: usize) -> std::ops::Range<usize>;
            fn set_parent(&mut self, node: Self::Node, parent: Self::Node);
            fn move_after_sibling(&mut self, node: Self::Node, after: Self::Node);
            fn move_before_sibling(&mut self, node: Self::Node, before: Self::Node);
            fn optype_mut(&mut self, node: Self::Node) -> &mut crate::ops::OpType;
            fn node_metadata_map_mut(&mut self, node: Self::Node) -> &mut crate::hugr::NodeMetadataMap;
            fn extensions_mut(&mut self) -> &mut crate::extension::ExtensionRegistry;
        }
    }
}

impl<H: HugrMut, Root: NodeHandle<H::Node>> HugrMut for SiblingMut<'_, H, Root> {
    super::impls::hugr_mut_methods! {this, &mut *this.hugr}
}

#[cfg(test)]
mod test {
    use std::borrow::Cow;

    use rstest::rstest;

    use crate::builder::test::simple_dfg_hugr;
    use crate::builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use crate::extension::prelude::{qb_t, usize_t};
    use crate::ops::handle::{CfgID, DfgID, FuncID};
    use crate::ops::OpType;
    use crate::ops::{dataflow::IOTrait, Input, OpTag, Output};
    use crate::types::Signature;
    use crate::utils::test_quantum_extension::EXTENSION_ID;
    use crate::IncomingPort;

    use super::super::descendants::test::make_module_hgr;
    use super::*;

    fn test_properties<T>(
        hugr: &Hugr,
        def: Node,
        inner: Node,
        region: T,
        inner_region: T,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        T: HugrView<Node = Node> + Sized,
    {
        let def_io = region.get_io(def).unwrap();

        assert_eq!(region.node_count(), 5);
        assert_eq!(region.portgraph().node_count(), 5);
        assert!(region.nodes().all(|n| n == def
            || hugr.get_parent(n) == Some(def)
            || hugr.get_parent(n) == Some(inner)));
        assert_eq!(region.children(inner).count(), 0);

        assert_eq!(
            region.poly_func_type(),
            Some(
                Signature::new_endo(vec![usize_t(), qb_t()])
                    .with_extension_delta(EXTENSION_ID)
                    .into()
            )
        );

        assert_eq!(
            inner_region.inner_function_type().map(Cow::into_owned),
            Some(Signature::new(vec![usize_t()], vec![usize_t()]))
        );
        assert_eq!(inner_region.node_count(), 3);
        assert_eq!(inner_region.edge_count(), 1);
        assert_eq!(inner_region.children(inner).count(), 2);
        assert_eq!(inner_region.children(hugr.root()).count(), 0);
        assert_eq!(
            inner_region.num_ports(inner, Direction::Outgoing),
            inner_region.node_ports(inner, Direction::Outgoing).count()
        );
        assert_eq!(
            inner_region.num_ports(inner, Direction::Incoming)
                + inner_region.num_ports(inner, Direction::Outgoing),
            inner_region.all_node_ports(inner).count()
        );

        // The inner region filters out the connections to the main function I/O nodes,
        // while the outer region includes them.
        assert_eq!(inner_region.node_connections(inner, def_io[1]).count(), 0);
        assert_eq!(region.node_connections(inner, def_io[1]).count(), 1);
        assert_eq!(
            inner_region
                .linked_ports(inner, IncomingPort::from(0))
                .count(),
            0
        );
        assert_eq!(region.linked_ports(inner, IncomingPort::from(0)).count(), 1);
        assert_eq!(
            inner_region.neighbours(inner, Direction::Outgoing).count(),
            0
        );
        assert_eq!(inner_region.all_neighbours(inner).count(), 0);
        assert_eq!(
            inner_region
                .linked_ports(inner, IncomingPort::from(0))
                .count(),
            0
        );

        Ok(())
    }

    #[rstest]
    fn sibling_graph_properties() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, def, inner) = make_module_hgr()?;

        test_properties::<SiblingGraph>(
            &hugr,
            def,
            inner,
            SiblingGraph::try_new(&hugr, def).unwrap(),
            SiblingGraph::try_new(&hugr, inner).unwrap(),
        )
    }

    #[rstest]
    fn sibling_mut_properties() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, def, inner) = make_module_hgr()?;
        let mut def_region_hugr = hugr.clone();
        let mut inner_region_hugr = hugr.clone();

        test_properties::<SiblingMut<_>>(
            &hugr,
            def,
            inner,
            SiblingMut::try_new(&mut def_region_hugr, def).unwrap(),
            SiblingMut::try_new(&mut inner_region_hugr, inner).unwrap(),
        )
    }

    #[test]
    fn nested_flat() -> Result<(), Box<dyn std::error::Error>> {
        let mut module_builder = ModuleBuilder::new();
        let fty = Signature::new(vec![usize_t()], vec![usize_t()]);
        let mut fbuild = module_builder.define_function("main", fty.clone())?;
        let dfg = fbuild.dfg_builder(fty, fbuild.input_wires())?;
        let ins = dfg.input_wires();
        let sub_dfg = dfg.finish_with_outputs(ins)?;
        let fun = fbuild.finish_with_outputs(sub_dfg.outputs())?;
        let h = module_builder.finish_hugr()?;
        let sub_dfg = sub_dfg.node();

        // We can create a view from a child or grandchild of a hugr:
        let dfg_view: SiblingGraph<'_, DfgID> = SiblingGraph::try_new(&h, sub_dfg)?;
        let fun_view: SiblingGraph<'_, FuncID<true>> = SiblingGraph::try_new(&h, fun.node())?;
        assert_eq!(fun_view.children(sub_dfg).count(), 0);

        // And also create a view from a child of another SiblingGraph
        let nested_dfg_view: SiblingGraph<'_, DfgID> = SiblingGraph::try_new(&fun_view, sub_dfg)?;

        // Both ways work:
        let just_io = vec![
            Input::new(vec![usize_t()]).into(),
            Output::new(vec![usize_t()]).into(),
        ];
        for d in [dfg_view, nested_dfg_view] {
            assert_eq!(
                d.children(sub_dfg).map(|n| d.get_optype(n)).collect_vec(),
                just_io.iter().collect_vec()
            );
        }

        Ok(())
    }

    /// Mutate a SiblingMut wrapper
    #[rstest]
    fn flat_mut(mut simple_dfg_hugr: Hugr) {
        simple_dfg_hugr.validate().unwrap();
        let root = simple_dfg_hugr.root();
        let signature = simple_dfg_hugr.inner_function_type().unwrap().into_owned();

        let sib_mut = SiblingMut::<_, CfgID>::try_new(&mut simple_dfg_hugr, root);
        assert_eq!(
            sib_mut.err(),
            Some(HugrError::InvalidTag {
                required: OpTag::Cfg,
                actual: OpTag::Dfg
            })
        );

        let bad_nodetype: OpType = crate::ops::CFG { signature }.into();

        // Performing this on the Hugr (where the allowed root type is 'Any') is only detected by validation
        simple_dfg_hugr.replace_op(root, bad_nodetype);
        assert!(simple_dfg_hugr.validate().is_err());
    }

    #[rstest]
    fn extract_hugr() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, _def, inner) = make_module_hgr()?;

        let region: SiblingGraph = SiblingGraph::try_new(&hugr, inner)?;
        let extracted = region.extract_hugr();
        extracted.validate()?;

        let region: SiblingGraph = SiblingGraph::try_new(&hugr, inner)?;

        assert_eq!(region.node_count(), extracted.node_count());
        assert_eq!(region.root_optype(), extracted.root_optype());

        Ok(())
    }
}
