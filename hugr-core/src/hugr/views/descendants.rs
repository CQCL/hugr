//! DescendantsGraph: view onto the subgraph of the HUGR starting from a root
//! (all descendants at all depths).

use itertools::Itertools;
use portgraph::{LinkView, MultiPortGraph, PortIndex, PortView};

use crate::hugr::HugrError;
use crate::ops::handle::NodeHandle;
use crate::{Direction, Hugr, Node, Port};

use super::{check_tag, ExtractHugr, HierarchyView, HugrInternals, HugrView};

type RegionGraph<'g> = portgraph::view::Region<'g, &'g MultiPortGraph>;

/// View of a HUGR descendants graph.
///
/// Includes the root node (which uniquely has no parent) and all its descendants.
///
/// See [`SiblingGraph`] for a view that includes only the root and
/// its immediate children.  Prefer using [`SiblingGraph`] when possible,
/// as it is more efficient.
///
/// Implements the [`HierarchyView`] trait, as well as [`HugrView`], it can be
/// used interchangeably with [`SiblingGraph`].
///
/// [`SiblingGraph`]: super::SiblingGraph
#[derive(Clone)]
pub struct DescendantsGraph<'g, Root = Node> {
    /// The chosen root node.
    // TODO: this can only be made generic once the call to base_hugr is removed
    // in try_new. See https://github.com/CQCL/hugr/issues/1926
    root: Node,

    /// The graph encoding the adjacency structure of the HUGR.
    graph: RegionGraph<'g>,

    /// The node hierarchy.
    hugr: &'g Hugr,

    /// The operation handle of the root node.
    _phantom: std::marker::PhantomData<Root>,
}
impl<Root: NodeHandle> HugrView for DescendantsGraph<'_, Root> {
    #[inline]
    fn root(&self) -> Self::Node {
        self.root
    }

    #[inline]
    fn contains_node(&self, node: Node) -> bool {
        self.graph.contains_node(self.to_portgraph_node(node))
    }

    #[inline]
    fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    #[inline]
    fn num_edges(&self) -> usize {
        self.graph.link_count()
    }

    #[inline]
    fn nodes(&self) -> impl Iterator<Item = Node> + Clone {
        self.graph
            .nodes_iter()
            .map(|index| self.from_portgraph_node(index))
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
            let port: PortIndex = link.into();
            let node = self.graph.port_node(port).unwrap();
            let offset = self.graph.port_offset(port).unwrap();
            (self.from_portgraph_node(node), offset.into())
        })
    }

    fn node_connections(&self, node: Node, other: Node) -> impl Iterator<Item = [Port; 2]> + Clone {
        self.graph
            .get_connections(self.to_portgraph_node(node), self.to_portgraph_node(other))
            .map(|(p1, p2)| {
                [p1, p2].map(|link| {
                    let offset = self.graph.port_offset(link).unwrap();
                    offset.into()
                })
            })
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(self.to_portgraph_node(node), dir)
    }

    #[inline]
    fn children(&self, node: Node) -> impl DoubleEndedIterator<Item = Node> + Clone {
        let hierarchy = self.hierarchy();
        let children = match self.graph.contains_node(self.to_portgraph_node(node)) {
            true => hierarchy.children(self.to_portgraph_node(node)),
            false => portgraph::hierarchy::Children::default(),
        };
        children.map(move |index| {
            let _ = hierarchy;
            self.from_portgraph_node(index)
        })
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> impl Iterator<Item = Node> + Clone {
        self.graph
            .neighbours(self.to_portgraph_node(node), dir)
            .map(|index| self.from_portgraph_node(index))
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone {
        self.graph
            .all_neighbours(self.to_portgraph_node(node))
            .map(|index| self.from_portgraph_node(index))
    }

    delegate::delegate! {
        to (&self.hugr) {
            fn get_parent(&self, node: Self::Node) -> Option<Self::Node>;
            fn get_optype(&self, node: Self::Node) -> &crate::ops::OpType;
            fn descendants(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;
            fn extensions(&self) -> &crate::extension::ExtensionRegistry;
        }
    }
}

impl<'a, Root> HierarchyView<'a> for DescendantsGraph<'a, Root>
where
    Root: NodeHandle,
{
    fn try_new(hugr: &'a impl HugrView<Node = Node>, root: Node) -> Result<Self, HugrError> {
        check_tag::<Root, _>(hugr, root)?;
        #[allow(deprecated)]
        let hugr = hugr.base_hugr();
        Ok(Self {
            root,
            graph: RegionGraph::new(&hugr.graph, &hugr.hierarchy, hugr.to_portgraph_node(root)),
            hugr,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<Root: NodeHandle> ExtractHugr for DescendantsGraph<'_, Root> {}

impl<'g, Root> super::HugrInternals for DescendantsGraph<'g, Root>
where
    Root: NodeHandle,
{
    type Portgraph<'p>
        = &'p RegionGraph<'g>
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
    fn to_portgraph_node(&self, node: impl NodeHandle<Self::Node>) -> portgraph::NodeIndex {
        self.hugr.to_portgraph_node(node)
    }

    #[inline]
    fn from_portgraph_node(&self, index: portgraph::NodeIndex) -> Node {
        self.hugr.from_portgraph_node(index)
    }

    fn node_metadata_map(&self, node: Self::Node) -> &crate::hugr::NodeMetadataMap {
        self.hugr.node_metadata_map(node)
    }

    fn base_hugr(&self) -> &Hugr {
        self.hugr
    }
}

#[cfg(test)]
pub(super) mod test {
    use std::borrow::Cow;

    use rstest::rstest;

    use crate::extension::prelude::{qb_t, usize_t};
    use crate::IncomingPort;
    use crate::{
        builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        types::Signature,
        utils::test_quantum_extension::{h_gate, EXTENSION_ID},
    };

    use super::*;

    /// Make a module hugr with a fn definition containing an inner dfg node.
    ///
    /// Returns the hugr, the fn node id, and the nested dgf node id.
    pub(in crate::hugr::views) fn make_module_hgr(
    ) -> Result<(Hugr, Node, Node), Box<dyn std::error::Error>> {
        let mut module_builder = ModuleBuilder::new();

        let (f_id, inner_id) = {
            let mut func_builder = module_builder.define_function(
                "main",
                Signature::new_endo(vec![usize_t(), qb_t()]).with_extension_delta(EXTENSION_ID),
            )?;

            let [int, qb] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb])?;

            let inner_id = {
                let inner_builder = func_builder
                    .dfg_builder(Signature::new(vec![usize_t()], vec![usize_t()]), [int])?;
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
    fn full_region() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, def, inner) = make_module_hgr()?;

        let region: DescendantsGraph = DescendantsGraph::try_new(&hugr, def)?;
        let def_io = region.get_io(def).unwrap();

        assert_eq!(region.num_nodes(), 7);
        assert!(region.nodes().all(|n| n == def
            || hugr.get_parent(n) == Some(def)
            || hugr.get_parent(n) == Some(inner)));
        assert_eq!(region.children(inner).count(), 2);

        assert_eq!(
            region.poly_func_type(),
            Some(
                Signature::new_endo(vec![usize_t(), qb_t()])
                    .with_extension_delta(EXTENSION_ID)
                    .into()
            )
        );

        let inner_region: DescendantsGraph = DescendantsGraph::try_new(&hugr, inner)?;
        assert_eq!(
            inner_region.inner_function_type().map(Cow::into_owned),
            Some(Signature::new(vec![usize_t()], vec![usize_t()]))
        );
        assert_eq!(inner_region.num_nodes(), 3);
        assert_eq!(inner_region.num_edges(), 1);
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
    fn extract_hugr() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, def, _inner) = make_module_hgr()?;

        let region: DescendantsGraph = DescendantsGraph::try_new(&hugr, def)?;
        let extracted = region.extract_hugr();
        extracted.validate()?;

        let region: DescendantsGraph = DescendantsGraph::try_new(&hugr, def)?;

        assert_eq!(region.num_nodes(), extracted.num_nodes());
        assert_eq!(region.root_optype(), extracted.root_optype());

        Ok(())
    }
}
