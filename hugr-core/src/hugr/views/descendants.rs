//! DescendantsGraph: view onto the subgraph of the HUGR starting from a root
//! (all descendants at all depths).

use itertools::Itertools;
use portgraph::{LinkView, MultiPortGraph, PortIndex, PortView};

use crate::hugr::HugrError;
use crate::ops::handle::NodeHandle;
use crate::{Direction, Hugr, Node, Port};

use super::{check_tag, ExtractHugr, HierarchyView, HugrInternals, HugrView, RootTagged};

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
    fn contains_node(&self, node: Node) -> bool {
        self.graph.contains_node(node.pg_index())
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
    fn nodes(&self) -> impl Iterator<Item = Node> + Clone {
        self.graph.nodes_iter().map_into()
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> impl Iterator<Item = Port> + Clone {
        self.graph.port_offsets(node.pg_index(), dir).map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> impl Iterator<Item = Port> + Clone {
        self.graph.all_port_offsets(node.pg_index()).map_into()
    }

    fn linked_ports(
        &self,
        node: Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Node, Port)> + Clone {
        let port = self
            .graph
            .port_index(node.pg_index(), port.into().pg_offset())
            .unwrap();
        self.graph.port_links(port).map(|(_, link)| {
            let port: PortIndex = link.into();
            let node = self.graph.port_node(port).unwrap();
            let offset = self.graph.port_offset(port).unwrap();
            (node.into(), offset.into())
        })
    }

    fn node_connections(&self, node: Node, other: Node) -> impl Iterator<Item = [Port; 2]> + Clone {
        self.graph
            .get_connections(node.pg_index(), other.pg_index())
            .map(|(p1, p2)| {
                [p1, p2].map(|link| {
                    let offset = self.graph.port_offset(link).unwrap();
                    offset.into()
                })
            })
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(node.pg_index(), dir)
    }

    #[inline]
    fn children(&self, node: Node) -> impl DoubleEndedIterator<Item = Node> + Clone {
        match self.graph.contains_node(node.pg_index()) {
            true => self
                .base_hugr()
                .hierarchy
                .children(node.pg_index())
                .map_into(),
            false => portgraph::hierarchy::Children::default().map_into(),
        }
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> impl Iterator<Item = Node> + Clone {
        self.graph.neighbours(node.pg_index(), dir).map_into()
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone {
        self.graph.all_neighbours(node.pg_index()).map_into()
    }
}
impl<Root: NodeHandle> RootTagged for DescendantsGraph<'_, Root> {
    type RootHandle = Root;
}

impl<'a, Root> HierarchyView<'a> for DescendantsGraph<'a, Root>
where
    Root: NodeHandle,
{
    fn try_new(hugr: &'a impl HugrView, root: Node) -> Result<Self, HugrError> {
        check_tag::<Root>(hugr, root)?;
        let hugr = hugr.base_hugr();
        Ok(Self {
            root,
            graph: RegionGraph::new_region(&hugr.graph, &hugr.hierarchy, root.pg_index()),
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

        assert_eq!(region.node_count(), 7);
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
        assert_eq!(inner_region.node_count(), 3);
        assert_eq!(inner_region.edge_count(), 2);
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

        assert_eq!(region.node_count(), extracted.node_count());
        assert_eq!(region.root_type(), extracted.root_type());

        Ok(())
    }
}
