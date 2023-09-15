//! DescendantsGraph: view onto the subgraph of the HUGR starting from a root
//! (all descendants at all depths).

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{LinkView, MultiPortGraph, PortIndex, PortView};

use crate::hugr::HugrError;
use crate::ops::handle::NodeHandle;
use crate::ops::OpTrait;
use crate::{Direction, Hugr, Node, Port};

use super::{sealed::HugrInternals, HierarchyView, HugrView, Holder};

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

impl<'g, Root> HugrView<'g> for DescendantsGraph<'g, Root>
where
    Root: NodeHandle + 'g,
{
    type RootHandle = Root;

    type Nodes<'a> = MapInto<<RegionGraph<'g> as PortView>::Nodes<'a>, Node>
    where
        Self: 'a;
    
    type Nodes2<'a> = Holder<RegionGraph<'g>, MapInto<<RegionGraph<'g> as PortView>::Nodes<'a>, Node>> where 'g: 'a;

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

    type NodeConnections<'a> = MapWithCtx<
        <RegionGraph<'g> as LinkView>::NodeConnections<'a>,
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

    fn nodes2(&self) -> Self::Nodes2<'g> {
        Holder::new(
            self.graph.clone(),
            |g| g.nodes_iter().map_into()
        )
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

impl<'a, Root> HierarchyView<'a> for DescendantsGraph<'a, Root>
where
    Root: NodeHandle + 'a,
{
    fn try_new(hugr: &'a impl HugrView<'a>, root: Node) -> Result<Self, HugrError> {
        hugr.valid_node(root)?;
        let root_tag = hugr.get_optype(root).tag();
        if !Root::TAG.is_superset(root_tag) {
            return Err(HugrError::InvalidNode(root));
        }
        let hugr = hugr.base_hugr();
        Ok(Self {
            root,
            graph: RegionGraph::new_region(&hugr.graph, &hugr.hierarchy, root.index),
            hugr,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<'g, Root> super::sealed::HugrInternals<'g> for DescendantsGraph<'g, Root>
where
    Root: NodeHandle,
{
    type Portgraph<'p> = &'p RegionGraph<'g> where Self: 'p;

    #[inline]
    fn portgraph(&self) -> Self::Portgraph<'_> {
        &self.graph
    }

    #[inline]
    fn base_hugr(&self) -> &'g Hugr {
        self.hugr
    }

    #[inline]
    fn root_node(&self) -> Node {
        self.root
    }
}

#[cfg(test)]
pub(super) mod test {
    use crate::{
        builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        ops::handle::NodeHandle,
        std_extensions::quantum::test::h_gate,
        type_row,
        types::{FunctionType, Type},
    };

    use super::*;

    const NAT: Type = crate::extension::prelude::USIZE_T;
    const QB: Type = crate::extension::prelude::QB_T;

    /// Make a module hugr with a fn definition containing an inner dfg node.
    ///
    /// Returns the hugr, the fn node id, and the nested dgf node id.
    pub(in crate::hugr::views) fn make_module_hgr(
    ) -> Result<(Hugr, Node, Node), Box<dyn std::error::Error>> {
        let mut module_builder = ModuleBuilder::new();

        let (f_id, inner_id) = {
            let mut func_builder = module_builder.define_function(
                "main",
                FunctionType::new(type_row![NAT, QB], type_row![NAT, QB]).pure(),
            )?;

            let [int, qb] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb])?;

            let inner_id = {
                let inner_builder = func_builder.dfg_builder(
                    FunctionType::new(type_row![NAT], type_row![NAT]),
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
        let hugr = module_builder.finish_prelude_hugr()?;
        Ok((hugr, f_id.handle().node(), inner_id.handle().node()))
    }

    #[test]
    fn full_region() -> Result<(), Box<dyn std::error::Error>> {
        let (hugr, def, inner) = make_module_hgr()?;

        let region: DescendantsGraph = DescendantsGraph::try_new(&hugr, def)?;

        assert_eq!(region.node_count(), 7);
        assert!(region.nodes().all(|n| n == def
            || hugr.get_parent(n) == Some(def)
            || hugr.get_parent(n) == Some(inner)));
        assert_eq!(region.children(inner).count(), 2);

        assert_eq!(
            region.get_function_type(),
            Some(&FunctionType::new(type_row![NAT, QB], type_row![NAT, QB]))
        );
        let inner_region: DescendantsGraph = DescendantsGraph::try_new(&hugr, inner)?;
        assert_eq!(
            inner_region.get_function_type(),
            Some(&FunctionType::new(type_row![NAT], type_row![NAT]))
        );

        Ok(())
    }
}
