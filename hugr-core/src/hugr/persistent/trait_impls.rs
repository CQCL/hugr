use crate::{
    hugr::internal::HugrInternals, hugr::patch::ApplyPatch, Direction, Hugr, HugrView, Port,
};

use super::{patch_graph::PatchNode, PersistentHugr, PersistentReplacement};

impl ApplyPatch<PersistentHugr> for PersistentReplacement {
    type Outcome = ();
    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply(self, h: &mut PersistentHugr) -> Result<Self::Outcome, Self::Error> {
        todo!()
    }
}

// TODO: It is impossible to satisfy this impl, must resolve https://github.com/CQCL/hugr/issues/1926
impl HugrInternals for PersistentHugr {
    type Portgraph<'p> = portgraph::PortGraph;

    type Node = PatchNode;

    fn portgraph(&self) -> Self::Portgraph<'_> {
        todo!()
    }

    fn base_hugr(&self) -> &Hugr {
        todo!()
    }

    fn root_node(&self) -> Self::Node {
        todo!()
    }

    fn get_pg_index(&self, node: Self::Node) -> portgraph::NodeIndex {
        todo!()
    }

    fn get_node(&self, index: portgraph::NodeIndex) -> Self::Node {
        todo!()
    }
}

impl HugrView for PersistentHugr {
    /// Returns whether the node exists.
    fn contains_node(&self, node: Self::Node) -> bool {
        todo!()
    }

    /// Returns the number of nodes in the hugr.
    fn node_count(&self) -> usize {
        todo!()
    }

    /// Returns the number of edges in the hugr.
    fn edge_count(&self) -> usize {
        todo!()
    }

    /// Iterates over the nodes in the port graph.
    fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone {
        todo!();
        [].into_iter()
    }

    /// Iterator over ports of node in a given direction.
    fn node_ports(&self, node: Self::Node, dir: Direction) -> impl Iterator<Item = Port> + Clone {
        todo!();
        [].into_iter()
    }

    fn all_node_ports(&self, node: Self::Node) -> impl Iterator<Item = Port> + Clone {
        todo!();
        [].into_iter()
    }

    /// Iterator over the nodes and ports connected to a port.
    fn linked_ports(
        &self,
        node: Self::Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Self::Node, Port)> + Clone {
        todo!();
        [].into_iter()
    }

    /// Iterator the links between two nodes.
    fn node_connections(
        &self,
        node: Self::Node,
        other: Self::Node,
    ) -> impl Iterator<Item = [Port; 2]> + Clone {
        todo!();
        [].into_iter()
    }

    /// Number of ports in node for a given direction.
    fn num_ports(&self, node: Self::Node, dir: Direction) -> usize {
        todo!()
    }

    /// Return iterator over the direct children of node.
    fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone {
        todo!();
        [].into_iter()
    }

    /// Iterates over neighbour nodes in the given direction.
    /// May contain duplicates if the graph has multiple links between nodes.
    fn neighbours(
        &self,
        node: Self::Node,
        dir: Direction,
    ) -> impl Iterator<Item = Self::Node> + Clone {
        todo!();
        [].into_iter()
    }

    /// Iterates over the input and output neighbours of the `node` in sequence.
    fn all_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        todo!();
        [].into_iter()
    }
}
