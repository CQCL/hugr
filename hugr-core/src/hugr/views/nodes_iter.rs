use crate::{HugrView, core::HugrNode, hugr::views::SiblingSubgraph};

/// Iterate over nodes within a graph view (Hugr, SiblingSubgraph, etc).
pub trait NodesIter {
    /// The type of nodes in the graph.
    type Node;

    /// Iterate over the nodes in the graph.
    fn nodes(&self) -> impl Iterator<Item = Self::Node> + '_;
}

impl<H: HugrView> NodesIter for H {
    type Node = H::Node;

    fn nodes(&self) -> impl Iterator<Item = Self::Node> + '_ {
        self.nodes()
    }
}

impl<N: HugrNode> NodesIter for SiblingSubgraph<N> {
    type Node = N;

    fn nodes(&self) -> impl Iterator<Item = Self::Node> + '_ {
        self.nodes().iter().copied()
    }
}
