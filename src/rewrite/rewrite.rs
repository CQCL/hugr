use std::collections::HashMap;

use portgraph::substitute::OpenGraph;
use portgraph::{NodeIndex, PortIndex};
use thiserror::Error;

use crate::Hugr;

/// A subset of the nodes in a sibling graph, i.e. all with the same parent,
/// and the ports that it is connected to.
#[derive(Debug, Clone, Default)]
pub struct SiblingSubgraph {
    /// Nodes in the subgraph.
    subgraph: portgraph::substitute::BoundedSubgraph,
}

impl SiblingSubgraph {
    /// Creates a new bounded subgraph.
    ///
    /// TODO: We should be able to automatically detect dangling ports by
    /// finding inputs and outputs in `hugr` that are connected to things
    /// outside. Can we do that efficiently?
    pub fn new(
        hugr: &Hugr,
        nodes: impl IntoIterator<Item = NodeIndex>,
    ) -> Result<Self, SiblingError> {
        let nodes: Vec<NodeIndex> = nodes.into_iter().collect();
        SiblingSubgraph::validate_siblings(hugr, &nodes)?;
        todo!()
    }

    fn validate_siblings(hugr: &Hugr, nodes: &Vec<NodeIndex>) -> Result<(), SiblingError> {
        let parent = hugr.get_parent(*nodes.first().ok_or(SiblingError::Empty)?);
        match parent {
            None => {
                if nodes.len() != 1 || nodes[0] != hugr.root() {
                    return Err(SiblingError::OnlyRoot);
                }
            }
            Some(_) => {
                for p_idx in nodes.iter().map(|n| hugr.get_parent(*n)) {
                    if p_idx != parent {
                        return Err(SiblingError::MultipleParents(parent, p_idx));
                    }
                }
            }
        }
        Ok(())
    }
}

/// A graph with explicit input and output ports.
#[derive(Clone, Default, Debug)]
pub struct OpenHugr {
    /// The graph.
    pub hugr: Hugr,
    /// Incoming dangling ports in the graph.
    pub dangling_inputs: Vec<PortIndex>,
    /// Outgoing dangling ports in the graph.
    pub dangling_outputs: Vec<PortIndex>,
}

impl OpenHugr {
    /// Creates a new open graph.
    ///
    /// TODO: We should be able to automatically detect dangling ports by
    /// finding inputs and outputs in `hugr` that are connected to things
    /// outside. Can we do that efficiently?
    pub fn new(_hugr: Hugr) -> Self {
        todo!()
    }

    /// Extracts the internal open graph, and returns the Hugr with additional components on the side.
    ///
    /// The returned Hugr will have no graph information.
    pub fn into_parts(self) -> (OpenGraph, Hugr) {
        let OpenHugr {
            mut hugr,
            dangling_inputs,
            dangling_outputs,
        } = self;
        let graph = std::mem::take(&mut hugr.graph);
        (
            OpenGraph {
                graph,
                dangling_inputs,
                dangling_outputs,
            },
            hugr,
        )
    }
}

pub type ParentsMap = HashMap<NodeIndex, NodeIndex>;

/// A rewrite operation that replaces a subgraph with another graph.
/// Includes the new weights for the nodes in the replacement graph.
#[derive(Debug, Clone)]
pub struct Rewrite {
    /// The subgraph to be replaced.
    subgraph: SiblingSubgraph,
    /// The replacement graph. This should be a forest, i.e. the nodes without parents
    /// will be assigned the same parent as the nodes in the subgraph being replaced.
    replacement: OpenHugr,
    /// A map from nodes in the subgraph to be replaced to nodes in the replacement graph.
    /// For each key-value pair, all children of the key will be transferred to be children of the value.
    parents: ParentsMap,
}

impl Rewrite {
    /// Creates a new rewrite operation.
    pub fn new(
        subgraph: SiblingSubgraph,
        replacement: OpenHugr,
        parents: impl Into<ParentsMap>,
    ) -> Self {
        Self {
            subgraph,
            replacement,
            parents: parents.into(),
        }
    }

    /// Extracts the internal graph rewrite, and returns the replacement Hugr
    /// with additional components on the side.
    ///
    /// The returned Hugr will have no graph information.
    pub(crate) fn into_parts(self) -> (portgraph::substitute::Rewrite, Hugr, ParentsMap) {
        let (open_graph, replacement) = self.replacement.into_parts();
        (
            portgraph::substitute::Rewrite::new(self.subgraph.subgraph, open_graph),
            replacement,
            self.parents,
        )
    }

    /// Checks that the rewrite is valid.
    ///
    /// This includes having a convex subgraph (TODO: include definition), and
    /// having matching numbers of ports on the boundaries.
    pub fn verify(&self) -> Result<(), RewriteError> {
        self.verify_convexity()?;
        self.verify_boundaries()?;
        Ok(())
    }

    pub fn verify_convexity(&self) -> Result<(), RewriteError> {
        todo!()
    }

    pub fn verify_boundaries(&self) -> Result<(), RewriteError> {
        todo!()
    }
}

/// Error generated when a rewrite fails.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RewriteError {
    /// The rewrite failed because the boundary defined by the
    /// [`Rewrite`] could not be matched to the dangling ports of the
    /// [`OpenHugr`].
    #[error("The boundary defined by the rewrite could not be matched to the dangling ports of the OpenHugr")]
    BoundarySize(#[source] portgraph::substitute::RewriteError),
    /// There was an error connecting the ports of the [`OpenHugr`] to the
    /// boundary.
    #[error("An error occurred while connecting the ports of the OpenHugr to the boundary")]
    ConnectionError(#[source] portgraph::LinkError),
    /// The rewrite target is not convex
    ///
    /// TODO: include context
    #[error("The rewrite target is not convex")]
    NotConvex(),
}

impl From<portgraph::substitute::RewriteError> for RewriteError {
    fn from(e: portgraph::substitute::RewriteError) -> Self {
        match e {
            portgraph::substitute::RewriteError::BoundarySize => Self::BoundarySize(e),
            portgraph::substitute::RewriteError::Link(e) => Self::ConnectionError(e),
        }
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum SiblingError {
    #[error("No nodes in subgraph")]
    Empty,
    #[error("Only the root node may have no parent")]
    OnlyRoot,
    #[error("Nodes in the subgraph were not siblings")]
    MultipleParents(Option<NodeIndex>, Option<NodeIndex>),
}
