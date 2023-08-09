//! Validation routines for instantiations of a resource ops and types in a
//! Hugr.

use std::collections::HashMap;

use thiserror::Error;

use crate::hugr::NodeType;
use crate::{Direction, Hugr, HugrView, Node, Port};

use super::ResourceSet;

/// Context for validating the resource requirements defined in a Hugr.
#[derive(Debug, Clone, Default)]
pub struct ResourceValidator {
    /// Resource requirements associated with each edge
    resources: HashMap<(Node, Direction), ResourceSet>,
}

impl ResourceValidator {
    /// Initialise a new resource validator, pre-computing the resource
    /// requirements for each node in the Hugr.
    pub fn new(hugr: &Hugr) -> Self {
        let mut validator = ResourceValidator {
            resources: HashMap::new(),
        };

        for node in hugr.nodes() {
            validator.gather_resources(&node, hugr.get_nodetype(node));
        }

        validator
    }

    /// Use the signature supplied by a dataflow node to work out the
    /// resource requirements for all of its input and output edges, then put
    /// those requirements in the resource validation context.
    fn gather_resources(&mut self, node: &Node, node_type: &NodeType) {
        if let Some(sig) = node_type.signature() {
            for dir in Direction::BOTH {
                assert!(self
                    .resources
                    .insert((*node, dir), sig.get_resources(&dir))
                    .is_none());
            }
        }
    }

    /// Get the input or output resource requirements for a particular node in the Hugr.
    ///
    /// # Errors
    ///
    /// If the node resources are missing.
    fn query_resources(&self, node: Node, dir: Direction) -> Result<&ResourceSet, ResourceError> {
        self.resources
            .get(&(node, dir))
            .ok_or(ResourceError::MissingInputResources(node))
    }

    /// Check that two `PortIndex` have compatible resource requirements,
    /// according to the information accumulated by `gather_resources`.
    ///
    /// This resource checking assumes that free resource variables
    ///   (e.g. implicit lifting of `A -> B` to `[R]A -> [R]B`)
    /// and adding of lift nodes
    ///   (i.e. those which transform an edge from `A` to `[R]A`)
    /// has already been done.
    pub fn check_resources_compatible(
        &self,
        src: &(Node, Port),
        tgt: &(Node, Port),
    ) -> Result<(), ResourceError> {
        let rs_src = self.query_resources(src.0, Direction::Outgoing)?;
        let rs_tgt = self.query_resources(tgt.0, Direction::Incoming)?;

        if rs_src == rs_tgt {
            Ok(())
        } else if rs_src.is_subset(rs_tgt) {
            // The extra resource requirements reside in the target node.
            // If so, we can fix this mismatch with a lift node
            Err(ResourceError::TgtExceedsSrcResources {
                from: src.0,
                from_offset: src.1,
                from_resources: rs_src.clone(),
                to: tgt.0,
                to_offset: tgt.1,
                to_resources: rs_tgt.clone(),
            })
        } else {
            Err(ResourceError::SrcExceedsTgtResources {
                from: src.0,
                from_offset: src.1,
                from_resources: rs_src.clone(),
                to: tgt.0,
                to_offset: tgt.1,
                to_resources: rs_tgt.clone(),
            })
        }
    }

    /// Check that a pair of input and output nodes declare the same resources
    /// as in the signature of their parents.
    pub fn validate_io_resources(
        &self,
        parent: Node,
        input: Node,
        output: Node,
    ) -> Result<(), ResourceError> {
        let parent_input_resources = self.query_resources(parent, Direction::Incoming)?;
        let parent_output_resources = self.query_resources(parent, Direction::Outgoing)?;
        for dir in Direction::BOTH {
            let input_resources = self.query_resources(input, dir)?;
            let output_resources = self.query_resources(output, dir)?;
            if parent_input_resources != input_resources {
                return Err(ResourceError::ParentIOResourceMismatch {
                    parent,
                    parent_resources: parent_input_resources.clone(),
                    child: input,
                    child_resources: input_resources.clone(),
                });
            };
            if parent_output_resources != output_resources {
                return Err(ResourceError::ParentIOResourceMismatch {
                    parent,
                    parent_resources: parent_output_resources.clone(),
                    child: output,
                    child_resources: output_resources.clone(),
                });
            };
        }
        Ok(())
    }
}

/// Errors that can occur while validating a Hugr.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum ResourceError {
    /// Missing lift node
    #[error("Resources at target node {to:?} ({to_offset:?}) ({to_resources}) exceed those at source {from:?} ({from_offset:?}) ({from_resources})")]
    TgtExceedsSrcResources {
        from: Node,
        from_offset: Port,
        from_resources: ResourceSet,
        to: Node,
        to_offset: Port,
        to_resources: ResourceSet,
    },
    /// Too many resource requirements coming from src
    #[error("Resources at source node {from:?} ({from_offset:?}) ({from_resources}) exceed those at target {to:?} ({to_offset:?}) ({to_resources})")]
    SrcExceedsTgtResources {
        from: Node,
        from_offset: Port,
        from_resources: ResourceSet,
        to: Node,
        to_offset: Port,
        to_resources: ResourceSet,
    },
    #[error("Missing input resources for node {0:?}")]
    MissingInputResources(Node),
    #[error("Resources of I/O node ({child:?}) {child_resources:?} don't match those expected by parent node ({parent:?}): {parent_resources:?}")]
    ParentIOResourceMismatch {
        parent: Node,
        parent_resources: ResourceSet,
        child: Node,
        child_resources: ResourceSet,
    },
}
