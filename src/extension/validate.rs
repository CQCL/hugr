//! Validation routines for instantiations of a resource ops and types in a
//! Hugr.

use std::collections::HashMap;

use thiserror::Error;

use crate::hugr::NodeType;
use crate::{Direction, Hugr, HugrView, Node, Port};

use super::ExtensionSet;

/// Context for validating the resource requirements defined in a Hugr.
#[derive(Debug, Clone, Default)]
pub struct ExtensionValidator {
    /// Resource requirements associated with each edge
    extensions: HashMap<(Node, Direction), ExtensionSet>,
}

impl ExtensionValidator {
    /// Initialise a new resource validator, pre-computing the resource
    /// requirements for each node in the Hugr.
    pub fn new(hugr: &Hugr) -> Self {
        let mut validator = ExtensionValidator {
            extensions: HashMap::new(),
        };

        for node in hugr.nodes() {
            validator.gather_extensions(&node, hugr.get_nodetype(node));
        }

        validator
    }

    /// Use the signature supplied by a dataflow node to work out the
    /// resource requirements for all of its input and output edges, then put
    /// those requirements in the resource validation context.
    fn gather_extensions(&mut self, node: &Node, node_type: &NodeType) {
        if let Some(sig) = node_type.signature() {
            for dir in Direction::BOTH {
                assert!(self
                    .extensions
                    .insert((*node, dir), sig.get_extension(&dir))
                    .is_none());
            }
        }
    }

    /// Get the input or output resource requirements for a particular node in the Hugr.
    ///
    /// # Errors
    ///
    /// If the node resources are missing.
    fn query_extensions(
        &self,
        node: Node,
        dir: Direction,
    ) -> Result<&ExtensionSet, ExtensionError> {
        self.extensions
            .get(&(node, dir))
            .ok_or(ExtensionError::MissingInputExtensions(node))
    }

    /// Check that two `PortIndex` have compatible resource requirements,
    /// according to the information accumulated by `gather_resources`.
    ///
    /// This resource checking assumes that free resource variables
    ///   (e.g. implicit lifting of `A -> B` to `[R]A -> [R]B`)
    /// and adding of lift nodes
    ///   (i.e. those which transform an edge from `A` to `[R]A`)
    /// has already been done.
    pub fn check_extensions_compatible(
        &self,
        src: &(Node, Port),
        tgt: &(Node, Port),
    ) -> Result<(), ExtensionError> {
        let rs_src = self.query_extensions(src.0, Direction::Outgoing)?;
        let rs_tgt = self.query_extensions(tgt.0, Direction::Incoming)?;

        if rs_src == rs_tgt {
            Ok(())
        } else if rs_src.is_subset(rs_tgt) {
            // The extra resource requirements reside in the target node.
            // If so, we can fix this mismatch with a lift node
            Err(ExtensionError::TgtExceedsSrcExtensions {
                from: src.0,
                from_offset: src.1,
                from_extensions: rs_src.clone(),
                to: tgt.0,
                to_offset: tgt.1,
                to_extensions: rs_tgt.clone(),
            })
        } else {
            Err(ExtensionError::SrcExceedsTgtExtensions {
                from: src.0,
                from_offset: src.1,
                from_extensions: rs_src.clone(),
                to: tgt.0,
                to_offset: tgt.1,
                to_extensions: rs_tgt.clone(),
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
    ) -> Result<(), ExtensionError> {
        let parent_input_resources = self.query_extensions(parent, Direction::Incoming)?;
        let parent_output_resources = self.query_extensions(parent, Direction::Outgoing)?;
        for dir in Direction::BOTH {
            let input_resources = self.query_extensions(input, dir)?;
            let output_resources = self.query_extensions(output, dir)?;
            if parent_input_resources != input_resources {
                return Err(ExtensionError::ParentIOExtensionMismatch {
                    parent,
                    parent_extensions: parent_input_resources.clone(),
                    child: input,
                    child_extensions: input_resources.clone(),
                });
            };
            if parent_output_resources != output_resources {
                return Err(ExtensionError::ParentIOExtensionMismatch {
                    parent,
                    parent_extensions: parent_output_resources.clone(),
                    child: output,
                    child_extensions: output_resources.clone(),
                });
            };
        }
        Ok(())
    }
}

/// Errors that can occur while validating a Hugr.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum ExtensionError {
    /// Missing lift node
    #[error("Resources at target node {to:?} ({to_offset:?}) ({to_extensions}) exceed those at source {from:?} ({from_offset:?}) ({from_extensions})")]
    TgtExceedsSrcExtensions {
        from: Node,
        from_offset: Port,
        from_extensions: ExtensionSet,
        to: Node,
        to_offset: Port,
        to_extensions: ExtensionSet,
    },
    /// Too many resource requirements coming from src
    #[error("Resources at source node {from:?} ({from_offset:?}) ({from_extensions}) exceed those at target {to:?} ({to_offset:?}) ({to_extensions})")]
    SrcExceedsTgtExtensions {
        from: Node,
        from_offset: Port,
        from_extensions: ExtensionSet,
        to: Node,
        to_offset: Port,
        to_extensions: ExtensionSet,
    },
    #[error("Missing input resources for node {0:?}")]
    MissingInputExtensions(Node),
    #[error("Resources of I/O node ({child:?}) {child_extensions:?} don't match those expected by parent node ({parent:?}): {parent_extensions:?}")]
    ParentIOExtensionMismatch {
        parent: Node,
        parent_extensions: ExtensionSet,
        child: Node,
        child_extensions: ExtensionSet,
    },
}
