use portgraph::{NodeIndex, PortIndex, PortOffset};
use thiserror::Error;

use crate::ops::validate::OpTypeValidator;
use crate::ops::{ModuleOp, OpType};
use crate::types::EdgeKind;
use crate::Hugr;

/// HUGR invariant checks.

impl Hugr {
    /// Check the validity of the HUGR.
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Root node must be a root in the hierarchy, and a module root operation.
        if !self.hierarchy.is_root(self.root) {
            return Err(ValidationError::RootNotRoot);
        }
        if self.get_optype(self.root) != &OpType::Module(ModuleOp::Root) {
            return Err(ValidationError::InvalidRootOpType {
                node: self.root,
                optype: self.get_optype(self.root).clone(),
            });
        }

        // Node-specific checks
        for node in self.graph.nodes_iter() {
            self.validate_node(node)?;
        }

        Ok(())
    }

    /// Check the constraints on a single node.
    ///
    /// This includes:
    /// - Matching the number of ports with the signature
    /// - Dataflow ports are correct. See `validate_df_port`
    fn validate_node(&self, node: NodeIndex) -> Result<(), ValidationError> {
        let optype = self.get_optype(node);
        let sig = optype.signature();

        // The parent must be compatible with the node operation.
        if node != self.root {
            let Some(parent) = self.get_parent(node) else {
                return Err(ValidationError::NoParent { node });
            };

            let parent_optype = self.get_optype(parent);
            if !optype.is_valid_parent(parent_optype) {
                return Err(ValidationError::InvalidParent {
                    node,
                    optype: optype.clone(),
                    parent_optype: parent_optype.clone(),
                });
            }
        }

        // TODO: Check the other `OpTypeValidator` constraints.

        // Check that we have enough ports.
        // The actual number may be larger than the signature if non-dataflow ports are present.
        let mut df_inputs = sig.input.len();
        let df_outputs = sig.output.len();
        if sig.const_input.is_some() {
            df_inputs += 1
        };
        if self.graph.num_inputs(node) < df_inputs || self.graph.num_outputs(node) < df_outputs {
            return Err(ValidationError::WrongNumberOfPorts {
                node,
                optype: optype.clone(),
                actual_inputs: sig.input.len(),
                actual_outputs: sig.output.len(),
            });
        }

        // Check port connections
        for (i, port) in self.graph.inputs(node).enumerate() {
            let offset = PortOffset::new_incoming(i);
            self.validate_port(node, port, offset, optype)?;
        }
        for (i, port) in self.graph.outputs(node).enumerate() {
            let offset = PortOffset::new_outgoing(i);
            self.validate_port(node, port, offset, optype)?;
        }

        Ok(())
    }

    /// Check whether a port is valid.
    /// - It must be connected
    /// - The linked port must have a compatible type
    fn validate_port(
        &self,
        node: NodeIndex,
        port: PortIndex,
        offset: PortOffset,
        optype: &OpType,
    ) -> Result<(), ValidationError> {
        let port_kind = optype.port_kind(offset).unwrap();

        // Ports must always be connected
        let Some(link) = self.graph.port_link(port) else {
            return Err(ValidationError::UnconnectedPort {
                node,
                port: offset,
                port_kind,
            });
        };

        let other_node = self.graph.port_node(link).unwrap();
        let other_offset = self.graph.port_offset(link).unwrap();
        let other_op = self.get_optype(other_node);

        let Some(other_kind) = other_op.port_kind(other_offset) else {
            // The number of ports in `other_node` does not match the operation definition.
            // This should be caught by `validate_node`.
            return Err(self.validate_node(other_node).unwrap_err());
        };
        // TODO: We will require some "unifiable" comparison instead of strict equality, to allow for pre-type inference hugrs.
        if other_kind != port_kind {
            return Err(ValidationError::IncompatiblePorts {
                port: (node, offset, port_kind),
                other: (other_node, other_offset, other_kind),
            });
        }
        Ok(())
    }
}

/// Errors that can occur while validating a Hugr.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidationError {
    /// The root node is not a root in the hierarchy.
    #[error("The root node is not a root in the hierarchy.")]
    RootNotRoot,
    /// Invalid root operation type.
    #[error("The operation type {optype:?} is not allowed as a root node. Expected Optype::Module(ModuleType::Root). In node {node:?}.")]
    InvalidRootOpType { node: NodeIndex, optype: OpType },
    /// The node ports do not match the operation signature.
    #[error("The node {node:?} has an invalid number of ports. The operation {optype:?} cannot have {actual_inputs:?} inputs and {actual_outputs:?} outputs.")]
    WrongNumberOfPorts {
        node: NodeIndex,
        optype: OpType,
        actual_inputs: usize,
        actual_outputs: usize,
    },
    /// A dataflow port is not connected.
    #[error("The node {node:?} has an unconnected port {port:?} of type {port_kind:?}.")]
    UnconnectedPort {
        node: NodeIndex,
        port: PortOffset,
        port_kind: EdgeKind,
    },
    /// Connected ports have different types, or non-unifiable types.
    #[error("Connected ports {:?} in node {:?} and {:?} in node {:?} have incompatible kinds. Cannot connect {:?} to {:?}.",
        port.1, port.0, other.1, other.0, port.2, other.2)]
    IncompatiblePorts {
        port: (NodeIndex, PortOffset, EdgeKind),
        other: (NodeIndex, PortOffset, EdgeKind),
    },
    /// The non-root node has no parent.
    #[error("The node {node:?} has no parent.")]
    NoParent { node: NodeIndex },
    /// The parent node is not compatible with the child node.
    #[error("The node {node:?} has an invalid parent. The operation {optype:?} cannot be a child of an operation {parent_optype:?}.")]
    InvalidParent {
        node: NodeIndex,
        optype: OpType,
        parent_optype: OpType,
    },
}

#[cfg(test)]
mod test {
    use crate::builder::BaseBuilder;

    #[test]
    fn test_empty() {
        let b = BaseBuilder::new();
        let hugr = b.finish().unwrap();
        assert_eq!(hugr.validate(), Ok(()));
    }
}
