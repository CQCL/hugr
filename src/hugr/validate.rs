use std::cmp::Ordering;

use portgraph::{NodeIndex, PortOffset};
use thiserror::Error;

use crate::ops::{ModuleOp, OpType};
use crate::types::{EdgeKind, Signature, SimpleType};
use crate::Hugr;

/// HUGR invariant checks.

impl Hugr {
    /// Check the validity of the HUGR.
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Root node must be a root in the hierarchy, and a module root operation.
        if !self.hierarchy.is_root(self.root) {
            return Err(ValidationError::RootNotRoot);
        }
        if self.get_optype(self.root) != &ModuleOp::Root.into() {
            return Err(ValidationError::OpTypeNotAllowed {
                node: self.root,
                op_type: self.get_optype(self.root).clone(),
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
        // TODO: Signature correctness, allowed children, children-specific
        // constraints (e.g. first/last types), children DAG, type of parent
        let optype = self.get_optype(node);
        let sig = optype.signature();

        // Check the number of ports.
        let mut df_inputs = sig.input.len();
        let df_outputs = sig.output.len();
        if sig.const_input.is_some() {
            df_inputs += 1
        };
        let valid_inputs = match self.graph.num_inputs(node).cmp(&df_inputs) {
            Ordering::Less => false,
            Ordering::Equal => true,
            Ordering::Greater => optype.other_inputs().is_some(),
        };
        let valid_outputs = match self.graph.num_outputs(node).cmp(&df_outputs) {
            Ordering::Less => false,
            Ordering::Equal => true,
            Ordering::Greater => optype.other_outputs().is_some(),
        };
        if !valid_inputs || !valid_outputs {
            return Err(ValidationError::WrongNumberOfPorts {
                node,
                optype: optype.clone(),
                actual_inputs: sig.input.len(),
                actual_outputs: sig.output.len(),
            });
        }

        // Check port connections
        for input in 0..sig.input.len() {
            let offset = PortOffset::new_incoming(input);
            self.validate_df_port(node, &sig, offset)?;
        }
        for output in 0..sig.output.len() {
            let offset = PortOffset::new_outgoing(output);
            self.validate_df_port(node, &sig, offset)?;
        }
        if sig.const_input.is_some() {
            let offset = PortOffset::new_incoming(sig.input.len());
            self.validate_df_port(node, &sig, offset)?;
        }

        Ok(())
    }

    /// Check that a port is valid.
    /// - Dataflow kinds must be connected
    /// - The connected port must have a compatible type
    fn validate_df_port(
        &self,
        node: NodeIndex,
        sig: &Signature,
        offset: PortOffset,
    ) -> Result<(), ValidationError> {
        let port = self.graph.port_index(node, offset).unwrap();
        let port_kind = sig.get(offset).unwrap();

        let Some(link) = self.graph.port_link(port) else {
            // Dataflow ports must always be connected
            if let EdgeKind::Value(typ) = port_kind {
                return Err(ValidationError::UnconnectedDFPort {
                    node,
                    port: offset,
                    wire_type: typ,
                });
            } else {
                return Ok(());
            }
        };

        let other_node = self.graph.port_node(link).unwrap();
        let other_offset = self.graph.port_offset(link).unwrap();
        let other_op = self.get_optype(other_node);
        let other_sig = other_op.signature();

        let Some(other_kind) = other_sig.get(other_offset) else {
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
    /// Invalid operation type.
    #[error("The node {node:?} is not allowed to have the operation type {op_type:?}.")]
    OpTypeNotAllowed { node: NodeIndex, op_type: OpType },
    /// The node ports do not match the operation signature.
    #[error("The node {node:?} has an invalid number of ports. The operation {optype:?} cannot have {actual_inputs:?} inputs and {actual_outputs:?} outputs.")]
    WrongNumberOfPorts {
        node: NodeIndex,
        optype: OpType,
        actual_inputs: usize,
        actual_outputs: usize,
    },
    /// A dataflow port is not connected.
    #[error("The node {node:?} has an unconnected port {port:?} of type {wire_type:?}.")]
    UnconnectedDFPort {
        node: NodeIndex,
        port: PortOffset,
        wire_type: SimpleType,
    },
    /// Connected ports have different types, or non-unifiable types.
    #[error("Connected ports {:?} in node {:?} and {:?} in node {:?} have incompatible kinds. Cannot connect {:?} to {:?}.",
        port.1, port.0, other.1, other.0, port.2, other.2)]
    IncompatiblePorts {
        port: (NodeIndex, PortOffset, EdgeKind),
        other: (NodeIndex, PortOffset, EdgeKind),
    },
}
