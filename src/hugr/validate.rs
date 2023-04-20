use std::collections::HashMap;

use portgraph::{Direction, NodeIndex, PortIndex, PortOffset};
use smol_str::SmolStr;
use thiserror::Error;

use crate::ops::validate::ValidParentSet;
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
            let allowed_parents = optype.validity_flags().allowed_parents;
            if !allowed_parents.contains(parent_optype) {
                return Err(ValidationError::InvalidParent {
                    child: node,
                    child_optype: optype.clone(),
                    parent,
                    parent_optype: parent_optype.clone(),
                    expected_parent: allowed_parents,
                });
            }
        }

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

        // Check operation-specific constraints
        self.validate_operation(node, optype)?;

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

        // Avoid double checking connected port types.
        if offset.direction() == Direction::Incoming {
            return Ok(());
        }

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

        // TODO: Check inter-graph edges

        Ok(())
    }

    /// Check operation-specific constraints.
    ///
    /// These are flags defined for each operation type by the [`OpTypeValidator`] trait.
    fn validate_operation(&self, node: NodeIndex, optype: &OpType) -> Result<(), ValidationError> {
        // Container related properties
        // Note: The `is_df_container` check is run by the children in `is_valid_parent`
        let flags = optype.validity_flags();
        if self.hierarchy.child_count(node) > 0 {
            if !flags.is_container {
                return Err(ValidationError::NonContainerWithChildren {
                    node,
                    optype: optype.clone(),
                });
            }

            /*
            let first_child = self.hierarchy.first(node).unwrap();
            let last_child = self.hierarchy.last(node).unwrap();
            let first_child_optype = self.get_optype(first_child);
            let last_child_optype = self.get_optype(last_child);

            if !optype.validate_first_child(first_child_optype) {
                return Err(ValidationError::InvalidChildOpType {
                    parent: node,
                    child: first_child,
                    parent_optype: optype.clone(),
                    child_optype: first_child_optype.clone(),
                    child_position: "first".into(),
                });
            }
            if !optype.validate_last_child(self.get_optype(last_child)) {
                return Err(ValidationError::InvalidChildOpType {
                    parent: node,
                    child: last_child,
                    parent_optype: optype.clone(),
                    child_optype: last_child_optype.clone(),
                    child_position: "last".into(),
                });
            }
            */

            if flags.require_dag {
                self.validate_children_dag(node, optype)?;
            }
        } else if flags.requires_children {
            return Err(ValidationError::ContainerWithoutChildren {
                node,
                optype: optype.clone(),
            });
        }

        Ok(())
    }

    /// Ensure that the children of a node form a direct acyclic graph. That is,
    /// their edges do not form cycles in the graph.
    ///
    /// Inter-graph edges are ignored. Only internal dataflow, constant, or
    /// state order edges are considered.
    fn validate_children_dag(
        &self,
        parent: NodeIndex,
        optype: &OpType,
    ) -> Result<(), ValidationError> {
        let ignore_port = |child: NodeIndex, child_optype: &OpType, port: PortOffset| {
            let kind = child_optype.port_kind(port).unwrap();
            if matches!(kind, EdgeKind::StateOrder | EdgeKind::Value(_)) {
                return true;
            }

            // Ignore ports that are not connected (that property is checked elsewhere)
            let Some(pred_port) = self.graph.port_index(child, port).and_then(|p| self.graph.port_link(p))  else {
                return true;
            };
            let pred = self.graph.port_node(pred_port).unwrap();

            // Ignore inter-graph edges
            //
            // TODO: Can these cause cycles?
            if Some(parent) != self.hierarchy.parent(pred) {
                return true;
            }

            false
        };

        // Number of nodes visited
        let mut nodes_visited = 0;

        // Number of input ports to a node that remain unvisited. Once this
        // reaches zero, the node is added to the candidate list.
        let mut unvisited_ports: HashMap<NodeIndex, usize> = HashMap::new();

        // Candidates with no unvisited predecessors.
        // Initially, all children with no incoming internal edges.
        let mut candidates: Vec<NodeIndex> = {
            self.hierarchy
                .children(parent)
                .filter(|&child| {
                    let child_optype = self.get_optype(child);
                    self.graph
                        .input_offsets(child)
                        .all(|off| ignore_port(child, child_optype, off))
                })
                .collect()
        };

        while let Some(child) = candidates.pop() {
            nodes_visited += 1;
            let child_optype = self.get_optype(child);

            // Add children with no unvisited predecessors to the candidate list.
            for offset in self.graph.output_offsets(child) {
                if ignore_port(child, child_optype, offset) {
                    continue;
                }
                let port = self.graph.port_index(child, offset).unwrap();
                let Some(successor) = self.graph.port_link(port).and_then(|p| self.graph.port_node(p)) else {
                    continue;
                };
                let visit_count = unvisited_ports.entry(successor).or_insert_with(|| {
                    self.graph
                        .input_offsets(successor)
                        .filter(|&p| !ignore_port(successor, child_optype, p))
                        .count()
                });
                *visit_count -= 1;
                if *visit_count == 0 {
                    candidates.push(successor);
                }
            }
        }

        if nodes_visited != self.hierarchy.child_count(parent) {
            return Err(ValidationError::NotADag {
                node: parent,
                optype: optype.clone(),
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
    /// Invalid first/last child.
    #[error("The operation {child_optype:?} is not allowed as a {child_position} child of operation {parent_optype:?}. In child {child:?} of node {parent:?}.")]
    InvalidChildOpType {
        parent: NodeIndex,
        child: NodeIndex,
        parent_optype: OpType,
        child_optype: OpType,
        child_position: SmolStr,
    },
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
    #[error("The operation {parent_optype:?} is not allowed as a parent of operation {child_optype:?}. Expected: {}. In node {child:?} with parent {parent:?}.", expected_parent.set_description())]
    InvalidParent {
        child: NodeIndex,
        child_optype: OpType,
        parent: NodeIndex,
        parent_optype: OpType,
        expected_parent: ValidParentSet,
    },
    /// The node operation is not a container, but has children.
    #[error("The node {node:?} with optype {optype:?} is not a container, but has children.")]
    NonContainerWithChildren { node: NodeIndex, optype: OpType },
    /// The node must have children, but has none.
    #[error("The node {node:?} with optype {optype:?} must have children, but has none.")]
    ContainerWithoutChildren { node: NodeIndex, optype: OpType },
    /// The children of a node have cycles.
    #[error("The operation {optype:?} does not allow cycles in its children. In node {node:?}.")]
    NotADag { node: NodeIndex, optype: OpType },
}

#[cfg(test)]
mod test {
    use crate::builder::BaseBuilder;

    #[test]
    fn test_empty() {
        let b = BaseBuilder::new();
        let hugr = b.finish();
        assert_eq!(hugr.err(), None);
    }
}
