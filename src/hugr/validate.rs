use portgraph::algorithms::toposort_filtered;
use portgraph::{Direction, NodeIndex, PortIndex, PortOffset};
use thiserror::Error;

use crate::ops::validate::{ChildrenValidationError, ValidOpSet};
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

        // The Hugr can have only one root node.
        if node != self.root {
            let Some(parent) = self.get_parent(node) else {
                return Err(ValidationError::NoParent { node });
            };

            let parent_optype = self.get_optype(parent);
            let allowed_children = parent_optype.validity_flags().allowed_children;
            if !allowed_children.contains(optype) {
                return Err(ValidationError::InvalidParent {
                    child: node,
                    child_optype: optype.clone(),
                    parent,
                    parent_optype: parent_optype.clone(),
                    allowed_children,
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
    /// These are flags defined for each operation type as an [`OpValidityFlags`] object.
    fn validate_operation(&self, node: NodeIndex, optype: &OpType) -> Result<(), ValidationError> {
        let flags = optype.validity_flags();

        if self.hierarchy.child_count(node) > 0 {
            if flags.allowed_children.is_empty() {
                return Err(ValidationError::NonContainerWithChildren {
                    node,
                    optype: optype.clone(),
                });
            }

            let first_child = self.get_optype(self.hierarchy.first(node).unwrap());
            if !flags.allowed_first_child.contains(first_child) {
                return Err(ValidationError::InvalidBoundaryChild {
                    parent: node,
                    parent_optype: optype.clone(),
                    optype: first_child.clone(),
                    expected: flags.allowed_first_child,
                    position: "first",
                });
            }

            let last_child = self.get_optype(self.hierarchy.last(node).unwrap());
            if !flags.allowed_last_child.contains(last_child) {
                return Err(ValidationError::InvalidBoundaryChild {
                    parent: node,
                    parent_optype: optype.clone(),
                    optype: last_child.clone(),
                    expected: flags.allowed_last_child,
                    position: "last",
                });
            }

            // Additional validations running over the full list of children optypes
            let children_optypes = self
                .hierarchy
                .children(node)
                .map(|c| (c, self.get_optype(c)));
            if let Err(source) = optype.validate_children(children_optypes) {
                return Err(ValidationError::InvalidChildren {
                    parent: node,
                    parent_optype: optype.clone(),
                    source,
                });
            }

            if flags.requires_dag {
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
        let Some(first_child) = self.hierarchy.first(parent) else {
            // No children, nothing to do
            return Ok(());
        };

        let topo = toposort_filtered(
            &self.graph,
            [first_child],
            Direction::Outgoing,
            |_| true,
            |n, p| self.df_port_filter(n, p),
        );

        // Compute the number of nodes visited and keep the last one.
        let (nodes_visited, last_node) = topo.fold((0, None), |(n, _), node| (n + 1, Some(node)));

        if nodes_visited != self.hierarchy.child_count(parent)
            || last_node != self.hierarchy.last(parent)
        {
            return Err(ValidationError::NotADag {
                node: parent,
                optype: optype.clone(),
            });
        }

        Ok(())
    }

    /// A filter function for internal dataflow edges.
    ///
    /// Returns `true` for ports that connect to a sibling node with a value or
    /// state order edge.
    fn df_port_filter(&self, node: NodeIndex, port: PortIndex) -> bool {
        let offset = self.graph.port_offset(port).unwrap();
        let node_optype = self.get_optype(node);

        let kind = node_optype.port_kind(offset).unwrap();
        if !matches!(kind, EdgeKind::StateOrder | EdgeKind::Value(_)) {
            return false;
        }

        // Ignore ports that are not connected (that property is checked elsewhere)
        let Some(other_port) = self.graph.port_index(node, offset).and_then(|p| self.graph.port_link(p))  else {
                return false;
            };
        let other = self.graph.port_node(other_port).unwrap();

        // Ignore inter-graph edges
        if self.hierarchy.parent(node) != self.hierarchy.parent(other) {
            return false;
        }

        true
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
    #[error("The operation {parent_optype:?} cannot contain a {child_optype:?} as a child. Allowed children: {}. In node {child:?} with parent {parent:?}.", allowed_children.set_description())]
    InvalidParent {
        child: NodeIndex,
        child_optype: OpType,
        parent: NodeIndex,
        parent_optype: OpType,
        allowed_children: ValidOpSet,
    },
    /// Invalid first/last child.
    #[error("A {optype:?} operation cannot be the {position} child of a {parent_optype:?}. Expected {expected}. In parent node {parent:?}")]
    InvalidBoundaryChild {
        parent: NodeIndex,
        parent_optype: OpType,
        optype: OpType,
        expected: ValidOpSet,
        position: &'static str,
    },
    /// The children list has invalid elements.
    #[error(
        "An operation {parent_optype:?} contains invalid children: {source}. In parent {parent:?}, child {child:?}",
        child=source.child(),
    )]
    InvalidChildren {
        parent: NodeIndex,
        parent_optype: OpType,
        source: ChildrenValidationError,
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
    use crate::hugr::HugrMut;

    #[test]
    fn test_empty() {
        let b = HugrMut::new();
        let hugr = b.finish();
        assert_eq!(hugr.err(), None);
    }
}
