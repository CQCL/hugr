//! HUGR invariant checks.

use std::collections::HashMap;
use std::iter;

use itertools::Itertools;
use portgraph::algorithms::{dominators_filtered, toposort_filtered, DominatorTree};
use portgraph::{Direction, NodeIndex, PortIndex, PortOffset};
use thiserror::Error;

use crate::ops::validate::{
    ChildrenEdgeData, ChildrenValidationError, EdgeValidationError, ValidOpSet,
};
use crate::ops::{ControlFlowOp, DataflowOp, LeafOp, ModuleOp, OpType};
use crate::types::{EdgeKind, SimpleType};
use crate::Hugr;

/// Structure keeping track of pre-computed information used in the validation
/// process.
///
/// TODO: Consider implementing updatable dominator trees and storing it in the
/// Hugr to avoid recomputing it every time.
struct ValidationContext<'a> {
    hugr: &'a Hugr,
    /// Dominator tree for each CFG region, using the container node as index.
    dominators: HashMap<NodeIndex, DominatorTree>,
}

impl Hugr {
    /// Check the validity of the HUGR.
    pub fn validate(&self) -> Result<(), ValidationError> {
        let mut validator = ValidationContext::new(self);
        validator.validate()
    }
}

impl<'a> ValidationContext<'a> {
    /// Create a new validation context.
    pub fn new(hugr: &'a Hugr) -> Self {
        Self {
            hugr,
            dominators: HashMap::new(),
        }
    }

    /// Check the validity of the HUGR.
    pub fn validate(&mut self) -> Result<(), ValidationError> {
        // Root node must be a root in the hierarchy, and a module root operation.
        if !self.hugr.hierarchy.is_root(self.hugr.root) {
            return Err(ValidationError::RootNotRoot {
                node: self.hugr.root,
            });
        }
        if self.hugr.get_optype(self.hugr.root) != &OpType::Module(ModuleOp::Root) {
            return Err(ValidationError::InvalidRootOpType {
                node: self.hugr.root,
                optype: self.hugr.get_optype(self.hugr.root).clone(),
            });
        }

        // Node-specific checks
        for node in self.hugr.graph.nodes_iter() {
            self.validate_node(node)?;
        }

        Ok(())
    }

    /// Returns the dominator tree for a CFG region, identified by its container node.
    /// May compute the dominator tree if it has not been computed yet.
    fn dominator_tree(&mut self, node: NodeIndex) -> &DominatorTree {
        self.dominators.entry(node).or_insert_with(|| {
            let entry = self.hugr.hierarchy.first(node).unwrap();
            dominators_filtered(
                &self.hugr.graph,
                entry,
                Direction::Outgoing,
                |n| matches!(self.hugr.get_optype(n), OpType::BasicBlock(_)),
                |_, _| true,
            )
        })
    }

    /// Check the constraints on a single node.
    ///
    /// This includes:
    /// - Matching the number of ports with the signature
    /// - Dataflow ports are correct. See `validate_df_port`
    fn validate_node(&mut self, node: NodeIndex) -> Result<(), ValidationError> {
        let optype = self.hugr.get_optype(node);
        let sig = optype.signature();
        let flags = optype.validity_flags();

        // The Hugr can have only one root node.
        if node != self.hugr.root {
            let Some(parent) = self.hugr.get_parent(node) else {
                return Err(ValidationError::NoParent { node });
            };

            let parent_optype = self.hugr.get_optype(parent);
            let allowed_children = parent_optype.validity_flags().allowed_children;
            if !allowed_children.contains(optype) {
                return Err(ValidationError::InvalidParentOp {
                    child: node,
                    child_optype: optype.clone(),
                    parent,
                    parent_optype: parent_optype.clone(),
                    allowed_children,
                });
            }
        }

        // Check that we have enough ports. If the `non_df_ports` flag is set
        // for the direction, we require exactly that number of ports after the
        // dataflow ports. Otherwise, we allow any number of extra ports.
        let check_extra_ports = |df_ports: usize, non_df_ports, actual| {
            if let Some(non_df) = non_df_ports {
                df_ports + non_df == actual
            } else {
                df_ports <= actual
            }
        };
        let df_const_input = sig.const_input.is_some() as usize;
        if !check_extra_ports(
            sig.input.len() + df_const_input,
            flags.non_df_ports.0,
            self.hugr.graph.num_inputs(node),
        ) || !check_extra_ports(
            sig.output.len(),
            flags.non_df_ports.1,
            self.hugr.graph.num_outputs(node),
        ) {
            return Err(ValidationError::WrongNumberOfPorts {
                node,
                optype: optype.clone(),
                actual_inputs: sig.input.len(),
                actual_outputs: sig.output.len(),
            });
        }

        // Check port connections
        for (i, port) in self.hugr.graph.inputs(node).enumerate() {
            let offset = PortOffset::new_incoming(i);
            self.validate_port(node, port, offset, optype)?;
        }
        for (i, port) in self.hugr.graph.outputs(node).enumerate() {
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
        &mut self,
        node: NodeIndex,
        port: PortIndex,
        offset: PortOffset,
        optype: &OpType,
    ) -> Result<(), ValidationError> {
        let port_kind = optype.port_kind(offset).unwrap();

        // Ports must always be connected
        let Some(link) = self.hugr.graph.port_link(port) else {
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

        let other_node = self.hugr.graph.port_node(link).unwrap();
        let other_offset = self.hugr.graph.port_offset(link).unwrap();
        let other_op = self.hugr.get_optype(other_node);

        let Some(other_kind) = other_op.port_kind(other_offset) else {
            // The number of ports in `other_node` does not match the operation definition.
            // This should be caught by `validate_node`.
            return Err(self.validate_node(other_node).unwrap_err());
        };
        // TODO: We will require some "unifiable" comparison instead of strict equality, to allow for pre-type inference hugrs.
        if other_kind != port_kind {
            return Err(ValidationError::IncompatiblePorts {
                from: node,
                from_port: offset,
                from_kind: port_kind,
                to: other_node,
                to_port: other_offset,
                to_kind: other_kind,
            });
        }

        self.validate_intergraph_edge(node, offset, optype, other_node, other_offset)?;

        Ok(())
    }

    /// Check operation-specific constraints.
    ///
    /// These are flags defined for each operation type as an [`OpValidityFlags`] object.
    fn validate_operation(&self, node: NodeIndex, optype: &OpType) -> Result<(), ValidationError> {
        let flags = optype.validity_flags();

        if self.hugr.hierarchy.child_count(node) > 0 {
            if flags.allowed_children.is_empty() {
                return Err(ValidationError::NonContainerWithChildren {
                    node,
                    optype: optype.clone(),
                });
            }

            let first_child = self
                .hugr
                .get_optype(self.hugr.hierarchy.first(node).unwrap());
            if !flags.allowed_first_child.contains(first_child) {
                return Err(ValidationError::InvalidBoundaryChild {
                    parent: node,
                    parent_optype: optype.clone(),
                    optype: first_child.clone(),
                    expected: flags.allowed_first_child,
                    position: "first",
                });
            }

            let last_child = self
                .hugr
                .get_optype(self.hugr.hierarchy.last(node).unwrap());
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
                .hugr
                .hierarchy
                .children(node)
                .map(|c| (c, self.hugr.get_optype(c)));
            if let Err(source) = optype.validate_children(children_optypes) {
                return Err(ValidationError::InvalidChildren {
                    parent: node,
                    parent_optype: optype.clone(),
                    source,
                });
            }

            // Additional validations running over the edges of the contained graph
            if let Some(edge_check) = flags.edge_check {
                for source in self.hugr.hierarchy.children(node) {
                    for target in self.hugr.graph.output_neighbours(source) {
                        if self.hugr.hierarchy.parent(target) != Some(node) {
                            continue;
                        }
                        let source_op = self.hugr.get_optype(source);
                        let target_op = self.hugr.get_optype(target);
                        for (source_port, target_port) in
                            self.hugr.graph.get_connections(source, target)
                        {
                            let edge_data = ChildrenEdgeData {
                                source,
                                target,
                                source_port: self.hugr.graph.port_offset(source_port).unwrap(),
                                target_port: self.hugr.graph.port_offset(target_port).unwrap(),
                                source_op: source_op.clone(),
                                target_op: target_op.clone(),
                            };
                            if let Err(source) = edge_check(edge_data) {
                                return Err(ValidationError::InvalidEdges {
                                    parent: node,
                                    parent_optype: optype.clone(),
                                    source,
                                });
                            }
                        }
                    }
                }
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
        let Some(first_child) = self.hugr.hierarchy.first(parent) else {
            // No children, nothing to do
            return Ok(());
        };

        let topo = toposort_filtered(
            &self.hugr.graph,
            [first_child],
            Direction::Outgoing,
            |_| true,
            |n, p| self.df_port_filter(n, p),
        );

        // Compute the number of nodes visited and keep the last one.
        let (nodes_visited, last_node) = topo.fold((0, None), |(n, _), node| (n + 1, Some(node)));

        if nodes_visited != self.hugr.hierarchy.child_count(parent)
            || last_node != self.hugr.hierarchy.last(parent)
        {
            return Err(ValidationError::NotADag {
                node: parent,
                optype: optype.clone(),
            });
        }

        Ok(())
    }

    /// Check inter-graph edges. These are classical value edges between a copy
    /// node and another non-sibling node.
    ///
    /// They come in two flavors depending on the type of the parent node of the
    /// source:
    /// - External edges, from a copy node to a sibling's descendant. There must
    ///   also be an order edge between the copy and the sibling.
    /// - Dominator edges, from a copy node in a beta node to a descendant of a
    ///   post-dominated sibling of the beta.
    fn validate_intergraph_edge(
        &mut self,
        from: NodeIndex,
        from_offset: PortOffset,
        from_optype: &OpType,
        to: NodeIndex,
        to_offset: PortOffset,
    ) -> Result<(), ValidationError> {
        let from_parent = self
            .hugr
            .get_parent(from)
            .expect("Root nodes cannot have ports");
        let to_parent = self.hugr.get_parent(to);
        if Some(from_parent) == to_parent {
            // Regular edge
            return Ok(());
        }

        match from_optype {
            OpType::Module(ModuleOp::Const(_)) => {
                // Inter-graph constant wires do not have restrictions
                return Ok(());
            }
            OpType::Function(DataflowOp::Leaf {
                op: LeafOp::Copy { .. },
            }) => {}
            _ => {
                return Err(InterGraphEdgeError::NonCopySource {
                    from,
                    from_offset,
                    from_optype: from_optype.clone(),
                    to,
                    to_offset,
                }
                .into())
            }
        }

        match from_optype.port_kind(from_offset).unwrap() {
            EdgeKind::Value(SimpleType::Classic(_)) => {}
            ty => {
                return Err(InterGraphEdgeError::NonClassicalData {
                    from,
                    from_offset,
                    to,
                    to_offset,
                    ty,
                }
                .into())
            }
        }

        // To detect either external or dominator edges, we traverse the ancestors
        // of the target until we find either `from_parent` (in the external
        // case), or the parent of `from_parent` (in the dominator case).
        //
        // This search could be sped-up with a pre-computed LCA structure, but
        // for valid Hugrs this search should be very short.
        let from_parent_parent = self
            .hugr
            .get_parent(from_parent)
            .expect("Copy nodes cannot have a root parent.");
        for (ancestor, ancestor_parent) in
            iter::successors(to_parent, |&p| self.hugr.get_parent(p)).tuple_windows()
        {
            if ancestor_parent == from_parent {
                // External edge. Must have an order edge.
                self.hugr
                    .graph
                    .get_connections(from, ancestor)
                    .find(|&(p, _)| {
                        let offset = self.hugr.graph.port_offset(p).unwrap();
                        from_optype.port_kind(offset) == Some(EdgeKind::StateOrder)
                    })
                    .ok_or(InterGraphEdgeError::MissingOrderEdge {
                        from,
                        from_offset,
                        to,
                        to_offset,
                        to_ancestor: ancestor,
                    })?;
                return Ok(());
            } else if ancestor_parent == from_parent_parent {
                // Dominator edge
                let ancestor_parent_op = self.hugr.get_optype(ancestor_parent);
                if !matches!(
                    ancestor_parent_op,
                    OpType::Function(DataflowOp::ControlFlow {
                        op: ControlFlowOp::CFG { .. }
                    })
                ) {
                    return Err(InterGraphEdgeError::NonCFGAncestor {
                        from,
                        from_offset,
                        to,
                        to_offset,
                        ancestor_parent_op: ancestor_parent_op.clone(),
                    }
                    .into());
                }

                // Check domination
                //
                // TODO: Add a more efficient lookup for dominator trees.
                let dominator_tree = self.dominator_tree(ancestor_parent);
                let mut dominators =
                    iter::successors(Some(ancestor), |&n| dominator_tree.immediate_dominator(n));
                if !dominators.any(|n| n == from_parent) {
                    return Err(InterGraphEdgeError::NonDominatedAncestor {
                        from,
                        from_offset,
                        to,
                        to_offset,
                        from_parent,
                        ancestor,
                    }
                    .into());
                }

                return Ok(());
            }
        }

        Err(InterGraphEdgeError::NoRelation {
            from,
            from_offset,
            to,
            to_offset,
        }
        .into())
    }

    /// A filter function for internal dataflow edges.
    ///
    /// Returns `true` for ports that connect to a sibling node with a value or
    /// state order edge.
    fn df_port_filter(&self, node: NodeIndex, port: PortIndex) -> bool {
        let offset = self.hugr.graph.port_offset(port).unwrap();
        let node_optype = self.hugr.get_optype(node);

        let kind = node_optype.port_kind(offset).unwrap();
        if !matches!(kind, EdgeKind::StateOrder | EdgeKind::Value(_)) {
            return false;
        }

        // Ignore ports that are not connected (that property is checked elsewhere)
        let Some(other_port) = self.hugr.graph.port_index(node, offset).and_then(|p| self.hugr.graph.port_link(p))  else {
                return false;
            };
        let other = self.hugr.graph.port_node(other_port).unwrap();

        // Ignore inter-graph edges
        if self.hugr.hierarchy.parent(node) != self.hugr.hierarchy.parent(other) {
            return false;
        }

        true
    }
}

/// Errors that can occur while validating a Hugr.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidationError {
    /// The root node of the Hugr is not a root in the hierarchy.
    #[error("The root node of the Hugr {node:?} is not a root in the hierarchy.")]
    RootNotRoot { node: NodeIndex },
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
    #[error("Connected ports {from_port:?} in node {from:?} and {to_port:?} in node {to:?} have incompatible kinds. Cannot connect {from_kind:?} to {to_kind:?}.")]
    IncompatiblePorts {
        from: NodeIndex,
        from_port: PortOffset,
        from_kind: EdgeKind,
        to: NodeIndex,
        to_port: PortOffset,
        to_kind: EdgeKind,
    },
    /// The non-root node has no parent.
    #[error("The node {node:?} has no parent.")]
    NoParent { node: NodeIndex },
    /// The parent node is not compatible with the child node.
    #[error("The operation {parent_optype:?} cannot contain a {child_optype:?} as a child. Allowed children: {}. In node {child:?} with parent {parent:?}.", allowed_children.set_description())]
    InvalidParentOp {
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
    /// The children graph has invalid edges.
    #[error(
        "An operation {parent_optype:?} contains invalid edges between its children: {source}. In parent {parent:?}, edge from {from:?} port {from_port:?} to {to:?} port {to_port:?}",
        from=source.edge().source,
        from_port=source.edge().source_port,
        to=source.edge().target,
        to_port=source.edge().target_port,
    )]
    InvalidEdges {
        parent: NodeIndex,
        parent_optype: OpType,
        source: EdgeValidationError,
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
    /// There are invalid inter-graph edges.
    #[error(transparent)]
    InterGraphEdgeError(#[from] InterGraphEdgeError),
}

/// Errors related to the inter-graph edge validations
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InterGraphEdgeError {
    /// Inter-Graph edges can only carry classical data
    #[error("Inter-graph edges can only carry classical data. In an inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}) with type {ty:?}.")]
    NonClassicalData {
        from: NodeIndex,
        from_offset: PortOffset,
        to: NodeIndex,
        to_offset: PortOffset,
        ty: EdgeKind,
    },
    /// Inter-Graph edges must start from a copy node
    #[error("Inter-graph edges must start from a copy node. Found operation {from_optype:?}. In an inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    NonCopySource {
        from: NodeIndex,
        from_offset: PortOffset,
        from_optype: OpType,
        to: NodeIndex,
        to_offset: PortOffset,
    },
    /// The grandparent of a dominator inter-graph edge must be a CFG container.
    #[error("The grandparent of a dominator inter-graph edge must be a CFG container. Found operation {ancestor_parent_op:?}. In a dominator inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    NonCFGAncestor {
        from: NodeIndex,
        from_offset: PortOffset,
        to: NodeIndex,
        to_offset: PortOffset,
        ancestor_parent_op: OpType,
    },
    /// The sibling ancestors of the external inter-graph edge endpoints must be have an order edge between them.
    #[error("Missing state order between the external inter-graph source {from:?} and the ancestor of the target {to_ancestor:?}. In an external inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    MissingOrderEdge {
        from: NodeIndex,
        from_offset: PortOffset,
        to: NodeIndex,
        to_offset: PortOffset,
        to_ancestor: NodeIndex,
    },
    /// The ancestors of an inter-graph edge are not related.
    #[error("The ancestors of an inter-graph edge are not related. In an inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    NoRelation {
        from: NodeIndex,
        from_offset: PortOffset,
        to: NodeIndex,
        to_offset: PortOffset,
    },
    /// The basic block containing the source node does not dominate the basic block containing the target node.
    #[error(" The basic block containing the source node does not dominate the basic block containing the target node in the CFG. Expected node {from_parent:?} to dominate {ancestor:?}. In a dominator inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    NonDominatedAncestor {
        from: NodeIndex,
        from_offset: PortOffset,
        to: NodeIndex,
        to_offset: PortOffset,
        from_parent: NodeIndex,
        ancestor: NodeIndex,
    },
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use super::*;
    use crate::hugr::HugrMut;
    use crate::ops::{ModuleOp, OpType};
    use crate::type_row;
    use crate::types::{ClassicType, Signature};

    #[test]
    fn invalid_root() {
        let declare_op: OpType = ModuleOp::Declare {
            signature: Default::default(),
        }
        .into();

        let mut b = HugrMut::new();
        let root = b.root();
        assert_eq!(b.hugr().validate(), Ok(()));

        // Add another hierarchy root
        let other = b.add_op(ModuleOp::Root);
        assert_matches!(
            b.hugr().validate(),
            Err(ValidationError::NoParent { node }) => assert_eq!(node, other)
        );
        b.set_parent(other, root).unwrap();
        b.replace_op(other, declare_op.clone());
        assert_eq!(b.hugr().validate(), Ok(()));

        // Change the root type
        b.replace_op(root, declare_op);
        assert_matches!(
            b.hugr().validate(),
            Err(ValidationError::InvalidRootOpType { node, .. }) => assert_eq!(node, root)
        );

        // Make the hugr root not a hierarchy root
        {
            let mut hugr = b.hugr().clone();
            hugr.root = other;
            assert_matches!(
                hugr.validate(),
                Err(ValidationError::RootNotRoot { node }) => assert_eq!(node, other)
            );
        }
    }

    /// Creates a hugr with a single function definition that copies a bit.
    fn make_simple_hugr() -> HugrMut {
        const B: SimpleType = SimpleType::Classic(ClassicType::bit());

        let def_op: OpType = ModuleOp::Def {
            signature: Signature::new_df(type_row![B], type_row![B, B]),
        }
        .into();

        let mut b = HugrMut::new();
        let root = b.root();

        let def = b.add_op_with_parent(root, def_op).unwrap();
        let input = b
            .add_op_with_parent(
                def,
                DataflowOp::Input {
                    types: type_row![B],
                },
            )
            .unwrap();
        let copy = b
            .add_op_with_parent(
                def,
                LeafOp::Copy {
                    n_copies: 2,
                    typ: ClassicType::bit(),
                },
            )
            .unwrap();
        let output = b
            .add_op_with_parent(
                def,
                DataflowOp::Output {
                    types: type_row![B, B],
                },
            )
            .unwrap();

        b.connect(input, 0, copy, 0).unwrap();
        b.connect(copy, 0, output, 0).unwrap();
        b.connect(copy, 1, output, 1).unwrap();

        b
    }

    #[test]
    fn simple_hugr() {
        let b = make_simple_hugr();
        assert_eq!(b.hugr().validate(), Ok(()));
    }

    #[test]
    fn invalid_ports() {
        let mut b = make_simple_hugr();
        let root = b.root();
        let def = b.hugr().hierarchy.first(root).unwrap();
        let (_input, copy, output) = b.hugr().hierarchy.children(def).collect_tuple().unwrap();

        // Missing an output port
        b.replace_op(
            copy,
            LeafOp::Copy {
                n_copies: 3,
                typ: ClassicType::bit(),
            },
        );
        assert_matches!(
            b.hugr().validate(),
            Err(ValidationError::WrongNumberOfPorts { node, .. }) => assert_eq!(node, copy)
        );

        // Allocate that port, it remains unconnected
        b.set_num_ports(copy, 1, 3);
        assert_matches!(
            b.hugr().validate(),
            Err(ValidationError::UnconnectedPort { node, .. }) => assert_eq!(node, copy)
        );

        // Make the 2nd copy output become an order edge, mismatching the output port
        b.set_num_ports(copy, 1, 2);
        b.replace_op(
            copy,
            LeafOp::Copy {
                n_copies: 1,
                typ: ClassicType::bit(),
            },
        );
        assert_matches!(
            b.hugr().validate(),
            Err(ValidationError::IncompatiblePorts { from, to, .. }) => {assert_eq!(from, copy); assert_eq!(to, output)}
        );
    }

    #[test]
    fn children_restrictions() {
        // TODO
        // InvalidParent
        // InvalidBoundaryChild
        // InvalidChildren (all of the variants)
        // NonContainerWithChildren
        // ContainerWithoutChildren
    }

    #[test]
    fn dags() {
        // TODO
        // NotADag
    }

    #[test]
    fn intergraph_edges() {
        // TODO
        // InterGraphEdge (all of the variants)
    }
}
