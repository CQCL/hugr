//! HUGR invariant checks.

use std::collections::{HashMap, HashSet};
use std::iter;

use itertools::Itertools;
use portgraph::algorithms::{dominators_filtered, toposort_filtered, DominatorTree};
use portgraph::PortIndex;
use thiserror::Error;

use crate::hugr::typecheck::{typecheck_const, ConstTypeError};
use crate::ops::tag::OpTag;
use crate::ops::validate::{ChildrenEdgeData, ChildrenValidationError, EdgeValidationError};
use crate::ops::{self, OpTrait, OpType, ValidateOp};
use crate::types::ClassicType;
use crate::types::{EdgeKind, SimpleType};
use crate::{Direction, Hugr, Node, Port};

use super::view::HugrView;

/// Structure keeping track of pre-computed information used in the validation
/// process.
///
/// TODO: Consider implementing updatable dominator trees and storing it in the
/// Hugr to avoid recomputing it every time.
struct ValidationContext<'a> {
    hugr: &'a Hugr,
    /// Dominator tree for each CFG region, using the container node as index.
    dominators: HashMap<Node, DominatorTree>,
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
        // Root node must be a root in the hierarchy.
        if !self.hugr.hierarchy.is_root(self.hugr.root) {
            return Err(ValidationError::RootNotRoot {
                node: self.hugr.root(),
            });
        }

        // Node-specific checks
        for node in self.hugr.graph.nodes_iter().map_into() {
            self.validate_node(node)?;
        }

        Ok(())
    }

    /// Compute the dominator tree for a CFG region, identified by its container
    /// node.
    ///
    /// The results of this computation should be cached in `self.dominators`.
    /// We don't do it here to avoid mutable borrows.
    //
    // TODO: Use a `DominatorTree<HashMap>` once that's supported
    //   see https://github.com/CQCL/portgraph/issues/55
    fn compute_dominator(&self, node: Node) -> DominatorTree {
        let entry = self.hugr.hierarchy.first(node.index).unwrap();
        dominators_filtered(
            self.hugr.graph.as_portgraph(),
            entry,
            Direction::Outgoing,
            |n| {
                // We include copy nodes in addition to basic blocks.
                // These are later filtered when iterating.
                !self.hugr.graph.contains_node(n)
                    || OpTag::BasicBlock.contains(self.hugr.get_optype(n.into()).tag())
            },
            |_, _| true,
        )
    }

    /// Check the constraints on a single node.
    ///
    /// This includes:
    /// - Matching the number of ports with the signature
    /// - Dataflow ports are correct. See `validate_df_port`
    fn validate_node(&mut self, node: Node) -> Result<(), ValidationError> {
        let optype = self.hugr.get_optype(node);
        let sig = optype.signature();
        let flags = optype.validity_flags();

        // The Hugr can have only one root node.
        if node == self.hugr.root() {
            // The root node has no edges.
            if self.hugr.graph.num_outputs(node.index) + self.hugr.graph.num_inputs(node.index) != 0
            {
                return Err(ValidationError::RootWithEdges { node });
            }
        } else {
            let Some(parent) = self.hugr.get_parent(node) else {
                return Err(ValidationError::NoParent { node });
            };

            let parent_optype = self.hugr.get_optype(parent);
            let allowed_children = parent_optype.validity_flags().allowed_children;
            if !allowed_children.contains(optype.tag()) {
                return Err(ValidationError::InvalidParentOp {
                    child: node,
                    child_optype: optype.clone(),
                    parent,
                    parent_optype: parent_optype.clone(),
                    allowed_children,
                });
            }

            // Check that we have the correct amount of ports and edges.
            for dir in Direction::BOTH {
                let num_ports = self.hugr.graph.num_ports(node.index, dir);
                let has_other_ports = optype.other_port(dir).is_some();
                let expected_ports = sig.port_count(dir) + has_other_ports as usize;
                if num_ports != expected_ports {
                    return Err(ValidationError::WrongNumberOfPorts {
                        node,
                        optype: optype.clone(),
                        actual: num_ports,
                        expected: expected_ports,
                        dir,
                    });
                }

                // Check the restrictions for the exact number of other edges.
                if let Some(expected_other_edges) = flags.non_df_edge_count(dir) {
                    let actual_count = if !has_other_ports {
                        0
                    } else {
                        let port = Port::new(dir, sig.port_count(dir));
                        self.hugr.linked_ports(node, port).count()
                    };
                    if actual_count != expected_other_edges {
                        return Err(ValidationError::WrongNumberOfNonDFEdges {
                            node,
                            optype: optype.clone(),
                            actual: actual_count,
                            expected: expected_other_edges,
                            dir,
                        });
                    }
                }
            }

            // Check port connections
            for (i, port_index) in self.hugr.graph.inputs(node.index).enumerate() {
                let port = Port::new_incoming(i);
                self.validate_port(node, port, port_index, optype)?;
            }
            for (i, port_index) in self.hugr.graph.outputs(node.index).enumerate() {
                let port = Port::new_outgoing(i);
                self.validate_port(node, port, port_index, optype)?;
            }
        }

        // Check operation-specific constraints
        self.validate_operation(node, optype)?;

        Ok(())
    }

    /// Check whether a port is valid.
    /// - Input ports and output linear ports must be connected
    /// - The linked port must have a compatible type.
    fn validate_port(
        &mut self,
        node: Node,
        port: Port,
        port_index: portgraph::PortIndex,
        optype: &OpType,
    ) -> Result<(), ValidationError> {
        let port_kind = optype.port_kind(port).unwrap();
        let dir = port.direction();

        // Input ports and output linear ports must always be connected
        let mut links = self.hugr.graph.port_links(port_index).peekable();
        let must_be_connected = match dir {
            Direction::Incoming => port_kind.is_linear() || matches!(port_kind, EdgeKind::Const(_)),
            Direction::Outgoing => port_kind.is_linear(),
        };
        if must_be_connected && links.peek().is_none() {
            return Err(ValidationError::UnconnectedPort {
                node,
                port,
                port_kind,
            });
        }

        // Avoid double checking connected port types.
        if dir == Direction::Incoming {
            return Ok(());
        }

        for (subport, link) in links {
            if port_kind.is_linear() && subport.offset() != 0 {
                return Err(ValidationError::TooManyConnections {
                    node,
                    port,
                    port_kind,
                });
            }

            let other_node: Node = self.hugr.graph.port_node(link).unwrap().into();
            let other_offset = self.hugr.graph.port_offset(link).unwrap().into();
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
                    from_port: port,
                    from_kind: port_kind,
                    to: other_node,
                    to_port: other_offset,
                    to_kind: other_kind,
                });
            }

            self.validate_intergraph_edge(node, port, optype, other_node, other_offset)?;
        }

        Ok(())
    }

    /// Check operation-specific constraints.
    ///
    /// These are flags defined for each operation type as an [`OpValidityFlags`] object.
    fn validate_operation(&self, node: Node, optype: &OpType) -> Result<(), ValidationError> {
        let flags = optype.validity_flags();

        if self.hugr.hierarchy.child_count(node.index) > 0 {
            if flags.allowed_children.is_empty() {
                return Err(ValidationError::NonContainerWithChildren {
                    node,
                    optype: optype.clone(),
                });
            }

            let all_children = self.hugr.children(node);
            let mut first_two_children = all_children.clone().take(2);
            let first_child = self.hugr.get_optype(first_two_children.next().unwrap());
            if !flags.allowed_first_child.contains(first_child.tag()) {
                return Err(ValidationError::InvalidInitialChild {
                    parent: node,
                    parent_optype: optype.clone(),
                    optype: first_child.clone(),
                    expected: flags.allowed_first_child,
                    position: "first",
                });
            }

            if let Some(second_child) = first_two_children
                .next()
                .map(|child| self.hugr.get_optype(child))
            {
                if !flags.allowed_second_child.contains(second_child.tag()) {
                    return Err(ValidationError::InvalidInitialChild {
                        parent: node,
                        parent_optype: optype.clone(),
                        optype: second_child.clone(),
                        expected: flags.allowed_second_child,
                        position: "second",
                    });
                }
            }
            // Additional validations running over the full list of children optypes
            let children_optypes = all_children.map(|c| (c.index, self.hugr.get_optype(c)));
            if let Err(source) = optype.validate_children(children_optypes) {
                return Err(ValidationError::InvalidChildren {
                    parent: node,
                    parent_optype: optype.clone(),
                    source,
                });
            }

            // Additional validations running over the edges of the contained graph
            if let Some(edge_check) = flags.edge_check {
                for source in self.hugr.hierarchy.children(node.index) {
                    for target in self.hugr.graph.output_neighbours(source) {
                        if self.hugr.hierarchy.parent(target) != Some(node.index) {
                            continue;
                        }
                        let source_op = self.hugr.get_optype(source.into());
                        let target_op = self.hugr.get_optype(target.into());
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

    /// Ensure that the children of a node form a direct acyclic graph with a
    /// single source and source. That is, their edges do not form cycles in the
    /// graph and there are no dangling nodes.
    ///
    /// Inter-graph edges are ignored. Only internal dataflow, constant, or
    /// state order edges are considered.
    fn validate_children_dag(&self, parent: Node, optype: &OpType) -> Result<(), ValidationError> {
        let Some(first_child) = self.hugr.hierarchy.first(parent.index) else {
            // No children, nothing to do
            return Ok(());
        };

        // TODO: Use a HUGR-specific toposort that ignores the copy nodes,
        // so we can be more efficient and avoid the `contains_node` filter.
        // https://github.com/CQCL-DEV/hugr/issues/125
        let topo = toposort_filtered::<HashSet<PortIndex>>(
            self.hugr.graph.as_portgraph(),
            [first_child],
            Direction::Outgoing,
            |_| true,
            |n, p| self.df_port_filter(n, p),
        )
        .filter(|&node| self.hugr.graph.contains_node(node));

        // Compute the number of nodes visited.
        let nodes_visited = topo.fold(0, |n, node| {
            // If there is a LoadConstant with a local constant, count that node too
            if OpTag::LoadConst == self.hugr.get_optype(node.into()).tag() {
                let const_node = self
                    .hugr
                    .graph
                    .input_neighbours(node)
                    .next()
                    .expect("LoadConstant must be connected to a Const node.")
                    .into();
                let const_parent = self
                    .hugr
                    .get_parent(const_node)
                    .expect("Const can't be root.");

                if const_parent == parent {
                    return n + 2;
                }
            }
            n + 1
        });

        if nodes_visited != self.hugr.hierarchy.child_count(parent.index) {
            return Err(ValidationError::NotABoundedDag {
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
    /// - Dominator edges, from a copy node in a BasicBlock node to a descendant of a
    ///   post-dominated sibling of the BasicBlock.
    fn validate_intergraph_edge(
        &mut self,
        from: Node,
        from_offset: Port,
        from_optype: &OpType,
        to: Node,
        to_offset: Port,
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

        match from_optype.port_kind(from_offset).unwrap() {
            // Inter-graph constant wires do not have restrictions
            EdgeKind::Const(typ) => {
                if let OpType::Const(ops::Const(val)) = from_optype {
                    return typecheck_const(&typ, val).map_err(ValidationError::from);
                } else {
                    // If const edges aren't coming from const nodes, they're graph
                    // edges coming from Declare or Def
                    return if OpTag::Function.contains(from_optype.tag()) {
                        Ok(())
                    } else {
                        Err(InterGraphEdgeError::InvalidConstSrc {
                            from,
                            from_offset,
                            typ,
                        }
                        .into())
                    };
                }
            }
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
        let from_parent_parent = self.hugr.get_parent(from_parent);
        for (ancestor, ancestor_parent) in
            iter::successors(to_parent, |&p| self.hugr.get_parent(p)).tuple_windows()
        {
            if ancestor_parent == from_parent {
                // External edge. Must have an order edge.
                self.hugr
                    .graph
                    .get_connections(from.index, ancestor.index)
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
            } else if Some(ancestor_parent) == from_parent_parent {
                // Dominator edge
                let ancestor_parent_op = self.hugr.get_optype(ancestor_parent);
                if ancestor_parent_op.tag() == OpTag::Cfg {
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
                //
                // TODO: Use a HUGR-specific dominator that ignores the copy nodes,
                // so we can be more efficient and avoid the `contains_node` filter.
                // https://github.com/CQCL-DEV/hugr/issues/125
                let dominator_tree = match self.dominators.get(&ancestor_parent) {
                    Some(tree) => tree,
                    None => {
                        let tree = self.compute_dominator(ancestor_parent);
                        self.dominators.insert(ancestor_parent, tree);
                        self.dominators.get(&ancestor_parent).unwrap()
                    }
                };
                let mut dominators = iter::successors(Some(ancestor.index), |&n| {
                    dominator_tree.immediate_dominator(n)
                })
                .filter(|&node| self.hugr.graph.contains_node(node))
                .map_into();
                if !dominators.any(|n: Node| n == from_parent) {
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

    /// A filter function for internal dataflow edges used in the toposort algorithm.
    ///
    /// Returns `true` for ports that connect to a sibling node with a value or
    /// state order edge.
    fn df_port_filter(&self, node: portgraph::NodeIndex, port: portgraph::PortIndex) -> bool {
        // Toposort operates on the internal portgraph. It may traverse copy nodes.
        let portgraph = self.hugr.graph.as_portgraph();
        let is_copy = !self.hugr.graph.contains_node(node);
        let offset = self.hugr.graph.port_offset(port).unwrap();

        // Always follow (non-intergraph) ports from copy nodes. These nodes must be filtered out
        // when using the toposort iterator.
        if !is_copy {
            let node_optype = self.hugr.get_optype(node.into());

            let kind = node_optype.port_kind(offset).unwrap();
            if !matches!(kind, EdgeKind::StateOrder | EdgeKind::Value(_)) {
                return false;
            }
        }

        // Ignore ports that are not connected (that property is checked elsewhere)
        let Some(other_port) = portgraph
            .port_index(node, offset)
            .and_then(|p| portgraph.port_link(p))
        else {
            return false;
        };
        let other_node = portgraph.port_node(other_port).unwrap();

        // Dereference any copy nodes
        let op_node = self.hugr.graph.pg_main_node(node);
        let other_op_node = self.hugr.graph.pg_main_node(other_node);
        let parent = self.hugr.hierarchy.parent(op_node);
        let other_parent = self.hugr.hierarchy.parent(other_op_node);
        if parent != other_parent {
            return false;
        }

        true
    }
}

/// Errors that can occur while validating a Hugr.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[allow(missing_docs)]
pub enum ValidationError {
    /// The root node of the Hugr is not a root in the hierarchy.
    #[error("The root node of the Hugr {node:?} is not a root in the hierarchy.")]
    RootNotRoot { node: Node },
    /// The root node of the Hugr should not have any edges.
    #[error("The root node of the Hugr {node:?} has edges when it should not.")]
    RootWithEdges { node: Node },
    /// The node ports do not match the operation signature.
    #[error("The node {node:?} has an invalid number of ports. The operation {optype:?} cannot have {actual:?} {dir:?} ports. Expected {expected:?}.")]
    WrongNumberOfPorts {
        node: Node,
        optype: OpType,
        actual: usize,
        expected: usize,
        dir: Direction,
    },
    /// The non-dataflow multiport has the wrong number of connections.
    #[error("The node {node:?} has an invalid number of non-dataflow edges. The operation {optype:?} cannot have {actual:?} {dir:?} non-dataflow edges. Expected {expected:?}.")]
    WrongNumberOfNonDFEdges {
        node: Node,
        optype: OpType,
        actual: usize,
        expected: usize,
        dir: Direction,
    },
    /// A dataflow port is not connected.
    #[error("The node {node:?} has an unconnected port {port:?} of type {port_kind:?}.")]
    UnconnectedPort {
        node: Node,
        port: Port,
        port_kind: EdgeKind,
    },
    /// A linear port is connected to more than one thing.
    #[error("The node {node:?} has a port {port:?} of type {port_kind:?} with more than one connection.")]
    TooManyConnections {
        node: Node,
        port: Port,
        port_kind: EdgeKind,
    },
    /// Connected ports have different types, or non-unifiable types.
    #[error("Connected ports {from_port:?} in node {from:?} and {to_port:?} in node {to:?} have incompatible kinds. Cannot connect {from_kind:?} to {to_kind:?}.")]
    IncompatiblePorts {
        from: Node,
        from_port: Port,
        from_kind: EdgeKind,
        to: Node,
        to_port: Port,
        to_kind: EdgeKind,
    },
    /// The non-root node has no parent.
    #[error("The node {node:?} has no parent.")]
    NoParent { node: Node },
    /// The parent node is not compatible with the child node.
    #[error("The operation {parent_optype:?} cannot contain a {child_optype:?} as a child. Allowed children: {}. In node {child:?} with parent {parent:?}.", allowed_children.description())]
    InvalidParentOp {
        child: Node,
        child_optype: OpType,
        parent: Node,
        parent_optype: OpType,
        allowed_children: OpTag,
    },
    /// Invalid first/second child.
    #[error("A {optype:?} operation cannot be the {position} child of a {parent_optype:?}. Expected {expected}. In parent node {parent:?}")]
    InvalidInitialChild {
        parent: Node,
        parent_optype: OpType,
        optype: OpType,
        expected: OpTag,
        position: &'static str,
    },
    /// The children list has invalid elements.
    #[error(
        "An operation {parent_optype:?} contains invalid children: {source}. In parent {parent:?}, child {child:?}",
        child=source.child(),
    )]
    InvalidChildren {
        parent: Node,
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
        parent: Node,
        parent_optype: OpType,
        source: EdgeValidationError,
    },
    /// The node operation is not a container, but has children.
    #[error("The node {node:?} with optype {optype:?} is not a container, but has children.")]
    NonContainerWithChildren { node: Node, optype: OpType },
    /// The node must have children, but has none.
    #[error("The node {node:?} with optype {optype:?} must have children, but has none.")]
    ContainerWithoutChildren { node: Node, optype: OpType },
    /// The children of a node do not form a dag with single source and sink.
    #[error("The children of an operation {optype:?} must form a dag with single source and sink. Loops are not allowed, nor are dangling nodes not in the path between the input and output. In node {node:?}.")]
    NotABoundedDag { node: Node, optype: OpType },
    /// There are invalid inter-graph edges.
    #[error(transparent)]
    InterGraphEdgeError(#[from] InterGraphEdgeError),
    /// Type error for constant values
    #[error("Type error for constant value: {0}.")]
    ConstTypeError(#[from] ConstTypeError),
}

/// Errors related to the inter-graph edge validations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[allow(missing_docs)]
pub enum InterGraphEdgeError {
    /// Inter-Graph edges can only carry classical data.
    #[error("Inter-graph edges can only carry classical data. In an inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}) with type {ty:?}.")]
    NonClassicalData {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        ty: EdgeKind,
    },
    /// The grandparent of a dominator inter-graph edge must be a CFG container.
    #[error("The grandparent of a dominator inter-graph edge must be a CFG container. Found operation {ancestor_parent_op:?}. In a dominator inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    NonCFGAncestor {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        ancestor_parent_op: OpType,
    },
    /// The sibling ancestors of the external inter-graph edge endpoints must be have an order edge between them.
    #[error("Missing state order between the external inter-graph source {from:?} and the ancestor of the target {to_ancestor:?}. In an external inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    MissingOrderEdge {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        to_ancestor: Node,
    },
    /// The ancestors of an inter-graph edge are not related.
    #[error("The ancestors of an inter-graph edge are not related. In an inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    NoRelation {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
    },
    /// The basic block containing the source node does not dominate the basic block containing the target node.
    #[error(" The basic block containing the source node does not dominate the basic block containing the target node in the CFG. Expected node {from_parent:?} to dominate {ancestor:?}. In a dominator inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}).")]
    NonDominatedAncestor {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        from_parent: Node,
        ancestor: Node,
    },
    #[error(
        "Const edge comes from an invalid node type: {from:?} ({from_offset:?}). Edge type: {typ}"
    )]
    InvalidConstSrc {
        from: Node,
        from_offset: Port,
        typ: ClassicType,
    },
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use super::*;
    use crate::hugr::HugrMut;
    use crate::ops::dataflow::IOTrait;
    use crate::ops::{self, ConstValue, LeafOp, OpType};
    use crate::types::{ClassicType, LinearType, Signature};
    use crate::{type_row, Node};

    const B: SimpleType = SimpleType::Classic(ClassicType::bit());
    const Q: SimpleType = SimpleType::Linear(LinearType::Qubit);

    /// Creates a hugr with a single function definition that copies a bit `copies` times.
    ///
    /// Returns the hugr and the node index of the definition.
    fn make_simple_hugr(copies: usize) -> (Hugr, Node) {
        let def_op: OpType = ops::Def {
            name: "main".into(),
            signature: Signature::new_df(type_row![B], vec![B; copies]),
        }
        .into();

        let mut b = Hugr::default();
        let root = b.root();

        let def = b.add_op_with_parent(root, def_op).unwrap();
        let _ = add_df_children(&mut b, def, copies);

        (b, def)
    }

    /// Adds an input{B}, copy{B -> B^copies}, and output{B^copies} operation to a dataflow container.
    ///
    /// Returns the node indices of each of the operations.
    fn add_df_children(b: &mut Hugr, parent: Node, copies: usize) -> (Node, Node, Node) {
        let input = b
            .add_op_with_parent(parent, ops::Input::new(type_row![B]))
            .unwrap();
        let output = b
            .add_op_with_parent(parent, ops::Output::new(vec![B; copies].into()))
            .unwrap();
        let copy = b
            .add_op_with_parent(parent, LeafOp::Noop(ClassicType::bit().into()))
            .unwrap();

        b.connect(input, 0, copy, 0).unwrap();
        for i in 0..copies {
            b.connect(copy, 0, output, i).unwrap();
        }

        (input, copy, output)
    }

    /// Adds an input{B}, tag_constant(0, B^pred_size), tag(B^pred_size), and
    /// output{Sum{unit^pred_size}, B} operation to a dataflow container.
    /// Intended to be used to populate a BasicBlock node in a CFG.
    ///
    /// Returns the node indices of each of the operations.
    fn add_block_children(
        b: &mut Hugr,
        parent: Node,
        predicate_size: usize,
    ) -> (Node, Node, Node, Node) {
        let const_op = ops::Const(ConstValue::simple_predicate(0, predicate_size));
        let tag_type = SimpleType::Classic(ClassicType::new_simple_predicate(predicate_size));

        let input = b
            .add_op_with_parent(parent, ops::Input::new(type_row![B]))
            .unwrap();
        let output = b
            .add_op_with_parent(parent, ops::Output::new(vec![tag_type.clone(), B].into()))
            .unwrap();
        let tag_def = b.add_op_with_parent(b.root(), const_op).unwrap();
        let tag = b
            .add_op_with_parent(
                parent,
                ops::LoadConstant {
                    datatype: tag_type.try_into().unwrap(),
                },
            )
            .unwrap();

        b.connect(tag_def, 0, tag, 0).unwrap();
        b.add_other_edge(input, tag).unwrap();
        b.connect(tag, 0, output, 0).unwrap();
        b.connect(input, 0, output, 1).unwrap();

        (input, tag_def, tag, output)
    }

    #[test]
    fn invalid_root() {
        let declare_op: OpType = ops::Declare {
            name: "main".into(),
            signature: Default::default(),
        }
        .into();

        let mut b = Hugr::default();
        let root = b.root();
        assert_eq!(b.validate(), Ok(()));

        // Add another hierarchy root
        let other = b.add_op(ops::Module);
        assert_matches!(
            b.validate(),
            Err(ValidationError::NoParent { node }) => assert_eq!(node, other)
        );
        b.set_parent(other, root).unwrap();
        b.replace_op(other, declare_op);
        b.add_ports(other, Direction::Outgoing, 1);
        assert_eq!(b.validate(), Ok(()));

        // Make the hugr root not a hierarchy root
        {
            let mut hugr = b.clone();
            hugr.root = other.index;
            assert_matches!(
                hugr.validate(),
                Err(ValidationError::RootNotRoot { node }) => assert_eq!(node, other)
            );
        }
    }

    #[test]
    fn leaf_root() {
        let leaf_op: OpType = LeafOp::Noop(ClassicType::F64.into()).into();

        let b = Hugr::new(leaf_op);
        assert_eq!(b.validate(), Ok(()));
    }

    #[test]
    fn dfg_root() {
        let dfg_op: OpType = ops::DFG {
            signature: Signature::new_linear(type_row![B]),
        }
        .into();

        let mut b = Hugr::new(dfg_op);
        let root = b.root();
        add_df_children(&mut b, root, 1);
        assert_eq!(b.validate(), Ok(()));
    }

    #[test]
    fn simple_hugr() {
        let b = make_simple_hugr(2).0;
        assert_eq!(b.validate(), Ok(()));
    }

    #[test]
    /// General children restrictions.
    fn children_restrictions() {
        let (mut b, def) = make_simple_hugr(2);
        let root = b.root();
        let (_input, copy, _output) = b
            .hierarchy
            .children(def.index)
            .map_into()
            .collect_tuple()
            .unwrap();

        // Add a definition without children
        let def_sig = Signature::new_df(type_row![B], type_row![B, B]);
        let new_def = b
            .add_op_with_parent(
                root,
                ops::Def {
                    signature: def_sig,
                    name: "main".into(),
                },
            )
            .unwrap();
        assert_matches!(
            b.validate(),
            Err(ValidationError::ContainerWithoutChildren { node, .. }) => assert_eq!(node, new_def)
        );

        // Add children to the definition, but move it to be a child of the copy
        add_df_children(&mut b, new_def, 2);
        b.set_parent(new_def, copy).unwrap();
        assert_matches!(
            b.validate(),
            Err(ValidationError::NonContainerWithChildren { node, .. }) => assert_eq!(node, copy)
        );
        b.set_parent(new_def, root).unwrap();

        // After moving the previous definition to a valid place,
        // add an input node to the module subgraph
        let new_input = b
            .add_op_with_parent(root, ops::Input::new(type_row![]))
            .unwrap();
        assert_matches!(
            b.validate(),
            Err(ValidationError::InvalidParentOp { parent, child, .. }) => {assert_eq!(parent, root); assert_eq!(child, new_input)}
        );
    }

    #[test]
    /// Validation errors in a dataflow subgraph.
    fn df_children_restrictions() {
        let (mut b, def) = make_simple_hugr(2);
        let (_input, output, copy) = b
            .hierarchy
            .children(def.index)
            .map_into()
            .collect_tuple()
            .unwrap();

        // Replace the output operation of the df subgraph with a copy
        b.replace_op(output, LeafOp::Noop(ClassicType::bit().into()));
        assert_matches!(
            b.validate(),
            Err(ValidationError::InvalidInitialChild { parent, .. }) => assert_eq!(parent, def)
        );

        // Revert it back to an output, but with the wrong number of ports
        b.replace_op(output, ops::Output::new(type_row![B]));
        assert_matches!(
            b.validate(),
            Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::IOSignatureMismatch { child, .. }, .. })
                => {assert_eq!(parent, def); assert_eq!(child, output.index)}
        );
        b.replace_op(output, ops::Output::new(type_row![B, B]));

        // After fixing the output back, replace the copy with an output op
        b.replace_op(copy, ops::Output::new(type_row![B, B]));
        assert_matches!(
            b.validate(),
            Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::InternalIOChildren { child, .. }, .. })
                => {assert_eq!(parent, def); assert_eq!(child, copy.index)}
        );
    }

    #[test]
    /// Validation errors in a dataflow subgraph.
    fn cfg_children_restrictions() {
        let (mut b, def) = make_simple_hugr(1);
        let (_input, _output, copy) = b
            .hierarchy
            .children(def.index)
            .map_into()
            .collect_tuple()
            .unwrap();

        b.replace_op(
            copy,
            ops::CFG {
                inputs: type_row![B],
                outputs: type_row![B],
            },
        );
        assert_matches!(
            b.validate(),
            Err(ValidationError::ContainerWithoutChildren { .. })
        );
        let cfg = copy;

        // Construct a valid CFG, with one BasicBlock node and one exit node
        let block = b
            .add_op_with_parent(
                cfg,
                ops::BasicBlock::Block {
                    inputs: type_row![B],
                    predicate_variants: vec![type_row![]],
                    other_outputs: type_row![B],
                },
            )
            .unwrap();
        add_block_children(&mut b, block, 1);
        let exit = b
            .add_op_with_parent(
                cfg,
                ops::BasicBlock::Exit {
                    cfg_outputs: type_row![B],
                },
            )
            .unwrap();
        b.add_other_edge(block, exit).unwrap();
        assert_eq!(b.validate(), Ok(()));

        // Test malformed errors

        // Add an internal exit node
        let exit2 = b
            .add_op_after(
                exit,
                ops::BasicBlock::Exit {
                    cfg_outputs: type_row![B],
                },
            )
            .unwrap();
        assert_matches!(
            b.validate(),
            Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::InternalExitChildren { child, .. }, .. })
                => {assert_eq!(parent, cfg); assert_eq!(child, exit2.index)}
        );
        b.remove_op(exit2).unwrap();

        // Change the types in the BasicBlock node to work on qubits instead of bits
        b.replace_op(
            block,
            ops::BasicBlock::Block {
                inputs: type_row![Q],
                predicate_variants: vec![type_row![]],
                other_outputs: type_row![Q],
            },
        );
        let mut block_children = b.hierarchy.children(block.index);
        let block_input = block_children.next().unwrap().into();
        let block_output = block_children.next_back().unwrap().into();
        b.replace_op(block_input, ops::Input::new(type_row![Q]));
        b.replace_op(
            block_output,
            ops::Output::new(vec![SimpleType::new_simple_predicate(1), Q].into()),
        );
        assert_matches!(
            b.validate(),
            Err(ValidationError::InvalidEdges { parent, source: EdgeValidationError::CFGEdgeSignatureMismatch { .. }, .. })
                => assert_eq!(parent, cfg)
        );
    }
}
