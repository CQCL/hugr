//! HUGR invariant checks.

use std::collections::HashMap;
use std::iter;

use itertools::Itertools;
use petgraph::algo::dominators::{self, Dominators};
use petgraph::visit::{Topo, Walker};
use portgraph::{LinkView, PortView};
use thiserror::Error;

use crate::extension::{SignatureError, TO_BE_INFERRED};

use crate::ops::constant::ConstTypeError;
use crate::ops::custom::{ExtensionOp, OpaqueOpError};
use crate::ops::validate::{ChildrenEdgeData, ChildrenValidationError, EdgeValidationError};
use crate::ops::{FuncDefn, NamedOp, OpName, OpParent, OpTag, OpTrait, OpType, ValidateOp};
use crate::types::type_param::TypeParam;
use crate::types::EdgeKind;
use crate::{Direction, Hugr, Node, Port};

use super::views::{HierarchyView, HugrView, SiblingGraph};
use super::ExtensionError;

/// Structure keeping track of pre-computed information used in the validation
/// process.
///
/// TODO: Consider implementing updatable dominator trees and storing it in the
/// Hugr to avoid recomputing it every time.
struct ValidationContext<'a> {
    hugr: &'a Hugr,
    /// Dominator tree for each CFG region, using the container node as index.
    dominators: HashMap<Node, Dominators<Node>>,
}

impl Hugr {
    /// Check the validity of the HUGR, assuming that it has no open extension
    /// variables.
    /// TODO: Add a version of validation which allows for open extension
    /// variables (see github issue #457)
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.validate_no_extensions()?;
        if cfg!(feature = "extension_inference") {
            self.validate_extensions()?;
        }
        Ok(())
    }

    /// Check the validity of the HUGR, but don't check consistency of extension
    /// requirements between connected nodes or between parents and children.
    pub fn validate_no_extensions(&self) -> Result<(), ValidationError> {
        let mut validator = ValidationContext::new(self);
        validator.validate()
    }

    /// Validate extensions, i.e. that extension deltas from parent nodes are reflected in their children.
    pub fn validate_extensions(&self) -> Result<(), ValidationError> {
        for parent in self.nodes() {
            let parent_op = self.get_optype(parent);
            if parent_op.extension_delta().contains(&TO_BE_INFERRED) {
                return Err(ValidationError::ExtensionsNotInferred { node: parent });
            }
            let parent_extensions = match parent_op.inner_function_type() {
                Some(s) => s.runtime_reqs.clone(),
                None => match parent_op.tag() {
                    OpTag::Cfg | OpTag::Conditional => parent_op.extension_delta(),
                    // ModuleRoot holds but does not execute its children, so allow any extensions
                    OpTag::ModuleRoot => continue,
                    _ => {
                        assert!(self.children(parent).next().is_none(),
                            "Unknown parent node type {} - not a DataflowParent, Module, Cfg or Conditional",
                            parent_op);
                        continue;
                    }
                },
            };
            for child in self.children(parent) {
                let child_extensions = self.get_optype(child).extension_delta();
                if !parent_extensions.is_superset(&child_extensions) {
                    return Err(ExtensionError {
                        parent,
                        parent_extensions,
                        child,
                        child_extensions,
                    }
                    .into());
                }
            }
        }
        Ok(())
    }
}

impl<'a> ValidationContext<'a> {
    /// Create a new validation context.
    // Allow unused "extension_closure" variable for when
    // the "extension_inference" feature is disabled.
    #[allow(unused_variables)]
    pub fn new(hugr: &'a Hugr) -> Self {
        let dominators = HashMap::new();
        Self { hugr, dominators }
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

        // Hierarchy and children. No type variables declared outside the root.
        self.validate_subtree(self.hugr.root(), &[])?;

        // In tests we take the opportunity to verify that the hugr
        // serialization round-trips. We verify the schema of the serialization
        // format only when an environment variable is set. This allows
        // a developer to modify the definition of serialized types locally
        // without having to change the schema.
        #[cfg(all(test, not(miri)))]
        {
            let test_schema = std::env::var("HUGR_TEST_SCHEMA").is_ok_and(|x| !x.is_empty());
            crate::hugr::serialize::test::check_hugr_roundtrip(self.hugr, test_schema);
        }

        Ok(())
    }

    /// Compute the dominator tree for a CFG region, identified by its container
    /// node.
    ///
    /// The results of this computation should be cached in `self.dominators`.
    /// We don't do it here to avoid mutable borrows.
    fn compute_dominator(&self, parent: Node) -> Dominators<Node> {
        let region: SiblingGraph = SiblingGraph::try_new(self.hugr, parent).unwrap();
        let entry_node = self.hugr.children(parent).next().unwrap();
        dominators::simple_fast(&region.as_petgraph(), entry_node)
    }

    /// Check the constraints on a single node.
    ///
    /// This includes:
    /// - Matching the number of ports with the signature
    /// - Dataflow ports are correct. See `validate_df_port`
    fn validate_node(&self, node: Node) -> Result<(), ValidationError> {
        let op_type = self.hugr.get_optype(node);

        if let OpType::OpaqueOp(opaque) = op_type {
            Err(OpaqueOpError::UnresolvedOp(
                node,
                opaque.op_name().clone(),
                opaque.extension().clone(),
            ))?;
        }
        // The Hugr can have only one root node.

        for dir in Direction::BOTH {
            // Check that we have the correct amount of ports and edges.
            let num_ports = self.hugr.graph.num_ports(node.pg_index(), dir);
            if num_ports != op_type.port_count(dir) {
                return Err(ValidationError::WrongNumberOfPorts {
                    node,
                    optype: op_type.clone(),
                    actual: num_ports,
                    expected: op_type.port_count(dir),
                    dir,
                });
            }
        }

        if node == self.hugr.root() {
            // The root node cannot have connected edges
            if self.hugr.all_linked_inputs(node).next().is_some()
                || self.hugr.all_linked_outputs(node).next().is_some()
            {
                return Err(ValidationError::RootWithEdges { node });
            }
        } else {
            let Some(parent) = self.hugr.get_parent(node) else {
                return Err(ValidationError::NoParent { node });
            };

            let parent_optype = self.hugr.get_optype(parent);
            let allowed_children = parent_optype.validity_flags().allowed_children;
            if !allowed_children.is_superset(op_type.tag()) {
                return Err(ValidationError::InvalidParentOp {
                    child: node,
                    child_optype: op_type.clone(),
                    parent,
                    parent_optype: parent_optype.clone(),
                    allowed_children,
                });
            }
        }

        // Thirdly that the node has correct children
        self.validate_children(node, op_type)?;

        // OpType-specific checks.
        // TODO Maybe we should delegate these checks to the OpTypes themselves.
        if let OpType::Const(c) = op_type {
            c.validate()?;
        };

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
        op_type: &OpType,
        var_decls: &[TypeParam],
    ) -> Result<(), ValidationError> {
        let port_kind = op_type.port_kind(port).unwrap();
        let dir = port.direction();

        let mut links = self.hugr.graph.port_links(port_index).peekable();
        // Linear dataflow values must be used, and control must have somewhere to flow.
        let outgoing_is_linear = port_kind.is_linear() || port_kind == EdgeKind::ControlFlow;
        let must_be_connected = match dir {
            // Incoming ports must be connected, except for state order ports, branch case nodes,
            // and CFG nodes.
            Direction::Incoming => {
                port_kind != EdgeKind::StateOrder
                    && port_kind != EdgeKind::ControlFlow
                    && op_type.tag() != OpTag::Case
            }
            Direction::Outgoing => outgoing_is_linear,
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

        self.validate_port_kind(&port_kind, var_decls)
            .map_err(|cause| ValidationError::SignatureError {
                node,
                op: op_type.name(),
                cause,
            })?;

        let mut link_cnt = 0;
        for (_, link) in links {
            link_cnt += 1;
            if outgoing_is_linear && link_cnt > 1 {
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
                panic!("The number of ports in {other_node} does not match the operation definition. This should have been caught by `validate_node`.");
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

            self.validate_edge(node, port, op_type, other_node, other_offset)?;
        }

        Ok(())
    }

    fn validate_port_kind(
        &self,
        port_kind: &EdgeKind,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        match &port_kind {
            EdgeKind::Value(ty) => ty.validate(self.hugr.extensions(), var_decls),
            // Static edges must *not* refer to type variables declared by enclosing FuncDefns
            // as these are only types at runtime.
            EdgeKind::Const(ty) => ty.validate(self.hugr.extensions(), &[]),
            EdgeKind::Function(pf) => pf.validate(self.hugr.extensions()),
            _ => Ok(()),
        }
    }

    /// Check operation-specific constraints.
    ///
    /// These are flags defined for each operation type as an [`OpValidityFlags`] object.
    fn validate_children(&self, node: Node, op_type: &OpType) -> Result<(), ValidationError> {
        let flags = op_type.validity_flags();

        if self.hugr.hierarchy.child_count(node.pg_index()) > 0 {
            if flags.allowed_children.is_empty() {
                return Err(ValidationError::NonContainerWithChildren {
                    node,
                    optype: op_type.clone(),
                });
            }

            let all_children = self.hugr.children(node);
            let mut first_two_children = all_children.clone().take(2);
            let first_child = self.hugr.get_optype(first_two_children.next().unwrap());
            if !flags.allowed_first_child.is_superset(first_child.tag()) {
                return Err(ValidationError::InvalidInitialChild {
                    parent: node,
                    parent_optype: op_type.clone(),
                    optype: first_child.clone(),
                    expected: flags.allowed_first_child,
                    position: "first",
                });
            }

            if let Some(second_child) = first_two_children
                .next()
                .map(|child| self.hugr.get_optype(child))
            {
                if !flags.allowed_second_child.is_superset(second_child.tag()) {
                    return Err(ValidationError::InvalidInitialChild {
                        parent: node,
                        parent_optype: op_type.clone(),
                        optype: second_child.clone(),
                        expected: flags.allowed_second_child,
                        position: "second",
                    });
                }
            }
            // Additional validations running over the full list of children optypes
            let children_optypes = all_children.map(|c| (c.pg_index(), self.hugr.get_optype(c)));
            if let Err(source) = op_type.validate_op_children(children_optypes) {
                return Err(ValidationError::InvalidChildren {
                    parent: node,
                    parent_optype: op_type.clone(),
                    source,
                });
            }

            // Additional validations running over the edges of the contained graph
            if let Some(edge_check) = flags.edge_check {
                for source in self.hugr.hierarchy.children(node.pg_index()) {
                    for target in self.hugr.graph.output_neighbours(source) {
                        if self.hugr.hierarchy.parent(target) != Some(node.pg_index()) {
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
                                    parent_optype: op_type.clone(),
                                    source,
                                });
                            }
                        }
                    }
                }
            }

            if flags.requires_dag {
                self.validate_children_dag(node, op_type)?;
            }
        } else if flags.requires_children {
            return Err(ValidationError::ContainerWithoutChildren {
                node,
                optype: op_type.clone(),
            });
        }

        Ok(())
    }

    /// Ensure that the children of a node form a directed acyclic graph.
    ///
    /// Inter-graph edges are ignored. Only internal dataflow, constant, or
    /// state order edges are considered.
    fn validate_children_dag(&self, parent: Node, op_type: &OpType) -> Result<(), ValidationError> {
        if !self.hugr.hierarchy.has_children(parent.pg_index()) {
            // No children, nothing to do
            return Ok(());
        };

        let region: SiblingGraph = SiblingGraph::try_new(self.hugr, parent).unwrap();
        let postorder = Topo::new(&region.as_petgraph());
        let nodes_visited = postorder
            .iter(&region.as_petgraph())
            .filter(|n| *n != parent)
            .count();
        let node_count = self.hugr.children(parent).count();
        if nodes_visited != node_count {
            return Err(ValidationError::NotADag {
                node: parent,
                optype: op_type.clone(),
            });
        }

        Ok(())
    }

    /// Check the edge is valid, i.e. the source/target nodes are at appropriate
    /// positions in the hierarchy for some locality:
    /// - Local edges, of any kind;
    /// - External edges, for static and value edges only: from a node to a sibling's descendant.
    ///   For Value edges, there must also be an order edge between the copy and the sibling.
    /// - Dominator edges, for value edges only: from a node in a BasicBlock node to
    ///   a descendant of a post-dominating sibling of the BasicBlock.
    fn validate_edge(
        &mut self,
        from: Node,
        from_offset: Port,
        from_optype: &OpType,
        to: Node,
        to_offset: Port,
    ) -> Result<(), InterGraphEdgeError> {
        let from_parent = self
            .hugr
            .get_parent(from)
            .expect("Root nodes cannot have ports");
        let to_parent = self.hugr.get_parent(to);
        let edge_kind = from_optype.port_kind(from_offset).unwrap();
        if Some(from_parent) == to_parent {
            return Ok(()); // Local edge
        }
        let is_static = edge_kind.is_static();
        if !is_static && !matches!(&edge_kind, EdgeKind::Value(t) if t.copyable()) {
            return Err(InterGraphEdgeError::NonCopyableData {
                from,
                from_offset,
                to,
                to_offset,
                ty: edge_kind,
            });
        };

        // To detect either external or dominator edges, we traverse the ancestors
        // of the target until we find either `from_parent` (in the external
        // case), or the parent of `from_parent` (in the dominator case).
        //
        // This search could be sped-up with a pre-computed LCA structure, but
        // for valid Hugrs this search should be very short.
        //
        // For Value edges only, we record any FuncDefn we went through; if there is
        // any such, then that is an error, but we report that only if the dom/ext
        // relation was otherwise ok (an error about an edge "entering" some ancestor
        // node could be misleading if the source isn't where it's expected)
        let mut err_entered_func = None;
        let from_parent_parent = self.hugr.get_parent(from_parent);
        for (ancestor, ancestor_parent) in
            iter::successors(to_parent, |&p| self.hugr.get_parent(p)).tuple_windows()
        {
            if !is_static && self.hugr.get_optype(ancestor).is_func_defn() {
                err_entered_func.get_or_insert(InterGraphEdgeError::ValueEdgeIntoFunc {
                    to,
                    to_offset,
                    from,
                    from_offset,
                    func: ancestor,
                });
            }
            if ancestor_parent == from_parent {
                // External edge.
                err_entered_func.map_or(Ok(()), Err)?;
                if !is_static {
                    // Must have an order edge.
                    self.hugr
                        .graph
                        .get_connections(from.pg_index(), ancestor.pg_index())
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
                }
                return Ok(());
            } else if Some(ancestor_parent) == from_parent_parent && !is_static {
                // Dominator edge
                let ancestor_parent_op = self.hugr.get_optype(ancestor_parent);
                if ancestor_parent_op.tag() != OpTag::Cfg {
                    return Err(InterGraphEdgeError::NonCFGAncestor {
                        from,
                        from_offset,
                        to,
                        to_offset,
                        ancestor_parent_op: ancestor_parent_op.clone(),
                    });
                }
                err_entered_func.map_or(Ok(()), Err)?;
                // Check domination
                let dominator_tree = match self.dominators.get(&ancestor_parent) {
                    Some(tree) => tree,
                    None => {
                        let tree = self.compute_dominator(ancestor_parent);
                        self.dominators.insert(ancestor_parent, tree);
                        self.dominators.get(&ancestor_parent).unwrap()
                    }
                };
                if !dominator_tree
                    .dominators(ancestor)
                    .map_or(false, |mut ds| ds.any(|n| n == from_parent))
                {
                    return Err(InterGraphEdgeError::NonDominatedAncestor {
                        from,
                        from_offset,
                        to,
                        to_offset,
                        from_parent,
                        ancestor,
                    });
                }

                return Ok(());
            }
        }

        Err(InterGraphEdgeError::NoRelation {
            from,
            from_offset,
            to,
            to_offset,
        })
    }

    /// Validates that TypeArgs are valid wrt the [ExtensionRegistry] and that nodes
    /// only refer to type variables declared by the closest enclosing FuncDefn.
    fn validate_subtree(
        &mut self,
        node: Node,
        var_decls: &[TypeParam],
    ) -> Result<(), ValidationError> {
        let op_type = self.hugr.get_optype(node);
        // The op_type must be defined only in terms of type variables defined outside the node

        let validate_ext = |ext_op: &ExtensionOp| -> Result<(), ValidationError> {
            // Check TypeArgs are valid, and if we can, fit the declared TypeParams
            ext_op
                .def()
                .validate_args(ext_op.args(), self.hugr.extensions(), var_decls)
                .map_err(|cause| ValidationError::SignatureError {
                    node,
                    op: op_type.name(),
                    cause,
                })
        };
        match op_type {
            OpType::ExtensionOp(ext_op) => validate_ext(ext_op)?,
            OpType::OpaqueOp(opaque) => {
                Err(OpaqueOpError::UnresolvedOp(
                    node,
                    opaque.op_name().clone(),
                    opaque.extension().clone(),
                ))?;
            }
            OpType::Call(c) => {
                c.validate(self.hugr.extensions()).map_err(|cause| {
                    ValidationError::SignatureError {
                        node,
                        op: op_type.name(),
                        cause,
                    }
                })?;
            }
            OpType::LoadFunction(c) => {
                c.validate(self.hugr.extensions()).map_err(|cause| {
                    ValidationError::SignatureError {
                        node,
                        op: op_type.name(),
                        cause,
                    }
                })?;
            }
            _ => (),
        }

        // Check port connections.
        //
        // Root nodes are ignored, as they cannot have connected edges.
        if node != self.hugr.root() {
            for dir in Direction::BOTH {
                for (i, port_index) in self.hugr.graph.ports(node.pg_index(), dir).enumerate() {
                    let port = Port::new(dir, i);
                    self.validate_port(node, port, port_index, op_type, var_decls)?;
                }
            }
        }

        // For FuncDefn's, only the type variables declared by the FuncDefn can be referred to by nodes
        // inside the function. (The same would be true for FuncDecl's, but they have no child nodes.)
        let var_decls = if let OpType::FuncDefn(FuncDefn { signature, .. }) = op_type {
            signature.params()
        } else {
            var_decls
        };

        for child in self.hugr.children(node) {
            self.validate_subtree(child, var_decls)?;
        }

        Ok(())
    }
}

/// Errors that can occur while validating a Hugr.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum ValidationError {
    /// The root node of the Hugr is not a root in the hierarchy.
    #[error("The root node of the Hugr {node} is not a root in the hierarchy.")]
    RootNotRoot { node: Node },
    /// The root node of the Hugr should not have any edges.
    #[error("The root node of the Hugr {node} has edges when it should not.")]
    RootWithEdges { node: Node },
    /// The node ports do not match the operation signature.
    #[error("The node {node} has an invalid number of ports. The operation {optype} cannot have {actual} {dir:?} ports. Expected {expected}.")]
    WrongNumberOfPorts {
        node: Node,
        optype: OpType,
        actual: usize,
        expected: usize,
        dir: Direction,
    },
    /// A dataflow port is not connected.
    #[error("The node {node} has an unconnected port {port} of type {port_kind}.")]
    UnconnectedPort {
        node: Node,
        port: Port,
        port_kind: EdgeKind,
    },
    /// A linear port is connected to more than one thing.
    #[error(
        "The node {node} has a port {port} of type {port_kind} with more than one connection."
    )]
    TooManyConnections {
        node: Node,
        port: Port,
        port_kind: EdgeKind,
    },
    /// Connected ports have different types, or non-unifiable types.
    #[error("Connected ports {from_port} in node {from} and {to_port} in node {to} have incompatible kinds. Cannot connect {from_kind} to {to_kind}.")]
    IncompatiblePorts {
        from: Node,
        from_port: Port,
        from_kind: EdgeKind,
        to: Node,
        to_port: Port,
        to_kind: EdgeKind,
    },
    /// The non-root node has no parent.
    #[error("The node {node} has no parent.")]
    NoParent { node: Node },
    /// The parent node is not compatible with the child node.
    #[error("The operation {parent_optype} cannot contain a {child_optype} as a child. Allowed children: {}. In node {child} with parent {parent}.", allowed_children.description())]
    InvalidParentOp {
        child: Node,
        child_optype: OpType,
        parent: Node,
        parent_optype: OpType,
        allowed_children: OpTag,
    },
    /// Invalid first/second child.
    #[error("A {optype} operation cannot be the {position} child of a {parent_optype}. Expected {expected}. In parent node {parent}")]
    InvalidInitialChild {
        parent: Node,
        parent_optype: OpType,
        optype: OpType,
        expected: OpTag,
        position: &'static str,
    },
    /// The children list has invalid elements.
    #[error(
        "An operation {parent_optype} contains invalid children: {source}. In parent {parent}, child Node({child})",
        child=source.child().index(),
    )]
    InvalidChildren {
        parent: Node,
        parent_optype: OpType,
        source: ChildrenValidationError,
    },
    /// The children graph has invalid edges.
    #[error(
        "An operation {parent_optype} contains invalid edges between its children: {source}. In parent {parent}, edge from {from:?} port {from_port:?} to {to:?} port {to_port:?}",
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
    #[error("The node {node} with optype {optype} is not a container, but has children.")]
    NonContainerWithChildren { node: Node, optype: OpType },
    /// The node must have children, but has none.
    #[error("The node {node} with optype {optype} must have children, but has none.")]
    ContainerWithoutChildren { node: Node, optype: OpType },
    /// The children of a node do not form a DAG.
    #[error("The children of an operation {optype} must form a DAG. Loops are not allowed. In node {node}.")]
    NotADag { node: Node, optype: OpType },
    /// There are invalid inter-graph edges.
    #[error(transparent)]
    InterGraphEdgeError(#[from] InterGraphEdgeError),
    /// There are errors in the extension deltas.
    #[error(transparent)]
    ExtensionError(#[from] ExtensionError),
    /// A node claims to still be awaiting extension inference. Perhaps it is not acted upon by inference.
    #[error("Node {node} needs a concrete ExtensionSet - inference will provide this for Case/CFG/Conditional/DataflowBlock/DFG/TailLoop only")]
    ExtensionsNotInferred { node: Node },
    /// Error in a node signature
    #[error("Error in signature of operation {op} at {node}: {cause}")]
    SignatureError {
        node: Node,
        op: OpName,
        #[source]
        cause: SignatureError,
    },
    /// Error in a [ExtensionOp] serialized as an [Opaque].
    ///
    /// [ExtensionOp]: crate::ops::ExtensionOp
    /// [Opaque]: crate::ops::OpaqueOp
    #[error(transparent)]
    OpaqueOpError(#[from] OpaqueOpError),
    /// A [Const] contained a [Value] of unexpected [Type].
    ///
    /// [Const]: crate::ops::Const
    /// [Value]: crate::ops::Value
    /// [Type]: crate::types::Type
    #[error(transparent)]
    ConstTypeError(#[from] ConstTypeError),
}

/// Errors related to the inter-graph edge validations.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum InterGraphEdgeError {
    /// Inter-Graph edges can only carry copyable data.
    #[error("Inter-graph edges can only carry copyable data. In an inter-graph edge from {from} ({from_offset}) to {to} ({to_offset}) with type {ty}.")]
    NonCopyableData {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        ty: EdgeKind,
    },
    /// Inter-Graph edges may not enter into FuncDefns unless they are static
    #[error("Inter-graph Value edges cannot enter into FuncDefns. Inter-graph edge from {from} ({from_offset}) to {to} ({to_offset} enters FuncDefn {func}")]
    ValueEdgeIntoFunc {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        func: Node,
    },
    /// The grandparent of a dominator inter-graph edge must be a CFG container.
    #[error("The grandparent of a dominator inter-graph edge must be a CFG container. Found operation {ancestor_parent_op}. In a dominator inter-graph edge from {from} ({from_offset}) to {to} ({to_offset}).")]
    NonCFGAncestor {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        ancestor_parent_op: OpType,
    },
    /// The sibling ancestors of the external inter-graph edge endpoints must be have an order edge between them.
    #[error("Missing state order between the external inter-graph source {from} and the ancestor of the target {to_ancestor}. In an external inter-graph edge from {from} ({from_offset}) to {to} ({to_offset}).")]
    MissingOrderEdge {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        to_ancestor: Node,
    },
    /// The ancestors of an inter-graph edge are not related.
    #[error("The ancestors of an inter-graph edge are not related. In an inter-graph edge from {from} ({from_offset}) to {to} ({to_offset}).")]
    NoRelation {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
    },
    /// The basic block containing the source node does not dominate the basic block containing the target node.
    #[error(" The basic block containing the source node does not dominate the basic block containing the target node in the CFG. Expected node {from_parent} to dominate {ancestor}. In a dominator inter-graph edge from {from} ({from_offset}) to {to} ({to_offset}).")]
    NonDominatedAncestor {
        from: Node,
        from_offset: Port,
        to: Node,
        to_offset: Port,
        from_parent: Node,
        ancestor: Node,
    },
}

#[cfg(test)]
pub(crate) mod test;
