//! HUGR invariant checks.

use std::collections::HashMap;
use std::iter;

use itertools::{Either, Itertools};
use petgraph::algo::dominators::{self, Dominators};
use petgraph::visit::{Topo, Walker};
use portgraph::{LinkView, PortView};
use thiserror::Error;

#[cfg(feature = "pyo3")]
use pyo3::{create_exception, exceptions::PyException, PyErr};

use crate::extension::SignatureError;
use crate::extension::{
    validate::{ExtensionError, ExtensionValidator},
    ExtensionRegistry, ExtensionSolution, InferExtensionError,
};

use crate::ops::custom::CustomOpError;
use crate::ops::custom::{resolve_opaque_op, ExtensionOp, ExternalOp};
use crate::ops::validate::{ChildrenEdgeData, ChildrenValidationError, EdgeValidationError};
use crate::ops::{FuncDecl, FuncDefn, OpTag, OpTrait, OpType, ValidateOp};
use crate::types::type_param::TypeParam;
use crate::types::{EdgeKind, Type};
use crate::{Direction, Hugr, Node, Port};

use super::views::{HierarchyView, HugrView, SiblingGraph};
use super::NodeType;

/// Structure keeping track of pre-computed information used in the validation
/// process.
///
/// TODO: Consider implementing updatable dominator trees and storing it in the
/// Hugr to avoid recomputing it every time.
struct ValidationContext<'a, 'b> {
    hugr: &'a Hugr,
    /// Dominator tree for each CFG region, using the container node as index.
    dominators: HashMap<Node, Dominators<Node>>,
    /// Context for the extension validation.
    extension_validator: ExtensionValidator,
    /// Registry of available Extensions
    extension_registry: &'b ExtensionRegistry,
}

impl Hugr {
    /// Check the validity of the HUGR, assuming that it has no open extension
    /// variables.
    /// TODO: Add a version of validation which allows for open extension
    /// variables (see github issue #457)
    pub fn validate(&self, extension_registry: &ExtensionRegistry) -> Result<(), ValidationError> {
        self.validate_with_extension_closure(HashMap::new(), extension_registry)
    }

    /// Check the validity of a hugr, taking an argument of a closure for the
    /// free extension variables
    pub fn validate_with_extension_closure(
        &self,
        closure: ExtensionSolution,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), ValidationError> {
        let mut validator = ValidationContext::new(self, closure, extension_registry);
        validator.validate()
    }
}

impl<'a, 'b> ValidationContext<'a, 'b> {
    /// Create a new validation context.
    pub fn new(
        hugr: &'a Hugr,
        extension_closure: ExtensionSolution,
        extension_registry: &'b ExtensionRegistry,
    ) -> Self {
        Self {
            hugr,
            dominators: HashMap::new(),
            extension_validator: ExtensionValidator::new(hugr, extension_closure),
            extension_registry,
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

        // Hierarchy and children. No type variables declared outside the root.
        self.validate_subtree(self.hugr.root(), &[])?;

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
        let node_type = self.hugr.get_nodetype(node);
        let op_type = &node_type.op;

        // The Hugr can have only one root node.
        if node == self.hugr.root() {
            // The root node has no edges.
            if self.hugr.graph.num_outputs(node.pg_index())
                + self.hugr.graph.num_inputs(node.pg_index())
                != 0
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
        }

        // Secondly that the node has correct children
        self.validate_children(node, node_type)?;

        // FuncDefns have no resources since they're static nodes, but the
        // functions they define can have any extension delta.
        if node_type.tag() != OpTag::FuncDefn {
            // If this is a container with I/O nodes, check that the extension they
            // define match the extensions of the container.
            if let Some([input, output]) = self.hugr.get_io(node) {
                self.extension_validator
                    .validate_io_extensions(node, input, output)?;
            }
        }

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
        let must_be_connected = match dir {
            // Incoming ports must be connected, except for state order ports, branch case nodes,
            // and CFG nodes.
            Direction::Incoming => {
                port_kind != EdgeKind::StateOrder
                    && port_kind != EdgeKind::ControlFlow
                    && op_type.tag() != OpTag::Case
            }
            // Linear dataflow values must be connected.
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

        match &port_kind {
            EdgeKind::Value(ty) | EdgeKind::Static(ty) => ty
                .validate(self.extension_registry, var_decls)
                .map_err(|cause| ValidationError::SignatureError { node, cause })?,
            _ => (),
        }

        let mut link_cnt = 0;
        for (_, link) in links {
            link_cnt += 1;
            if port_kind.is_linear() && link_cnt > 1 {
                return Err(ValidationError::TooManyConnections {
                    node,
                    port,
                    port_kind,
                });
            }

            let other_node: Node = self.hugr.graph.port_node(link).unwrap().into();
            let other_offset = self.hugr.graph.port_offset(link).unwrap().into();

            self.extension_validator
                .check_extensions_compatible(&(node, port), &(other_node, other_offset))?;

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

    /// Check operation-specific constraints.
    ///
    /// These are flags defined for each operation type as an [`OpValidityFlags`] object.
    fn validate_children(&self, node: Node, node_type: &NodeType) -> Result<(), ValidationError> {
        let op_type = &node_type.op;
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
    ) -> Result<(), ValidationError> {
        let from_parent = self
            .hugr
            .get_parent(from)
            .expect("Root nodes cannot have ports");
        let to_parent = self.hugr.get_parent(to);
        let local = Some(from_parent) == to_parent;

        let is_static = match from_optype.port_kind(from_offset).unwrap() {
            EdgeKind::Static(typ) => {
                if !(OpTag::Const.is_superset(from_optype.tag())
                    || OpTag::Function.is_superset(from_optype.tag()))
                {
                    return Err(InterGraphEdgeError::InvalidConstSrc {
                        from,
                        from_offset,
                        typ,
                    }
                    .into());
                };
                true
            }
            ty => {
                if !local && !matches!(&ty, EdgeKind::Value(t) if t.copyable()) {
                    return Err(InterGraphEdgeError::NonCopyableData {
                        from,
                        from_offset,
                        to,
                        to_offset,
                        ty,
                    }
                    .into());
                }
                false
            }
        };
        if local {
            return Ok(());
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
                // External edge.
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
                    }
                    .into());
                }

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

    fn validate_subtree(
        &mut self,
        node: Node,
        var_decls: &[TypeParam],
    ) -> Result<(), ValidationError> {
        let op_type = self.hugr.get_optype(node);
        // The op_type must be defined only in terms of type variables defined outside the node
        // TODO consider turning this match into a trait method?
        match op_type {
            OpType::LeafOp(crate::ops::LeafOp::CustomOp(b)) => {
                // Try to resolve serialized names to actual OpDefs in Extensions.
                let e: Option<ExtensionOp>;
                let ext_or_opaq = match &**b {
                    ExternalOp::Opaque(op) => {
                        // If resolve_extension_ops has been called first, this would always return Ok(None)
                        e = resolve_opaque_op(node, op, self.extension_registry)?;
                        e.as_ref().map(Either::Left).unwrap_or(Either::Right(op))
                    }
                    ExternalOp::Extension(ext) => Either::Left(ext),
                };
                // If successful, check TypeArgs are valid for the declared TypeParams
                match ext_or_opaq {
                    Either::Left(exten) => exten
                        .def()
                        .validate_args(exten.args(), self.extension_registry, var_decls)
                        .map_err(|cause| ValidationError::SignatureError { node, cause })?,
                    Either::Right(opaq) => {
                        // Best effort. Just check TypeArgs are valid (in themselves, not necessarily wrt the TypeParams)
                        // and assuming none are binary params (all may contain type vars)
                        for arg in opaq.args() {
                            arg.validate(self.extension_registry, var_decls)
                                .map_err(|cause| ValidationError::SignatureError { node, cause })?;
                        }
                    }
                }
            }
            OpType::LeafOp(crate::ops::LeafOp::TypeApply { ta }) => {
                ta.validate(self.extension_registry)
                    .map_err(|cause| ValidationError::SignatureError { node, cause })?;
            }
            _ => (),
        }

        // Check port connections.
        for dir in Direction::BOTH {
            for (i, port_index) in self.hugr.graph.ports(node.pg_index(), dir).enumerate() {
                let port = Port::new(dir, i);
                self.validate_port(node, port, port_index, op_type, var_decls)?;
            }
        }

        // Compute cumulative TypeParams for nested FuncDefns, i.e. allow inner FuncDefns
        // to refer to type parameters from outer FuncDefns.
        let mut v: Vec<TypeParam>;
        let var_decls = if let OpType::FuncDefn(FuncDefn { signature, .. })
        | OpType::FuncDecl(FuncDecl { signature, .. }) = op_type
        {
            v = signature.params().to_owned();
            v.extend(var_decls.iter().cloned());
            v.as_ref()
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
    /// The children of a node do not form a DAG.
    #[error("The children of an operation {optype:?} must form a DAG. Loops are not allowed. In node {node:?}.")]
    NotADag { node: Node, optype: OpType },
    /// There are invalid inter-graph edges.
    #[error(transparent)]
    InterGraphEdgeError(#[from] InterGraphEdgeError),
    /// There are errors in the extension declarations.
    #[error(transparent)]
    ExtensionError(#[from] ExtensionError),
    #[error(transparent)]
    CantInfer(#[from] InferExtensionError),
    /// Error in a node signature
    #[error("Error in signature of node {node:?}: {cause}")]
    SignatureError { node: Node, cause: SignatureError },
    /// Error in a [CustomOp] serialized as an [Opaque]
    ///
    /// [CustomOp]: crate::ops::LeafOp::CustomOp
    /// [Opaque]: crate::ops::custom::ExternalOp::Opaque
    #[error(transparent)]
    CustomOpError(#[from] CustomOpError),
}

#[cfg(feature = "pyo3")]
create_exception!(
    pyrs,
    PyValidationError,
    PyException,
    "Errors that can occur while validating a Hugr"
);

#[cfg(feature = "pyo3")]
impl From<ValidationError> for PyErr {
    fn from(err: ValidationError) -> Self {
        PyValidationError::new_err(err.to_string())
    }
}

/// Errors related to the inter-graph edge validations.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum InterGraphEdgeError {
    /// Inter-Graph edges can only carry copyable data.
    #[error("Inter-graph edges can only carry copyable data. In an inter-graph edge from {from:?} ({from_offset:?}) to {to:?} ({to_offset:?}) with type {ty:?}.")]
    NonCopyableData {
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
        typ: Type,
    },
}

#[cfg(test)]
mod test;
