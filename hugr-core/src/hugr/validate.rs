//! HUGR invariant checks.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::iter;

use itertools::Itertools;
use petgraph::algo::dominators::{self, Dominators};
use petgraph::visit::{Topo, Walker};
use thiserror::Error;

use crate::core::HugrNode;
use crate::extension::SignatureError;

use crate::ops::constant::ConstTypeError;
use crate::ops::custom::{ExtensionOp, OpaqueOpError};
use crate::ops::validate::{
    ChildrenEdgeData, ChildrenValidationError, EdgeValidationError, OpValidityFlags,
};
use crate::ops::{NamedOp, OpName, OpTag, OpTrait, OpType, ValidateOp};
use crate::types::EdgeKind;
use crate::types::type_param::TypeParam;
use crate::{Direction, Port, Visibility};

use super::internal::PortgraphNodeMap;
use super::views::HugrView;

/// Structure keeping track of pre-computed information used in the validation
/// process.
///
/// TODO: Consider implementing updatable dominator trees and storing it in the
/// Hugr to avoid recomputing it every time.
pub(super) struct ValidationContext<'a, H: HugrView> {
    hugr: &'a H,
    /// Dominator tree for each CFG region, using the container node as index.
    dominators: HashMap<H::Node, (Dominators<portgraph::NodeIndex>, H::RegionPortgraphNodes)>,
}

impl<'a, H: HugrView> ValidationContext<'a, H> {
    /// Create a new validation context.
    pub fn new(hugr: &'a H) -> Self {
        let dominators = HashMap::new();
        Self { hugr, dominators }
    }

    /// Check the validity of the HUGR.
    pub fn validate(&mut self) -> Result<(), ValidationError<H::Node>> {
        // Root node must be a root in the hierarchy.
        if self.hugr.get_parent(self.hugr.module_root()).is_some() {
            return Err(ValidationError::RootNotRoot {
                node: self.hugr.module_root(),
            });
        }

        // Node-specific checks
        for node in self.hugr.nodes().map_into() {
            self.validate_node(node)?;
        }

        // Hierarchy and children. No type variables declared outside the root.
        self.validate_subtree(self.hugr.entrypoint(), &[])?;

        self.validate_linkage()?;
        // In tests we take the opportunity to verify that the hugr
        // serialization round-trips. We verify the schema of the serialization
        // format only when an environment variable is set. This allows
        // a developer to modify the definition of serialized types locally
        // without having to change the schema.
        #[cfg(all(test, not(miri)))]
        {
            use crate::envelope::EnvelopeConfig;
            use crate::hugr::hugrmut::HugrMut;
            use crate::hugr::views::ExtractionResult;

            let (mut hugr, node_map) = self.hugr.extract_hugr(self.hugr.module_root());
            hugr.set_entrypoint(node_map.extracted_node(self.hugr.entrypoint()));
            // TODO: Currently fails when using `hugr-model`
            //crate::envelope::test::check_hugr_roundtrip(&hugr, EnvelopeConfig::binary());
            crate::envelope::test::check_hugr_roundtrip(&hugr, EnvelopeConfig::text());
        }

        Ok(())
    }

    fn validate_linkage(&self) -> Result<(), ValidationError<H::Node>> {
        // Map from func_name, for visible funcs only, to *tuple of*
        //    Node with that func_name,
        //    Signature,
        //    bool - true for FuncDefn
        let mut node_sig_defn = HashMap::new();

        for c in self.hugr.children(self.hugr.module_root()) {
            let (func_name, sig, is_defn) = match self.hugr.get_optype(c) {
                OpType::FuncDecl(fd) if fd.visibility() == &Visibility::Public => {
                    (fd.func_name(), fd.signature(), false)
                }
                OpType::FuncDefn(fd) if fd.visibility() == &Visibility::Public => {
                    (fd.func_name(), fd.signature(), true)
                }
                _ => continue,
            };
            match node_sig_defn.entry(func_name) {
                Entry::Vacant(ve) => {
                    ve.insert((c, sig, is_defn));
                }
                Entry::Occupied(oe) => {
                    // Allow two decls of the same sig (aliasing - we are allowing some laziness here).
                    // Reject if at least one Defn - either two conflicting impls,
                    //                               or Decl+Defn which should have been linked
                    let (prev_c, prev_sig, prev_defn) = oe.get();
                    if prev_sig != &sig || is_defn || *prev_defn {
                        return Err(ValidationError::DuplicateExport {
                            link_name: func_name.clone(),
                            children: [*prev_c, c],
                        });
                    };
                }
            }
        }
        Ok(())
    }

    /// Compute the dominator tree for a CFG region, identified by its container
    /// node.
    ///
    /// The results of this computation should be cached in `self.dominators`.
    /// We don't do it here to avoid mutable borrows.
    fn compute_dominator(
        &self,
        parent: H::Node,
    ) -> (Dominators<portgraph::NodeIndex>, H::RegionPortgraphNodes) {
        let (region, node_map) = self.hugr.region_portgraph(parent);
        let entry_node = self.hugr.children(parent).next().unwrap();
        let doms = dominators::simple_fast(&region, node_map.to_portgraph(entry_node));
        (doms, node_map)
    }

    /// Check the constraints on a single node.
    ///
    /// This includes:
    /// - Matching the number of ports with the signature
    /// - Dataflow ports are correct. See `validate_df_port`
    fn validate_node(&self, node: H::Node) -> Result<(), ValidationError<H::Node>> {
        let op_type = self.hugr.get_optype(node);

        if let OpType::OpaqueOp(opaque) = op_type {
            Err(OpaqueOpError::UnresolvedOp(
                node,
                opaque.unqualified_id().clone(),
                opaque.extension().clone(),
            ))?;
        }
        // The Hugr can have only one root node.

        for dir in Direction::BOTH {
            // Check that we have the correct amount of ports and edges.
            let num_ports = self.hugr.num_ports(node, dir);
            if num_ports != op_type.port_count(dir) {
                return Err(ValidationError::WrongNumberOfPorts {
                    node,
                    optype: Box::new(op_type.clone()),
                    actual: num_ports,
                    expected: op_type.port_count(dir),
                    dir,
                });
            }
        }

        if node != self.hugr.module_root() {
            let Some(parent) = self.hugr.get_parent(node) else {
                return Err(ValidationError::NoParent { node });
            };

            let parent_optype = self.hugr.get_optype(parent);
            let allowed_children = parent_optype.validity_flags::<H::Node>().allowed_children;
            if !allowed_children.is_superset(op_type.tag()) {
                return Err(ValidationError::InvalidParentOp {
                    child: node,
                    child_optype: Box::new(op_type.clone()),
                    parent,
                    parent_optype: Box::new(parent_optype.clone()),
                    allowed_children,
                });
            }
        }

        // Entrypoints must be region containers.
        if node == self.hugr.entrypoint() {
            let validity_flags: OpValidityFlags = op_type.validity_flags();
            if validity_flags.allowed_children == OpTag::None {
                return Err(ValidationError::EntrypointNotContainer {
                    node,
                    optype: Box::new(op_type.clone()),
                });
            }
        }

        // Thirdly that the node has correct children
        self.validate_children(node, op_type)?;

        // OpType-specific checks.
        // TODO Maybe we should delegate these checks to the OpTypes themselves.
        if let OpType::Const(c) = op_type {
            c.validate()?;
        }

        Ok(())
    }

    /// Check whether a port is valid.
    /// - Input ports and output linear ports must be connected
    /// - The linked port must have a compatible type.
    fn validate_port(
        &mut self,
        node: H::Node,
        port: Port,
        op_type: &OpType,
        var_decls: &[TypeParam],
    ) -> Result<(), ValidationError<H::Node>> {
        let port_kind = op_type.port_kind(port).unwrap();
        let dir = port.direction();

        let mut links = self.hugr.linked_ports(node, port).peekable();
        // Linear dataflow values must be used, and control must have somewhere to flow.
        let outgoing_is_unique = port_kind.is_linear() || port_kind == EdgeKind::ControlFlow;
        // All dataflow wires must have a unique source.
        let incoming_is_unique = port_kind.is_value() || port_kind.is_const();
        let must_be_connected = match dir {
            // Incoming ports must be connected, except for state order ports, branch case nodes,
            // and CFG nodes.
            Direction::Incoming => {
                port_kind != EdgeKind::StateOrder
                    && port_kind != EdgeKind::ControlFlow
                    && op_type.tag() != OpTag::Case
            }
            Direction::Outgoing => outgoing_is_unique,
        };
        if must_be_connected && links.peek().is_none() {
            return Err(ValidationError::UnconnectedPort {
                node,
                port,
                port_kind: Box::new(port_kind),
            });
        }

        // Avoid double checking connected port types.
        if dir == Direction::Incoming {
            if incoming_is_unique && links.nth(1).is_some() {
                return Err(ValidationError::TooManyConnections {
                    node,
                    port,
                    port_kind: Box::new(port_kind),
                });
            }
            return Ok(());
        }

        self.validate_port_kind(&port_kind, var_decls)
            .map_err(|cause| ValidationError::SignatureError {
                node,
                op: op_type.name(),
                cause,
            })?;

        let mut link_cnt = 0;
        for (other_node, other_offset) in links {
            link_cnt += 1;
            if outgoing_is_unique && link_cnt > 1 {
                return Err(ValidationError::TooManyConnections {
                    node,
                    port,
                    port_kind: Box::new(port_kind),
                });
            }

            let other_op = self.hugr.get_optype(other_node);
            let Some(other_kind) = other_op.port_kind(other_offset) else {
                panic!(
                    "The number of ports in {other_node} does not match the operation definition. This should have been caught by `validate_node`."
                );
            };
            if other_kind != port_kind {
                return Err(ValidationError::IncompatiblePorts {
                    from: node,
                    from_port: port,
                    from_kind: Box::new(port_kind),
                    to: other_node,
                    to_port: other_offset,
                    to_kind: Box::new(other_kind),
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
            EdgeKind::Value(ty) => ty.validate(var_decls),
            // Static edges must *not* refer to type variables declared by enclosing FuncDefns
            // as these are only types at runtime.
            EdgeKind::Const(ty) => ty.validate(&[]),
            EdgeKind::Function(pf) => pf.validate(),
            _ => Ok(()),
        }
    }

    /// Check operation-specific constraints.
    ///
    /// These are flags defined for each operation type as an [`OpValidityFlags`] object.
    fn validate_children(
        &self,
        node: H::Node,
        op_type: &OpType,
    ) -> Result<(), ValidationError<H::Node>> {
        let flags = op_type.validity_flags();

        if self.hugr.children(node).count() > 0 {
            if flags.allowed_children.is_empty() {
                return Err(ValidationError::NonContainerWithChildren {
                    node,
                    optype: Box::new(op_type.clone()),
                });
            }

            let all_children = self.hugr.children(node);
            let mut first_two_children = all_children.clone().take(2);
            let first_child = self.hugr.get_optype(first_two_children.next().unwrap());
            if !flags.allowed_first_child.is_superset(first_child.tag()) {
                return Err(ValidationError::InvalidInitialChild {
                    parent: node,
                    parent_optype: Box::new(op_type.clone()),
                    optype: Box::new(first_child.clone()),
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
                        parent_optype: Box::new(op_type.clone()),
                        optype: Box::new(second_child.clone()),
                        expected: flags.allowed_second_child,
                        position: "second",
                    });
                }
            }
            // Additional validations running over the full list of children optypes
            let children_optypes = all_children.map(|c| (c, self.hugr.get_optype(c)));
            if let Err(source) = op_type.validate_op_children(children_optypes) {
                return Err(ValidationError::InvalidChildren {
                    parent: node,
                    parent_optype: Box::new(op_type.clone()),
                    source,
                });
            }

            // Additional validations running over the edges of the contained graph
            if let Some(edge_check) = flags.edge_check {
                for source in self.hugr.children(node) {
                    for target in self.hugr.output_neighbours(source) {
                        if self.hugr.get_parent(target) != Some(node) {
                            continue;
                        }
                        let source_op = self.hugr.get_optype(source);
                        let target_op = self.hugr.get_optype(target);
                        for [source_port, target_port] in self.hugr.node_connections(source, target)
                        {
                            let edge_data = ChildrenEdgeData {
                                source,
                                target,
                                source_port,
                                target_port,
                                source_op: source_op.clone(),
                                target_op: target_op.clone(),
                            };
                            if let Err(source) = edge_check(edge_data) {
                                return Err(ValidationError::InvalidEdges {
                                    parent: node,
                                    parent_optype: Box::new(op_type.clone()),
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
                optype: Box::new(op_type.clone()),
            });
        }

        Ok(())
    }

    /// Ensure that the children of a node form a directed acyclic graph.
    ///
    /// Inter-graph edges are ignored. Only internal dataflow, constant, or
    /// state order edges are considered.
    fn validate_children_dag(
        &self,
        parent: H::Node,
        op_type: &OpType,
    ) -> Result<(), ValidationError<H::Node>> {
        if self.hugr.children(parent).next().is_none() {
            // No children, nothing to do
            return Ok(());
        }

        let (region, node_map) = self.hugr.region_portgraph(parent);
        let postorder = Topo::new(&region);
        let nodes_visited = postorder
            .iter(&region)
            .filter(|n| *n != node_map.to_portgraph(parent))
            .count();
        let node_count = self.hugr.children(parent).count();
        if nodes_visited != node_count {
            return Err(ValidationError::NotADag {
                node: parent,
                optype: Box::new(op_type.clone()),
            });
        }

        Ok(())
    }

    /// Check the edge is valid, i.e. the source/target nodes are at appropriate
    /// positions in the hierarchy for some locality:
    /// - Local edges, of any kind;
    /// - External edges, for static and value edges only: from a node to a sibling's descendant.
    ///   For Value edges, there must also be an order edge between the copy and the sibling.
    /// - Dominator edges, for value edges only: from a node in a `BasicBlock` node to
    ///   a descendant of a post-dominating sibling of the `BasicBlock`.
    fn validate_edge(
        &mut self,
        from: H::Node,
        from_offset: Port,
        from_optype: &OpType,
        to: H::Node,
        to_offset: Port,
    ) -> Result<(), InterGraphEdgeError<H::Node>> {
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
                ty: Box::new(edge_kind),
            });
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
                        .node_connections(from, ancestor)
                        .find(|&[p, _]| from_optype.port_kind(p) == Some(EdgeKind::StateOrder))
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
                        ancestor_parent_op: Box::new(ancestor_parent_op.clone()),
                    });
                }

                // Check domination
                let (dominator_tree, node_map) =
                    if let Some(tree) = self.dominators.get(&ancestor_parent) {
                        tree
                    } else {
                        let (tree, node_map) = self.compute_dominator(ancestor_parent);
                        self.dominators.insert(ancestor_parent, (tree, node_map));
                        self.dominators.get(&ancestor_parent).unwrap()
                    };
                if !dominator_tree
                    .dominators(node_map.to_portgraph(ancestor))
                    .is_some_and(|mut ds| ds.any(|n| n == node_map.to_portgraph(from_parent)))
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

    /// Validates that `TypeArgs` are valid wrt the [`ExtensionRegistry`] and that nodes
    /// only refer to type variables declared by the closest enclosing `FuncDefn`.
    fn validate_subtree(
        &mut self,
        node: H::Node,
        var_decls: &[TypeParam],
    ) -> Result<(), ValidationError<H::Node>> {
        let op_type = self.hugr.get_optype(node);
        // The op_type must be defined only in terms of type variables defined outside the node

        let validate_ext = |ext_op: &ExtensionOp| -> Result<(), ValidationError<H::Node>> {
            // Check TypeArgs are valid, and if we can, fit the declared TypeParams
            ext_op
                .def()
                .validate_args(ext_op.args(), var_decls)
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
                    opaque.qualified_id(),
                    opaque.extension().clone(),
                ))?;
            }
            OpType::Call(c) => {
                c.validate()
                    .map_err(|cause| ValidationError::SignatureError {
                        node,
                        op: op_type.name(),
                        cause,
                    })?;
            }
            OpType::LoadFunction(c) => {
                c.validate()
                    .map_err(|cause| ValidationError::SignatureError {
                        node,
                        op: op_type.name(),
                        cause,
                    })?;
            }
            _ => (),
        }

        // Check port connections.
        //
        // Root nodes are ignored, as they cannot have connected edges.
        if node != self.hugr.entrypoint() {
            for dir in Direction::BOTH {
                for port in self.hugr.node_ports(node, dir) {
                    self.validate_port(node, port, op_type, var_decls)?;
                }
            }
        }

        // For FuncDefn's, only the type variables declared by the FuncDefn can be referred to by nodes
        // inside the function. (The same would be true for FuncDecl's, but they have no child nodes.)
        let var_decls = if let OpType::FuncDefn(fd) = op_type {
            fd.signature().params()
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
pub enum ValidationError<N: HugrNode> {
    /// The root node of the Hugr is not a root in the hierarchy.
    #[error("The root node of the Hugr ({node}) is not a root in the hierarchy.")]
    RootNotRoot { node: N },
    /// The node ports do not match the operation signature.
    #[error(
        "{node} has an invalid number of ports. The operation {optype} cannot have {actual} {dir:?} ports. Expected {expected}."
    )]
    WrongNumberOfPorts {
        node: N,
        optype: Box<OpType>,
        actual: usize,
        expected: usize,
        dir: Direction,
    },
    /// A dataflow port is not connected.
    #[error("{node} has an unconnected port {port} of type {port_kind}.")]
    UnconnectedPort {
        node: N,
        port: Port,
        port_kind: Box<EdgeKind>,
    },
    /// A linear port is connected to more than one thing.
    #[error("{node} has a port {port} of type {port_kind} with more than one connection.")]
    TooManyConnections {
        node: N,
        port: Port,
        port_kind: Box<EdgeKind>,
    },
    /// Connected ports have different types, or non-unifiable types.
    #[error(
        "Connected ports {from_port} in {from} and {to_port} in {to} have incompatible kinds. Cannot connect {from_kind} to {to_kind}."
    )]
    IncompatiblePorts {
        from: N,
        from_port: Port,
        from_kind: Box<EdgeKind>,
        to: N,
        to_port: Port,
        to_kind: Box<EdgeKind>,
    },
    /// The non-root node has no parent.
    #[error("{node} has no parent.")]
    NoParent { node: N },
    /// The parent node is not compatible with the child node.
    #[error("The operation {parent_optype} cannot contain a {child_optype} as a child. Allowed children: {}. In {child} with parent {parent}.", allowed_children.description())]
    InvalidParentOp {
        child: N,
        child_optype: Box<OpType>,
        parent: N,
        parent_optype: Box<OpType>,
        allowed_children: OpTag,
    },
    /// Invalid first/second child.
    #[error(
        "A {optype} operation cannot be the {position} child of a {parent_optype}. Expected {expected}. In parent {parent}"
    )]
    InvalidInitialChild {
        parent: N,
        parent_optype: Box<OpType>,
        optype: Box<OpType>,
        expected: OpTag,
        position: &'static str,
    },
    /// The children list has invalid elements.
    #[error(
        "An operation {parent_optype} contains invalid children: {source}. In parent {parent}, child {child}",
        child=source.child(),
    )]
    InvalidChildren {
        parent: N,
        parent_optype: Box<OpType>,
        source: ChildrenValidationError<N>,
    },
    /// Multiple, incompatible, nodes with [Visibility::Public] use the same `func_name`
    /// in a [Module](super::Module). (Multiple [`FuncDecl`](crate::ops::FuncDecl)s with
    /// the same signature are allowed)
    #[error("FuncDefn/Decl {} is exported under same name {link_name} as earlier node {}", children[0], children[1])]
    DuplicateExport {
        /// The `func_name` of a public `FuncDecl` or `FuncDefn`
        link_name: String,
        /// Two nodes using that name
        children: [N; 2],
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
        parent: N,
        parent_optype: Box<OpType>,
        source: EdgeValidationError<N>,
    },
    /// The node operation is not a container, but has children.
    #[error("{node} with optype {optype} is not a container, but has children.")]
    NonContainerWithChildren { node: N, optype: Box<OpType> },
    /// The node must have children, but has none.
    #[error("{node} with optype {optype} must have children, but has none.")]
    ContainerWithoutChildren { node: N, optype: Box<OpType> },
    /// The children of a node do not form a DAG.
    #[error(
        "The children of an operation {optype} must form a DAG. Loops are not allowed. In {node}."
    )]
    NotADag { node: N, optype: Box<OpType> },
    /// There are invalid inter-graph edges.
    #[error(transparent)]
    InterGraphEdgeError(#[from] InterGraphEdgeError<N>),
    /// A node claims to still be awaiting extension inference. Perhaps it is not acted upon by inference.
    #[error(
        "{node} needs a concrete ExtensionSet - inference will provide this for Case/CFG/Conditional/DataflowBlock/DFG/TailLoop only"
    )]
    ExtensionsNotInferred { node: N },
    /// Error in a node signature
    #[error("Error in signature of operation {op} at {node}: {cause}")]
    SignatureError {
        node: N,
        op: OpName,
        #[source]
        cause: SignatureError,
    },
    /// Error in a [`ExtensionOp`] serialized as an [Opaque].
    ///
    /// [ExtensionOp]: crate::ops::ExtensionOp
    /// [Opaque]: crate::ops::OpaqueOp
    #[error(transparent)]
    OpaqueOpError(#[from] OpaqueOpError<N>),
    /// A [Const] contained a [Value] of unexpected [Type].
    ///
    /// [Const]: crate::ops::Const
    /// [Value]: crate::ops::Value
    /// [Type]: crate::types::Type
    #[error(transparent)]
    ConstTypeError(#[from] ConstTypeError),
    /// The HUGR entrypoint must be a region container.
    #[error("The HUGR entrypoint ({node}) must be a region container, but '{}' does not accept children.", optype.name())]
    EntrypointNotContainer { node: N, optype: Box<OpType> },
}

/// Errors related to the inter-graph edge validations.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum InterGraphEdgeError<N: HugrNode> {
    /// Inter-Graph edges can only carry copyable data.
    #[error(
        "Inter-graph edges can only carry copyable data. In an inter-graph edge from {from} ({from_offset}) to {to} ({to_offset}) with type {ty}."
    )]
    NonCopyableData {
        from: N,
        from_offset: Port,
        to: N,
        to_offset: Port,
        ty: Box<EdgeKind>,
    },
    /// The grandparent of a dominator inter-graph edge must be a CFG container.
    #[error(
        "The grandparent of a dominator inter-graph edge must be a CFG container. Found operation {ancestor_parent_op}. In a dominator inter-graph edge from {from} ({from_offset}) to {to} ({to_offset})."
    )]
    NonCFGAncestor {
        from: N,
        from_offset: Port,
        to: N,
        to_offset: Port,
        ancestor_parent_op: Box<OpType>,
    },
    /// The sibling ancestors of the external inter-graph edge endpoints must be have an order edge between them.
    #[error(
        "Missing state order between the external inter-graph source {from} and the ancestor of the target {to_ancestor}. In an external inter-graph edge from {from} ({from_offset}) to {to} ({to_offset})."
    )]
    MissingOrderEdge {
        from: N,
        from_offset: Port,
        to: N,
        to_offset: Port,
        to_ancestor: N,
    },
    /// The ancestors of an inter-graph edge are not related.
    #[error(
        "The ancestors of an inter-graph edge are not related. In an inter-graph edge from {from} ({from_offset}) to {to} ({to_offset})."
    )]
    NoRelation {
        from: N,
        from_offset: Port,
        to: N,
        to_offset: Port,
    },
    /// The basic block containing the source node does not dominate the basic block containing the target node.
    #[error(
        " The basic block containing the source node does not dominate the basic block containing the target node in the CFG. Expected {from_parent} to dominate {ancestor}. In a dominator inter-graph edge from {from} ({from_offset}) to {to} ({to_offset})."
    )]
    NonDominatedAncestor {
        from: N,
        from_offset: Port,
        to: N,
        to_offset: Port,
        from_parent: N,
        ancestor: N,
    },
}

#[cfg(test)]
pub(crate) mod test;
