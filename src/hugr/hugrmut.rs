//! Low-level interface for modifying a HUGR.

use std::collections::HashMap;
use std::ops::Range;

use portgraph::view::{NodeFilter, NodeFiltered};
use portgraph::{LinkMut, NodeIndex, PortMut, PortView, SecondaryMap};

use crate::hugr::{Direction, HugrError, HugrView, Node, NodeType, RootTagged};
use crate::ops::OpType;

use crate::{Hugr, Port};

use self::sealed::HugrMutInternals;

use super::views::SiblingSubgraph;
use super::{IncomingPort, NodeMetadata, OutgoingPort, PortIndex, Rewrite};

/// Functions for low-level building of a HUGR.
pub trait HugrMut: HugrMutInternals {
    /// Returns the metadata associated with a node.
    fn get_metadata_mut(&mut self, node: Node) -> Result<&mut NodeMetadata, HugrError> {
        self.valid_node(node)?;
        Ok(self.hugr_mut().metadata.get_mut(node.index))
    }

    /// Sets the metadata associated with a node.
    fn set_metadata(&mut self, node: Node, metadata: NodeMetadata) -> Result<(), HugrError> {
        *self.get_metadata_mut(node)? = metadata;
        Ok(())
    }

    /// Add a node to the graph with a parent in the hierarchy.
    ///
    /// The node becomes the parent's last child.
    #[inline]
    fn add_op_with_parent(
        &mut self,
        parent: Node,
        op: impl Into<OpType>,
    ) -> Result<Node, HugrError> {
        // TODO: Default to `NodeType::open_extensions` once we can infer extensions
        self.add_node_with_parent(parent, NodeType::pure(op))
    }

    /// Add a node to the graph with a parent in the hierarchy.
    ///
    /// The node becomes the parent's last child.
    #[inline]
    fn add_node_with_parent(&mut self, parent: Node, op: NodeType) -> Result<Node, HugrError> {
        self.valid_node(parent)?;
        self.hugr_mut().add_node_with_parent(parent, op)
    }

    /// Add a node to the graph as the previous sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Errors
    ///
    ///  - If the sibling node does not have a parent.
    ///  - If the attachment would introduce a cycle.
    #[inline]
    fn add_op_before(&mut self, sibling: Node, op: impl Into<OpType>) -> Result<Node, HugrError> {
        self.valid_non_root(sibling)?;
        self.hugr_mut().add_op_before(sibling, op)
    }

    /// A generalisation of [`HugrMut::add_op_before`], needed temporarily until
    /// add_op type methods all default to creating nodes with open extensions.
    /// See issue #424
    #[inline]
    fn add_node_before(&mut self, sibling: Node, nodetype: NodeType) -> Result<Node, HugrError> {
        self.valid_non_root(sibling)?;
        self.hugr_mut().add_node_before(sibling, nodetype)
    }

    /// Add a node to the graph as the next sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Errors
    ///
    ///  - If the sibling node does not have a parent.
    ///  - If the attachment would introduce a cycle.
    #[inline]
    fn add_op_after(&mut self, sibling: Node, op: impl Into<OpType>) -> Result<Node, HugrError> {
        self.valid_non_root(sibling)?;
        self.hugr_mut().add_op_after(sibling, op)
    }

    /// Remove a node from the graph.
    ///
    /// # Panics
    ///
    /// Panics if the node is the root node.
    #[inline]
    fn remove_node(&mut self, node: Node) -> Result<(), HugrError> {
        self.valid_non_root(node)?;
        self.hugr_mut().remove_node(node)
    }

    /// Connect two nodes at the given ports.
    ///
    /// The port must have already been created. See [`add_ports`] and [`set_num_ports`].
    ///
    /// [`add_ports`]: #method.add_ports
    /// [`set_num_ports`]: #method.set_num_ports.
    #[inline]
    fn connect(
        &mut self,
        src: Node,
        src_port: impl TryInto<OutgoingPort>,
        dst: Node,
        dst_port: impl TryInto<IncomingPort>,
    ) -> Result<(), HugrError> {
        self.valid_node(src)?;
        self.valid_node(dst)?;
        self.hugr_mut().connect(src, src_port, dst, dst_port)
    }

    /// Disconnects all edges from the given port.
    ///
    /// The port is left in place.
    #[inline]
    fn disconnect(&mut self, node: Node, port: Port) -> Result<(), HugrError> {
        self.valid_node(node)?;
        self.hugr_mut().disconnect(node, port)
    }

    /// Adds a non-dataflow edge between two nodes. The kind is given by the
    /// operation's [`OpTrait::other_input`] or [`OpTrait::other_output`].
    ///
    /// Returns the offsets of the new input and output ports, or an error if
    /// the connection failed.
    ///
    /// [`OpTrait::other_input`]: crate::ops::OpTrait::other_input
    /// [`OpTrait::other_output`]: crate::ops::OpTrait::other_output
    fn add_other_edge(&mut self, src: Node, dst: Node) -> Result<(Port, Port), HugrError> {
        self.valid_node(src)?;
        self.valid_node(dst)?;
        self.hugr_mut().add_other_edge(src, dst)
    }

    /// Insert another hugr into this one, under a given root node.
    #[inline]
    fn insert_hugr(&mut self, root: Node, other: Hugr) -> Result<InsertionResult, HugrError> {
        self.valid_node(root)?;
        self.hugr_mut().insert_hugr(root, other)
    }

    /// Copy another hugr into this one, under a given root node.
    #[inline]
    fn insert_from_view(
        &mut self,
        root: Node,
        other: &impl HugrView,
    ) -> Result<InsertionResult, HugrError> {
        self.valid_node(root)?;
        self.hugr_mut().insert_from_view(root, other)
    }

    /// Copy a subgraph from another hugr into this one, under a given root node.
    ///
    /// Sibling order is not preserved.
    ///
    /// The returned `InsertionResult` does not contain a `new_root` value, since
    /// a subgraph may not have a defined root.
    //
    // TODO: Try to preserve the order when possible? We cannot always ensure
    // it, since the subgraph may have arbitrary nodes without including their
    // parent.
    fn insert_subgraph(
        &mut self,
        root: Node,
        other: &impl HugrView,
        subgraph: &SiblingSubgraph,
    ) -> Result<InsertionResult, HugrError> {
        self.valid_node(root)?;
        self.hugr_mut().insert_subgraph(root, other, subgraph)
    }

    /// Applies a rewrite to the graph.
    fn apply_rewrite<R, E>(&mut self, rw: impl Rewrite<ApplyResult = R, Error = E>) -> Result<R, E>
    where
        Self: Sized,
    {
        rw.apply(self)
    }
}

/// Records the result of inserting a Hugr or view
/// via [HugrMut::insert_hugr] or [HugrMut::insert_from_view]
pub struct InsertionResult {
    /// The node, after insertion, that was the root of the inserted Hugr.
    ///
    /// That is, the value in [InsertionResult::node_map] under the key that was the [HugrView::root]
    ///
    /// When inserting a subgraph, this value is `None`.
    pub new_root: Option<Node>,
    /// Map from nodes in the Hugr/view that was inserted, to their new
    /// positions in the Hugr into which said was inserted.
    pub node_map: HashMap<Node, Node>,
}

impl InsertionResult {
    fn translating_indices(
        new_root: Option<Node>,
        node_map: HashMap<NodeIndex, NodeIndex>,
    ) -> Self {
        Self {
            new_root,
            node_map: HashMap::from_iter(node_map.into_iter().map(|(k, v)| (k.into(), v.into()))),
        }
    }
}

/// Impl for non-wrapped Hugrs. Overwrites the recursive default-impls to directly use the hugr.
impl<T: RootTagged + AsMut<Hugr>> HugrMut for T {
    fn add_node_with_parent(&mut self, parent: Node, node: NodeType) -> Result<Node, HugrError> {
        let node = self.as_mut().add_node(node);
        self.as_mut()
            .hierarchy
            .push_child(node.index, parent.index)?;
        Ok(node)
    }

    fn add_op_before(&mut self, sibling: Node, op: impl Into<OpType>) -> Result<Node, HugrError> {
        self.add_node_before(sibling, NodeType::pure(op))
    }

    fn add_node_before(&mut self, sibling: Node, nodetype: NodeType) -> Result<Node, HugrError> {
        let node = self.as_mut().add_node(nodetype);
        self.as_mut()
            .hierarchy
            .insert_before(node.index, sibling.index)?;
        Ok(node)
    }

    fn add_op_after(&mut self, sibling: Node, op: impl Into<OpType>) -> Result<Node, HugrError> {
        let node = self.as_mut().add_op(op);
        self.as_mut()
            .hierarchy
            .insert_after(node.index, sibling.index)?;
        Ok(node)
    }

    fn remove_node(&mut self, node: Node) -> Result<(), HugrError> {
        if node == self.root() {
            // TODO: Add a HugrMutError ?
            panic!("cannot remove root node");
        }
        self.as_mut().hierarchy.remove(node.index);
        self.as_mut().graph.remove_node(node.index);
        self.as_mut().op_types.remove(node.index);
        Ok(())
    }

    fn connect(
        &mut self,
        src: Node,
        src_port: impl TryInto<OutgoingPort>,
        dst: Node,
        dst_port: impl TryInto<IncomingPort>,
    ) -> Result<(), HugrError> {
        self.as_mut().graph.link_nodes(
            src.index,
            Port::try_new_outgoing(src_port)?.index(),
            dst.index,
            Port::try_new_incoming(dst_port)?.index(),
        )?;
        Ok(())
    }

    fn disconnect(&mut self, node: Node, port: Port) -> Result<(), HugrError> {
        let offset = port.offset;
        let port = self.as_mut().graph.port_index(node.index, offset).ok_or(
            portgraph::LinkError::UnknownOffset {
                node: node.index,
                offset,
            },
        )?;
        self.as_mut().graph.unlink_port(port);
        Ok(())
    }

    fn add_other_edge(&mut self, src: Node, dst: Node) -> Result<(Port, Port), HugrError> {
        let src_port: Port = self
            .get_optype(src)
            .other_port_index(Direction::Outgoing)
            .expect("Source operation has no non-dataflow outgoing edges");
        let dst_port: Port = self
            .get_optype(dst)
            .other_port_index(Direction::Incoming)
            .expect("Destination operation has no non-dataflow incoming edges");
        self.connect(src, src_port, dst, dst_port)?;
        Ok((src_port, dst_port))
    }

    fn insert_hugr(&mut self, root: Node, mut other: Hugr) -> Result<InsertionResult, HugrError> {
        let (other_root, node_map) = insert_hugr_internal(self.as_mut(), root, &other)?;
        // Update the optypes and metadata, taking them from the other graph.
        for (&node, &new_node) in node_map.iter() {
            let optype = other.op_types.take(node);
            self.as_mut().op_types.set(new_node, optype);
            let meta = other.metadata.take(node);
            self.as_mut().set_metadata(new_node.into(), meta).unwrap();
        }
        debug_assert_eq!(Some(&other_root.index), node_map.get(&other.root().index));
        Ok(InsertionResult::translating_indices(
            Some(other_root),
            node_map,
        ))
    }

    fn insert_from_view(
        &mut self,
        root: Node,
        other: &impl HugrView,
    ) -> Result<InsertionResult, HugrError> {
        let (other_root, node_map) = insert_hugr_internal(self.as_mut(), root, other)?;
        // Update the optypes and metadata, copying them from the other graph.
        for (&node, &new_node) in node_map.iter() {
            let nodetype = other.get_nodetype(node.into());
            self.as_mut().op_types.set(new_node, nodetype.clone());
            let meta = other.get_metadata(node.into());
            self.as_mut()
                .set_metadata(new_node.into(), meta.clone())
                .unwrap();
        }
        debug_assert_eq!(Some(&other_root.index), node_map.get(&other.root().index));
        Ok(InsertionResult::translating_indices(
            Some(other_root),
            node_map,
        ))
    }

    fn insert_subgraph(
        &mut self,
        root: Node,
        other: &impl HugrView,
        subgraph: &SiblingSubgraph,
    ) -> Result<InsertionResult, HugrError> {
        // Create a portgraph view with the explicit list of nodes defined by the subgraph.
        let portgraph: NodeFiltered<_, NodeFilter<&[Node]>, &[Node]> =
            NodeFiltered::new_node_filtered(
                other.portgraph(),
                |node, ctx| ctx.contains(&node.into()),
                subgraph.nodes(),
            );
        let node_map = insert_subgraph_internal(self.as_mut(), root, other, &portgraph)?;
        // Update the optypes and metadata, copying them from the other graph.
        for (&node, &new_node) in node_map.iter() {
            let nodetype = other.get_nodetype(node.into());
            self.as_mut().op_types.set(new_node, nodetype.clone());
            let meta = other.get_metadata(node.into());
            self.as_mut()
                .set_metadata(new_node.into(), meta.clone())
                .unwrap();
        }
        Ok(InsertionResult::translating_indices(None, node_map))
    }
}

/// Internal implementation of `insert_hugr` and `insert_view` methods for
/// AsMut<Hugr>.
///
/// Returns the root node of the inserted hierarchy and a mapping from the nodes
/// in the inserted graph to their new indices in `hugr`.
///
/// This function does not update the optypes of the inserted nodes, so the
/// caller must do that.
fn insert_hugr_internal(
    hugr: &mut Hugr,
    root: Node,
    other: &impl HugrView,
) -> Result<(Node, HashMap<NodeIndex, NodeIndex>), HugrError> {
    let node_map = hugr.graph.insert_graph(&other.portgraph())?;
    let other_root = node_map[&other.root().index];

    // Update hierarchy and optypes
    hugr.hierarchy.push_child(other_root, root.index)?;
    for (&node, &new_node) in node_map.iter() {
        other
            .children(node.into())
            .try_for_each(|child| -> Result<(), HugrError> {
                hugr.hierarchy
                    .push_child(node_map[&child.index], new_node)?;
                Ok(())
            })?;
    }

    // The root node didn't have any ports.
    let root_optype = other.get_optype(other.root());
    hugr.set_num_ports(
        other_root.into(),
        root_optype.input_count(),
        root_optype.output_count(),
    );

    Ok((other_root.into(), node_map))
}

/// Internal implementation of the `insert_subgraph` method for AsMut<Hugr>.
///
/// Returns a mapping from the nodes in the inserted graph to their new indices
/// in `hugr`.
///
/// This function does not update the optypes of the inserted nodes, so the
/// caller must do that.
///
/// In contrast to `insert_hugr_internal`, this function does not preserve
/// sibling order in the hierarchy. This is due to the subgraph not necessarily
/// having a single root, so the logic for reconstructing the hierarchy is not
/// able to just do a BFS.
fn insert_subgraph_internal(
    hugr: &mut Hugr,
    root: Node,
    other: &impl HugrView,
    portgraph: &impl portgraph::LinkView,
) -> Result<HashMap<NodeIndex, NodeIndex>, HugrError> {
    let node_map = hugr.graph.insert_graph(&portgraph)?;

    // A map for nodes that we inserted before their parent, so we couldn't
    // update the hierarchy with their new id.
    for (&node, &new_node) in node_map.iter() {
        let new_parent = other
            .get_parent(node.into())
            .and_then(|parent| node_map.get(&parent.index).copied())
            .unwrap_or(root.index);
        hugr.hierarchy.push_child(new_node, new_parent)?;
    }

    Ok(node_map)
}

pub(crate) mod sealed {
    use super::*;
    use crate::ops::handle::NodeHandle;

    /// Trait for accessing the mutable internals of a Hugr(Mut).
    ///
    /// Specifically, this trait lets you apply arbitrary modifications that may
    /// invalidate the HUGR.
    pub trait HugrMutInternals: RootTagged {
        /// Returns the Hugr at the base of a chain of views.
        fn hugr_mut(&mut self) -> &mut Hugr;

        /// Set the number of ports on a node. This may invalidate the node's `PortIndex`.
        fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
            self.valid_node(node).unwrap_or_else(|e| panic!("{}", e));
            self.hugr_mut().set_num_ports(node, incoming, outgoing)
        }

        /// Alter the number of ports on a node and returns a range with the new
        /// port offsets, if any. This may invalidate the node's `PortIndex`.
        ///
        /// The `direction` parameter specifies whether to add ports to the incoming
        /// or outgoing list.
        fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize> {
            self.valid_node(node).unwrap_or_else(|e| panic!("{}", e));
            self.hugr_mut().add_ports(node, direction, amount)
        }

        /// Sets the parent of a node.
        ///
        /// The node becomes the parent's last child.
        fn set_parent(&mut self, node: Node, parent: Node) -> Result<(), HugrError> {
            self.valid_node(parent)?;
            self.valid_non_root(node)?;
            self.hugr_mut().set_parent(node, parent)
        }

        /// Move a node in the hierarchy to be the subsequent sibling of another
        /// node.
        ///
        /// The sibling node's parent becomes the new node's parent.
        ///
        /// The node becomes the parent's last child.
        fn move_after_sibling(&mut self, node: Node, after: Node) -> Result<(), HugrError> {
            self.valid_non_root(node)?;
            self.valid_non_root(after)?;
            self.hugr_mut().move_after_sibling(node, after)
        }

        /// Move a node in the hierarchy to be the prior sibling of another node.
        ///
        /// The sibling node's parent becomes the new node's parent.
        ///
        /// The node becomes the parent's last child.
        fn move_before_sibling(&mut self, node: Node, before: Node) -> Result<(), HugrError> {
            self.valid_non_root(node)?;
            self.valid_non_root(before)?;
            self.hugr_mut().move_before_sibling(node, before)
        }

        /// Replace the OpType at node and return the old OpType.
        /// In general this invalidates the ports, which may need to be resized to
        /// match the OpType signature.
        /// TODO: Add a version which ignores input extensions
        ///
        /// # Errors
        /// Returns a [`HugrError::InvalidTag`] if this would break the bound
        /// ([`Self::RootHandle`]) on the root node's [OpTag]
        fn replace_op(&mut self, node: Node, op: NodeType) -> Result<NodeType, HugrError> {
            self.valid_node(node)?;
            if node == self.root() && !Self::RootHandle::TAG.is_superset(op.tag()) {
                return Err(HugrError::InvalidTag {
                    required: Self::RootHandle::TAG,
                    actual: op.tag(),
                });
            }
            self.hugr_mut().replace_op(node, op)
        }
    }

    /// Impl for non-wrapped Hugrs. Overwrites the recursive default-impls to directly use the hugr.
    impl<T: RootTagged + AsMut<Hugr>> HugrMutInternals for T {
        fn hugr_mut(&mut self) -> &mut Hugr {
            self.as_mut()
        }

        #[inline]
        fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
            self.hugr_mut()
                .graph
                .set_num_ports(node.index, incoming, outgoing, |_, _| {})
        }

        fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize> {
            let mut incoming = self.hugr_mut().graph.num_inputs(node.index);
            let mut outgoing = self.hugr_mut().graph.num_outputs(node.index);
            let increment = |num: &mut usize| {
                let new = num.saturating_add_signed(amount);
                let range = *num..new;
                *num = new;
                range
            };
            let range = match direction {
                Direction::Incoming => increment(&mut incoming),
                Direction::Outgoing => increment(&mut outgoing),
            };
            self.hugr_mut()
                .graph
                .set_num_ports(node.index, incoming, outgoing, |_, _| {});
            range
        }

        fn set_parent(&mut self, node: Node, parent: Node) -> Result<(), HugrError> {
            self.hugr_mut().hierarchy.detach(node.index);
            self.hugr_mut()
                .hierarchy
                .push_child(node.index, parent.index)?;
            Ok(())
        }

        fn move_after_sibling(&mut self, node: Node, after: Node) -> Result<(), HugrError> {
            self.hugr_mut().hierarchy.detach(node.index);
            self.hugr_mut()
                .hierarchy
                .insert_after(node.index, after.index)?;
            Ok(())
        }

        fn move_before_sibling(&mut self, node: Node, before: Node) -> Result<(), HugrError> {
            self.hugr_mut().hierarchy.detach(node.index);
            self.hugr_mut()
                .hierarchy
                .insert_before(node.index, before.index)?;
            Ok(())
        }

        fn replace_op(&mut self, node: Node, op: NodeType) -> Result<NodeType, HugrError> {
            // No possibility of failure here since Self::RootHandle == Any
            let cur = self.hugr_mut().op_types.get_mut(node.index);
            Ok(std::mem::replace(cur, op))
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        extension::prelude::USIZE_T,
        extension::PRELUDE_REGISTRY,
        macros::type_row,
        ops::{self, dataflow::IOTrait, LeafOp},
        types::{FunctionType, Type},
    };

    use super::*;

    const NAT: Type = USIZE_T;

    #[test]
    fn simple_function() {
        // Starts an empty builder
        let mut builder = Hugr::default();

        // Create the root module definition
        let module: Node = builder.root();

        // Start a main function with two nat inputs.
        //
        // `add_op` is equivalent to `add_root_op` followed by `set_parent`
        let f: Node = builder
            .add_op_with_parent(
                module,
                ops::FuncDefn {
                    name: "main".into(),
                    signature: FunctionType::new(type_row![NAT], type_row![NAT, NAT]),
                },
            )
            .expect("Failed to add function definition node");

        {
            let f_in = builder
                .add_op_with_parent(f, ops::Input::new(type_row![NAT]))
                .unwrap();
            let f_out = builder
                .add_op_with_parent(f, ops::Output::new(type_row![NAT, NAT]))
                .unwrap();
            let noop = builder
                .add_op_with_parent(f, LeafOp::Noop { ty: NAT })
                .unwrap();

            assert!(builder.connect(f_in, 0, noop, 0).is_ok());
            assert!(builder.connect(noop, 0, f_out, 0).is_ok());
            assert!(builder.connect(noop, 0, f_out, 1).is_ok());
        }

        // Finish the construction and create the HUGR
        builder.validate(&PRELUDE_REGISTRY).unwrap();
    }
}
