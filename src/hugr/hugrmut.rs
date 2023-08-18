//! Base HUGR builder providing low-level building blocks.

use std::collections::HashMap;
use std::ops::Range;

use portgraph::{LinkMut, NodeIndex, PortMut, PortView, SecondaryMap};

use crate::hugr::{Direction, HugrError, HugrView, Node, NodeType};
use crate::ops::OpType;

use crate::ops::handle::NodeHandle;
use crate::{Hugr, Port};

use super::NodeMetadata;

pub(crate) use sealed::HugrInternalsMut;

pub trait HugrMut: sealed::HugrInternalsMut {
    /// The kind of handle that can be used to refer to the root node.
    ///
    /// The handle is guaranteed to be able to contain the operation returned by
    /// [`HugrView::root_type`].
    type RootHandle: NodeHandle;

    /// The view interface for this mutable reference to a Hugr.
    type View<'h>: 'h + HugrView<RootHandle = Self::RootHandle>
    where
        Self: 'h;

    /// Returns a shared view of the Hugr.
    fn view(&self) -> Self::View<'_>;
}

impl<T> HugrMut for T
where
    T: AsRef<Hugr> + AsMut<Hugr>,
{
    type RootHandle = Node;

    type View<'h> = &'h Hugr where Self: 'h;

    fn view(&self) -> Self::View<'_> {
        self.as_ref()
    }
}

pub(crate) mod sealed {
    use super::*;
    /// Functions for low-level building of a HUGR. (Or, in the future, a subregion thereof)
    pub trait HugrInternalsMut {
        /// Add a node to the graph, with the default conversion from OpType to NodeType
        fn add_op(&mut self, op: impl Into<OpType>) -> Node;

        /// Add a node to the graph.
        fn add_node(&mut self, node: NodeType) -> Node;

        /// Remove a node from the graph.
        ///
        /// # Panics
        ///
        /// Panics if the node is the root node.
        fn remove_node(&mut self, node: Node) -> Result<(), HugrError>;

        /// Returns the metadata associated with a node.
        fn get_metadata_mut(&mut self, node: Node) -> &mut NodeMetadata;

        /// Sets the metadata associated with a node.
        fn set_metadata(&mut self, node: Node, metadata: NodeMetadata) {
            *self.get_metadata_mut(node) = metadata;
        }

        /// Connect two nodes at the given ports.
        ///
        /// The port must have already been created. See [`add_ports`] and [`set_num_ports`].
        ///
        /// [`add_ports`]: #method.add_ports
        /// [`set_num_ports`]: #method.set_num_ports.
        fn connect(
            &mut self,
            src: Node,
            src_port: usize,
            dst: Node,
            dst_port: usize,
        ) -> Result<(), HugrError>;

        /// Disconnects all edges from the given port.
        ///
        /// The port is left in place.
        fn disconnect(&mut self, node: Node, port: Port) -> Result<(), HugrError>;

        /// Adds a non-dataflow edge between two nodes. The kind is given by the
        /// operation's [`OpType::other_input`] or [`OpType::other_output`].
        ///
        /// Returns the offsets of the new input and output ports, or an error if
        /// the connection failed.
        ///
        /// [`OpType::other_input`]: crate::ops::OpType::other_input
        /// [`OpType::other_output`]: crate::ops::OpType::other_output.
        fn add_other_edge(&mut self, src: Node, dst: Node) -> Result<(Port, Port), HugrError>;

        /// Set the number of ports on a node. This may invalidate the node's `PortIndex`.
        fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize);

        /// Alter the number of ports on a node and returns a range with the new
        /// port offsets, if any. This may invalidate the node's `PortIndex`.
        ///
        /// The `direction` parameter specifies whether to add ports to the incoming
        /// or outgoing list.
        fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize>;

        /// Sets the parent of a node.
        ///
        /// The node becomes the parent's last child.
        fn set_parent(&mut self, node: Node, parent: Node) -> Result<(), HugrError>;

        /// Move a node in the hierarchy to be the subsequent sibling of another
        /// node.
        ///
        /// The sibling node's parent becomes the new node's parent.
        ///
        /// The node becomes the parent's last child.
        fn move_after_sibling(&mut self, node: Node, after: Node) -> Result<(), HugrError>;

        /// Move a node in the hierarchy to be the prior sibling of another node.
        ///
        /// The sibling node's parent becomes the new node's parent.
        ///
        /// The node becomes the parent's last child.
        fn move_before_sibling(&mut self, node: Node, before: Node) -> Result<(), HugrError>;

        /// Add a node to the graph with a parent in the hierarchy.
        ///
        /// The node becomes the parent's last child.
        fn add_op_with_parent(
            &mut self,
            parent: Node,
            op: impl Into<OpType>,
        ) -> Result<Node, HugrError>;

        /// Add a node to the graph with a parent in the hierarchy.
        ///
        /// The node becomes the parent's last child.
        fn add_node_with_parent(&mut self, parent: Node, op: NodeType) -> Result<Node, HugrError>;

        /// Add a node to the graph as the previous sibling of another node.
        ///
        /// The sibling node's parent becomes the new node's parent.
        ///
        /// # Errors
        ///
        ///  - If the sibling node does not have a parent.
        ///  - If the attachment would introduce a cycle.
        fn add_op_before(
            &mut self,
            sibling: Node,
            op: impl Into<OpType>,
        ) -> Result<Node, HugrError>;

        /// Add a node to the graph as the next sibling of another node.
        ///
        /// The sibling node's parent becomes the new node's parent.
        ///
        /// # Errors
        ///
        ///  - If the sibling node does not have a parent.
        ///  - If the attachment would introduce a cycle.
        fn add_op_after(&mut self, sibling: Node, op: impl Into<OpType>)
            -> Result<Node, HugrError>;

        /// Replace the OpType at node and return the old OpType.
        /// In general this invalidates the ports, which may need to be resized to
        /// match the OpType signature.
        /// TODO: Add a version which ignores input extensions
        fn replace_op(&mut self, node: Node, op: NodeType) -> NodeType;

        /// Insert another hugr into this one, under a given root node.
        ///
        /// Returns the root node of the inserted hugr.
        fn insert_hugr(&mut self, root: Node, other: Hugr) -> Result<Node, HugrError>;

        /// Copy another hugr into this one, under a given root node.
        ///
        /// Returns the root node of the inserted hugr.
        fn insert_from_view(
            &mut self,
            root: Node,
            other: &impl HugrView,
        ) -> Result<Node, HugrError>;

        /// Compact the nodes indices of the hugr to be contiguous, and order them as a breadth-first
        /// traversal of the hierarchy.
        ///
        /// The rekey function is called for each moved node with the old and new indices.
        ///
        /// After this operation, a serialization and deserialization of the Hugr is guaranteed to
        /// preserve the indices.
        fn canonicalize_nodes(&mut self, rekey: impl FnMut(Node, Node));
    }

    impl<T> HugrInternalsMut for T
    where
        T: AsRef<Hugr> + AsMut<Hugr>,
    {
        fn add_node(&mut self, nodetype: NodeType) -> Node {
            let node = self
                .as_mut()
                .graph
                .add_node(nodetype.input_count(), nodetype.output_count());
            self.as_mut().op_types[node] = nodetype;
            node.into()
        }

        fn add_op(&mut self, op: impl Into<OpType>) -> Node {
            // TODO: Default to `NodeType::open_extensions` once we can infer extensions
            self.add_node(NodeType::pure(op))
        }

        fn remove_node(&mut self, node: Node) -> Result<(), HugrError> {
            if node.index == self.as_ref().root {
                // TODO: Add a HugrMutError ?
                panic!("cannot remove root node");
            }
            self.as_mut().hierarchy.remove(node.index);
            self.as_mut().graph.remove_node(node.index);
            self.as_mut().op_types.remove(node.index);
            Ok(())
        }

        fn get_metadata_mut(&mut self, node: Node) -> &mut NodeMetadata {
            self.as_mut().metadata.get_mut(node.index)
        }

        fn connect(
            &mut self,
            src: Node,
            src_port: usize,
            dst: Node,
            dst_port: usize,
        ) -> Result<(), HugrError> {
            self.as_mut()
                .graph
                .link_nodes(src.index, src_port, dst.index, dst_port)?;
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
            self.connect(src, src_port.index(), dst, dst_port.index())?;
            Ok((src_port, dst_port))
        }

        #[inline]
        fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
            self.as_mut()
                .graph
                .set_num_ports(node.index, incoming, outgoing, |_, _| {})
        }

        #[inline]
        fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize> {
            let mut incoming = self.as_mut().graph.num_inputs(node.index);
            let mut outgoing = self.as_mut().graph.num_outputs(node.index);
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
            self.as_mut()
                .graph
                .set_num_ports(node.index, incoming, outgoing, |_, _| {});
            range
        }

        fn set_parent(&mut self, node: Node, parent: Node) -> Result<(), HugrError> {
            self.as_mut().hierarchy.detach(node.index);
            self.as_mut()
                .hierarchy
                .push_child(node.index, parent.index)?;
            Ok(())
        }

        fn move_after_sibling(&mut self, node: Node, after: Node) -> Result<(), HugrError> {
            self.as_mut().hierarchy.detach(node.index);
            self.as_mut()
                .hierarchy
                .insert_after(node.index, after.index)?;
            Ok(())
        }

        fn move_before_sibling(&mut self, node: Node, before: Node) -> Result<(), HugrError> {
            self.as_mut().hierarchy.detach(node.index);
            self.as_mut()
                .hierarchy
                .insert_before(node.index, before.index)?;
            Ok(())
        }

        fn add_op_with_parent(
            &mut self,
            parent: Node,
            op: impl Into<OpType>,
        ) -> Result<Node, HugrError> {
            // TODO: Default to `NodeType::open_extensions` once we can infer extensions
            self.add_node_with_parent(parent, NodeType::pure(op))
        }

        fn add_node_with_parent(
            &mut self,
            parent: Node,
            node: NodeType,
        ) -> Result<Node, HugrError> {
            let node = self.add_node(node);
            self.as_mut()
                .hierarchy
                .push_child(node.index, parent.index)?;
            Ok(node)
        }

        fn add_op_before(
            &mut self,
            sibling: Node,
            op: impl Into<OpType>,
        ) -> Result<Node, HugrError> {
            let node = self.add_op(op);
            self.as_mut()
                .hierarchy
                .insert_before(node.index, sibling.index)?;
            Ok(node)
        }

        fn add_op_after(
            &mut self,
            sibling: Node,
            op: impl Into<OpType>,
        ) -> Result<Node, HugrError> {
            let node = self.add_op(op);
            self.as_mut()
                .hierarchy
                .insert_after(node.index, sibling.index)?;
            Ok(node)
        }

        fn replace_op(&mut self, node: Node, op: NodeType) -> NodeType {
            let cur = self.as_mut().op_types.get_mut(node.index);
            std::mem::replace(cur, op)
        }

        fn insert_hugr(&mut self, root: Node, mut other: Hugr) -> Result<Node, HugrError> {
            let (other_root, node_map) = insert_hugr_internal(self.as_mut(), root, &other)?;
            // Update the optypes and metadata, taking them from the other graph.
            for (&node, &new_node) in node_map.iter() {
                let optype = other.op_types.take(node);
                self.as_mut().op_types.set(new_node, optype);
                let meta = other.metadata.take(node);
                self.as_mut().set_metadata(node.into(), meta);
            }
            Ok(other_root)
        }

        fn insert_from_view(
            &mut self,
            root: Node,
            other: &impl HugrView,
        ) -> Result<Node, HugrError> {
            let (other_root, node_map) = insert_hugr_internal(self.as_mut(), root, other)?;
            // Update the optypes and metadata, copying them from the other graph.
            for (&node, &new_node) in node_map.iter() {
                let nodetype = other.get_nodetype(node.into());
                self.as_mut().op_types.set(new_node, nodetype.clone());
                let meta = other.get_metadata(node.into());
                self.as_mut().set_metadata(node.into(), meta.clone());
            }
            Ok(other_root)
        }

        fn canonicalize_nodes(&mut self, mut rekey: impl FnMut(Node, Node)) {
            // Generate the ordered list of nodes
            let mut ordered = Vec::with_capacity(self.node_count());
            ordered.extend(self.as_ref().canonical_order());

            // Permute the nodes in the graph to match the order.
            //
            // Invariant: All the elements before `position` are in the correct place.
            for position in 0..ordered.len() {
                // Find the element's location. If it originally came from a previous position
                // then it has been swapped somewhere else, so we follow the permutation chain.
                let mut source: Node = ordered[position];
                while position > source.index.index() {
                    source = ordered[source.index.index()];
                }

                let target: Node = NodeIndex::new(position).into();
                if target != source {
                    let hugr = self.as_mut();
                    hugr.graph.swap_nodes(target.index, source.index);
                    hugr.op_types.swap(target.index, source.index);
                    hugr.hierarchy.swap_nodes(target.index, source.index);
                    rekey(source, target);
                }
            }
            self.as_mut().root = NodeIndex::new(0);

            // Finish by compacting the copy nodes.
            // The operation nodes will be left in place.
            // This step is not strictly necessary.
            self.as_mut().graph.compact_nodes(|_, _| {});
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
        let node_map = hugr.graph.insert_graph(other.portgraph())?;
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
}

#[cfg(test)]
mod test {
    use crate::{
        hugr::HugrView,
        macros::type_row,
        ops::{self, dataflow::IOTrait, LeafOp},
        types::{test::COPYABLE_T, AbstractSignature, Type},
    };

    use super::sealed::HugrInternalsMut;
    use super::*;

    const NAT: Type = COPYABLE_T;

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
                    signature: AbstractSignature::new_df(type_row![NAT], type_row![NAT, NAT]),
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
                .add_op_with_parent(f, LeafOp::Noop { ty: COPYABLE_T })
                .unwrap();

            assert!(builder.connect(f_in, 0, noop, 0).is_ok());
            assert!(builder.connect(noop, 0, f_out, 0).is_ok());
            assert!(builder.connect(noop, 0, f_out, 1).is_ok());
        }

        // Finish the construction and create the HUGR
        builder.validate().unwrap();
    }
}
