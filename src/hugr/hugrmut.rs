//! Base HUGR builder providing low-level building blocks.

use std::ops::Range;

use portgraph::SecondaryMap;

use crate::hugr::{Direction, HugrError, Node};
use crate::ops::OpType;
use crate::{Hugr, Port};

/// Functions for low-level building of a HUGR. (Or, in the future, a subregion thereof)
pub(crate) trait HugrMut: AsRef<Hugr> + AsMut<Hugr> {
    /// Add a node to the graph.
    fn add_op(&mut self, op: impl Into<OpType>) -> Node;

    /// Remove a node from the graph.
    ///
    /// # Panics
    ///
    /// Panics if the node is the root node.
    fn remove_op(&mut self, node: Node) -> Result<(), HugrError>;

    /// Remove a node from the graph
    fn remove_node(&mut self, node: Node) -> Result<(), HugrError>;

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

    /// Add a node to the graph as the previous sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Errors
    ///
    ///  - If the sibling node does not have a parent.
    ///  - If the attachment would introduce a cycle.
    fn add_op_before(&mut self, sibling: Node, op: impl Into<OpType>) -> Result<Node, HugrError>;

    /// Add a node to the graph as the next sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Errors
    ///
    ///  - If the sibling node does not have a parent.
    ///  - If the attachment would introduce a cycle.
    fn add_op_after(&mut self, sibling: Node, op: impl Into<OpType>) -> Result<Node, HugrError>;

    /// Replace the OpType at node and return the old OpType.
    /// In general this invalidates the ports, which may need to be resized to
    /// match the OpType signature.
    fn replace_op(&mut self, node: Node, op: impl Into<OpType>) -> OpType;
}

impl HugrMut for Hugr {
    fn add_op(&mut self, op: impl Into<OpType>) -> Node {
        let op: OpType = op.into();
        let node = self.graph.add_node(op.input_count(), op.output_count());
        self.op_types[node] = op;
        node.into()
    }

    fn remove_op(&mut self, node: Node) -> Result<(), HugrError> {
        if node.index == self.root {
            // TODO: Add a HugrMutError ?
            panic!("cannot remove root node");
        }
        self.remove_node(node)
    }

    fn remove_node(&mut self, node: Node) -> Result<(), HugrError> {
        self.hierarchy.remove(node.index);
        self.graph.remove_node(node.index);
        self.op_types.remove(node.index);
        Ok(())
    }

    fn connect(
        &mut self,
        src: Node,
        src_port: usize,
        dst: Node,
        dst_port: usize,
    ) -> Result<(), HugrError> {
        self.graph
            .link_nodes(src.index, src_port, dst.index, dst_port)?;
        Ok(())
    }

    fn disconnect(&mut self, node: Node, port: Port) -> Result<(), HugrError> {
        let offset = port.offset;
        let port = self.graph.port_index(node.index, offset).ok_or(
            portgraph::LinkError::UnknownOffset {
                node: node.index,
                offset,
            },
        )?;
        self.graph.unlink_port(port);
        Ok(())
    }

    #[inline]
    fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
        self.graph
            .set_num_ports(node.index, incoming, outgoing, |_, _| {})
    }

    #[inline]
    fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize> {
        let mut incoming = self.graph.num_inputs(node.index);
        let mut outgoing = self.graph.num_outputs(node.index);
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
        self.graph
            .set_num_ports(node.index, incoming, outgoing, |_, _| {});
        range
    }

    fn set_parent(&mut self, node: Node, parent: Node) -> Result<(), HugrError> {
        self.hierarchy.detach(node.index);
        self.hierarchy.push_child(node.index, parent.index)?;
        Ok(())
    }

    fn move_after_sibling(&mut self, node: Node, after: Node) -> Result<(), HugrError> {
        self.hierarchy.detach(node.index);
        self.hierarchy.insert_after(node.index, after.index)?;
        Ok(())
    }

    fn move_before_sibling(&mut self, node: Node, before: Node) -> Result<(), HugrError> {
        self.hierarchy.detach(node.index);
        self.hierarchy.insert_before(node.index, before.index)?;
        Ok(())
    }

    fn add_op_with_parent(
        &mut self,
        parent: Node,
        op: impl Into<OpType>,
    ) -> Result<Node, HugrError> {
        let node = self.add_op(op.into());
        self.hierarchy.push_child(node.index, parent.index)?;
        Ok(node)
    }

    fn add_op_before(&mut self, sibling: Node, op: impl Into<OpType>) -> Result<Node, HugrError> {
        let node = self.add_op(op.into());
        self.hierarchy.insert_before(node.index, sibling.index)?;
        Ok(node)
    }

    fn add_op_after(&mut self, sibling: Node, op: impl Into<OpType>) -> Result<Node, HugrError> {
        let node = self.add_op(op.into());
        self.hierarchy.insert_after(node.index, sibling.index)?;
        Ok(node)
    }

    fn replace_op(&mut self, node: Node, op: impl Into<OpType>) -> OpType {
        let cur = self.op_types.get_mut(node.index);
        std::mem::replace(cur, op.into())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        hugr::HugrView,
        macros::type_row,
        ops::{self, dataflow::IOTrait, LeafOp},
        types::{ClassicType, Signature, SimpleType},
    };

    use super::*;

    const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());

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
                ops::Def {
                    name: "main".into(),
                    signature: Signature::new_df(type_row![NAT], type_row![NAT, NAT]),
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
                .add_op_with_parent(f, LeafOp::Noop(ClassicType::i64().into()))
                .unwrap();

            assert!(builder.connect(f_in, 0, noop, 0).is_ok());
            assert!(builder.connect(noop, 0, f_out, 0).is_ok());
            assert!(builder.connect(noop, 0, f_out, 1).is_ok());
        }

        // Finish the construction and create the HUGR
        builder.validate().unwrap();
    }
}
