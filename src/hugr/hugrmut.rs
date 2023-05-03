//! Base HUGR builder providing low-level building blocks.

use std::ops::Range;

use itertools::Itertools;
use portgraph::{Direction, NodeIndex, PortOffset};

use crate::{
    hugr::{HugrError, ValidationError},
    ops::OpType,
    Hugr,
};

/// A low-level builder for a HUGR.
#[derive(Clone, Debug, Default)]
pub struct HugrMut {
    /// The partial HUGR being built.
    hugr: Hugr,
}

impl HugrMut {
    /// Initialize a new builder.
    pub fn new() -> Self {
        Default::default()
    }

    /// Return index of HUGR root node.
    #[inline]
    pub fn root(&self) -> NodeIndex {
        self.hugr.root
    }

    /// Add a node to the graph.
    pub fn add_op(&mut self, op: impl Into<OpType>) -> NodeIndex {
        let op: OpType = op.into();
        let sig = op.signature();
        let node = self.hugr.graph.add_node(
            sig.input.len() + sig.const_input.iter().count(),
            sig.output.len(),
        );
        self.hugr.op_types[node] = op;
        node
    }

    /// Remove a node from the graph.
    ///
    /// # Panics
    ///
    /// Panics if the node is the root node.
    pub fn remove_op(&mut self, node: NodeIndex) -> Result<(), HugrError> {
        if node == self.hugr.root {
            // TODO: Add a HugrMutError ?
            panic!("cannot remove root node");
        }
        self.hugr.hierarchy.detach(node);
        self.hugr.graph.remove_node(node);
        Ok(())
    }

    /// Connect two nodes at the given ports.
    ///
    /// The port must have already been created. See [`add_ports`] and [`set_num_ports`].
    ///
    /// [`add_ports`]: #method.add_ports
    /// [`set_num_ports`]: #method.set_num_ports
    pub fn connect(
        &mut self,
        src: NodeIndex,
        src_port: usize,
        dst: NodeIndex,
        dst_port: usize,
    ) -> Result<(), HugrError> {
        self.hugr.graph.link_nodes(src, src_port, dst, dst_port)?;
        Ok(())
    }

    /// Disconnects the given ports.
    ///
    /// The port is left in place.
    pub fn disconnect(
        &mut self,
        node: NodeIndex,
        port: usize,
        direction: Direction,
    ) -> Result<(), HugrError> {
        let port = self
            .hugr
            .graph
            .port_index(node, PortOffset::new(direction, port))
            .ok_or(portgraph::LinkError::UnknownOffset {
                node,
                offset: PortOffset::new_outgoing(port),
            })?;
        self.hugr.graph.unlink_port(port);
        Ok(())
    }

    /// Adds a non-dataflow edge between two nodes, allocating new ports for the
    /// connection. The kind is given by the operation's
    /// [`OpType::other_inputs`] or [`OpType::other_outputs`].
    ///
    /// Returns the offsets of the new input and output ports, or an error if
    /// the connection failed.
    ///
    /// [`OpType::other_inputs`]: crate::ops::OpType::other_inputs
    /// [`OpType::other_outputs`]: crate::ops::OpType::other_outputs
    pub fn add_other_wire(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
    ) -> Result<(usize, usize), HugrError> {
        let src_port: usize = self.add_ports(src, Direction::Outgoing, 1).collect_vec()[0];
        let dst_port: usize = self.add_ports(dst, Direction::Incoming, 1).collect_vec()[0];
        self.connect(src, src_port, dst, dst_port)?;
        Ok((src_port, dst_port))
    }

    /// Set the number of ports on a node. This may invalidate the node's `PortIndex`.
    #[inline]
    pub fn set_num_ports(&mut self, node: NodeIndex, incoming: usize, outgoing: usize) {
        self.hugr
            .graph
            .set_num_ports(node, incoming, outgoing, |_, _| {})
    }

    /// Alter the number of ports on a node and returns a range with the new
    /// port offsets, if any. This may invalidate the node's `PortIndex`.
    ///
    /// The `direction` parameter specifies whether to add ports to the incoming
    /// or outgoing list.
    #[inline]
    pub fn add_ports(
        &mut self,
        node: NodeIndex,
        direction: Direction,
        amount: isize,
    ) -> Range<usize> {
        let mut incoming = self.hugr.graph.num_inputs(node);
        let mut outgoing = self.hugr.graph.num_outputs(node);
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
        self.hugr
            .graph
            .set_num_ports(node, incoming, outgoing, |_, _| {});
        range
    }

    /// Sets the parent of a node.
    ///
    /// The node becomes the parent's last child.
    pub fn set_parent(&mut self, node: NodeIndex, parent: NodeIndex) -> Result<(), HugrError> {
        self.hugr.hierarchy.detach(node);
        self.hugr.hierarchy.push_child(node, parent)?;
        Ok(())
    }

    /// Move a node in the hierarchy to be the subsequent sibling of another
    /// node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// The node becomes the parent's last child.
    pub fn move_after_sibling(
        &mut self,
        node: NodeIndex,
        after: NodeIndex,
    ) -> Result<(), HugrError> {
        self.hugr.hierarchy.detach(node);
        self.hugr.hierarchy.insert_after(node, after)?;
        Ok(())
    }

    /// Move a node in the hierarchy to be the prior sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// The node becomes the parent's last child.
    pub fn move_before_sibling(
        &mut self,
        node: NodeIndex,
        before: NodeIndex,
    ) -> Result<(), HugrError> {
        self.hugr.hierarchy.detach(node);
        self.hugr.hierarchy.insert_before(node, before)?;
        Ok(())
    }

    /// Add a node to the graph with a parent in the hierarchy.
    ///
    /// The node becomes the parent's last child.
    pub fn add_op_with_parent(
        &mut self,
        parent: NodeIndex,
        op: impl Into<OpType>,
    ) -> Result<NodeIndex, HugrError> {
        let node = self.add_op(op.into());
        self.hugr.hierarchy.push_child(node, parent)?;
        Ok(node)
    }

    /// Add a node to the graph as the previous sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Errors
    ///
    ///  - If the sibling node does not have a parent.
    ///  - If the attachment would introduce a cycle.
    pub fn add_op_before(
        &mut self,
        sibling: NodeIndex,
        op: impl Into<OpType>,
    ) -> Result<NodeIndex, HugrError> {
        let node = self.add_op(op.into());
        self.hugr.hierarchy.insert_before(node, sibling)?;
        Ok(node)
    }

    /// Add a node to the graph as the next sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Errors
    ///
    ///  - If the sibling node does not have a parent.
    ///  - If the attachment would introduce a cycle.
    pub fn add_op_after(
        &mut self,
        sibling: NodeIndex,
        op: impl Into<OpType>,
    ) -> Result<NodeIndex, HugrError> {
        let node = self.add_op(op.into());
        self.hugr.hierarchy.insert_after(node, sibling)?;
        Ok(node)
    }

    /// Build the HUGR, returning an error if the graph is not valid.
    pub fn finish(self) -> Result<Hugr, ValidationError> {
        let hugr = self.hugr;

        hugr.validate()?;

        Ok(hugr)
    }

    // Immutable reference to HUGR being built
    #[inline]
    pub fn hugr(&self) -> &Hugr {
        &self.hugr
    }

    /// Replace the OpType at node and return the old OpType.
    /// In general this invalidates the ports, which may need to be resized to
    /// match the OpType signature
    pub fn replace_op(&mut self, node: NodeIndex, op: impl Into<OpType>) -> OpType {
        let cur = self.hugr.op_types.get_mut(node);
        std::mem::replace(cur, op.into())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        macros::type_row,
        ops::{DataflowOp, LeafOp, ModuleOp},
        types::{ClassicType, Signature, SimpleType},
    };

    use super::*;

    const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());

    #[test]
    fn simple_function() {
        // Starts an empty builder
        let mut builder = HugrMut::new();

        // Create the root module definition
        let module: NodeIndex = builder.root();

        // Start a main function with two nat inputs.
        //
        // `add_op` is equivalent to `add_root_op` followed by `set_parent`
        let f: NodeIndex = builder
            .add_op_with_parent(
                module,
                ModuleOp::Def {
                    signature: Signature::new_df(type_row![NAT], type_row![NAT, NAT]),
                },
            )
            .expect("Failed to add function definition node");

        {
            let f_in = builder
                .add_op_with_parent(
                    f,
                    DataflowOp::Input {
                        types: type_row![NAT],
                    },
                )
                .unwrap();
            let copy = builder
                .add_op_with_parent(
                    f,
                    LeafOp::Copy {
                        n_copies: 2,
                        typ: ClassicType::i64(),
                    },
                )
                .unwrap();
            let f_out = builder
                .add_op_with_parent(
                    f,
                    DataflowOp::Output {
                        types: type_row![NAT, NAT],
                    },
                )
                .unwrap();

            assert!(builder.connect(f_in, 0, copy, 0).is_ok());
            assert!(builder.connect(copy, 0, f_out, 0).is_ok());
            assert!(builder.connect(copy, 1, f_out, 1).is_ok());
        }

        // Finish the construction and create the HUGR
        let hugr: Result<Hugr, ValidationError> = builder.finish();
        assert_eq!(hugr.err(), None);
    }
}
