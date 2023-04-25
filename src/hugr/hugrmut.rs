//! Base HUGR builder providing low-level building blocks.

use portgraph::NodeIndex;
use thiserror::Error;

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

    /// Connect two nodes at the given ports.
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

    pub fn set_num_ports(&mut self, n: NodeIndex, incoming: usize, outgoing: usize) {
        self.hugr
            .graph
            .set_num_ports(n, incoming, outgoing, |_, _| {})
    }

    /// Sets the parent of a node.
    ///
    /// The node becomes the parent's last child.
    pub fn set_parent(&mut self, node: NodeIndex, parent: NodeIndex) -> Result<(), HugrError> {
        self.hugr.hierarchy.push_child(node, parent)?;
        Ok(())
    }

    /// Add a node to the graph with a parent in the hierarchy.
    pub fn add_op_with_parent(
        &mut self,
        parent: NodeIndex,
        op: impl Into<OpType>,
    ) -> Result<NodeIndex, HugrError> {
        let node = self.add_op(op.into());
        self.set_parent(node, parent)?;
        Ok(node)
    }

    /// Add a node to the graph with a parent in the hierarchy.
    pub fn add_op_before(
        &mut self,
        sibling: NodeIndex,
        op: impl Into<OpType>,
    ) -> Result<NodeIndex, HugrError> {
        let node = self.add_op(op.into());
        self.hugr.hierarchy.insert_before(node, sibling)?;
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
