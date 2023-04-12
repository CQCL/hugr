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
pub struct BaseBuilder {
    /// The partial HUGR being built.
    hugr: Hugr,
}

impl BaseBuilder {
    /// Initialize a new builder.
    pub fn new() -> Self {
        Default::default()
    }

    /// Return index of HUGR root node.
    pub fn root(&self) -> NodeIndex {
        self.hugr.root()
    }

    /// Add a node to the graph with a parent in the hierarchy.
    pub fn add_op(
        &mut self,
        parent: NodeIndex,
        op: impl Into<OpType>,
    ) -> Result<NodeIndex, HugrError> {
        let node = self.hugr.add_node(op.into());
        self.hugr.set_parent(node, parent)?;
        Ok(node)
    }

    /// Set the parent of a node.
    pub fn set_parent(&mut self, node: NodeIndex, parent: NodeIndex) -> Result<(), HugrError> {
        self.hugr.set_parent(node, parent)?;
        Ok(())
    }

    /// Connect two nodes at the given ports.
    pub fn connect(
        &mut self,
        src: NodeIndex,
        src_port: usize,
        dst: NodeIndex,
        dst_port: usize,
    ) -> Result<(), HugrError> {
        self.hugr.connect(src, src_port, dst, dst_port)
    }

    /// Build the HUGR, returning an error if the graph is not valid.
    pub fn finish(self) -> Result<Hugr, BuildError> {
        let hugr = self.hugr;

        hugr.validate()?;

        Ok(hugr)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BuildError {
    /// The hierarchy has to many root nodes.
    #[error("The hierarchy has too many roots: {roots:?}.")]
    TooManyRoots { roots: Vec<NodeIndex> },
    /// The constructed HUGR is invalid.
    #[error("The constructed HUGR is invalid: {0:?}.")]
    InvalidHUGR(#[from] ValidationError),
}

#[cfg(test)]
mod test {
    use crate::{
        macros::type_row,
        ops::{FunctionOp, LeafOp, ModuleOp},
        types::{ClassicType, Signature, SimpleType},
    };

    use super::*;

    const NAT: SimpleType = SimpleType::Classic(ClassicType::Nat);

    #[test]
    fn simple_function() {
        // Starts an empty builder
        let mut builder = BaseBuilder::new();

        // Create the root module definition
        let module: NodeIndex = builder.root();

        // Start a main function with two nat inputs.
        //
        // `add_op` is equivalent to `add_root_op` followed by `set_parent`
        let f: NodeIndex = builder
            .add_op(
                module,
                ModuleOp::Def {
                    signature: Signature::new_df(type_row![NAT], type_row![NAT, NAT]),
                },
            )
            .expect("Failed to add function definition node");

        {
            let f_in = builder
                .add_op(
                    f,
                    FunctionOp::Input {
                        types: type_row![NAT],
                    },
                )
                .unwrap();
            let copy = builder
                .add_op(
                    f,
                    LeafOp::Copy {
                        n_copies: 2,
                        typ: ClassicType::Nat,
                    },
                )
                .unwrap();
            let f_out = builder
                .add_op(
                    f,
                    FunctionOp::Output {
                        types: type_row![NAT, NAT],
                    },
                )
                .unwrap();

            assert!(builder.connect(f_in, 0, copy, 0).is_ok());
            assert!(builder.connect(copy, 0, f_out, 0).is_ok());
            assert!(builder.connect(copy, 1, f_out, 1).is_ok());
        }

        // Finish the construction and create the HUGR
        let hugr: Result<Hugr, BuildError> = builder.finish();
        assert!(hugr.is_ok());
    }
}
