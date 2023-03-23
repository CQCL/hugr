pub mod controlflow;
pub mod custom;
pub mod function;
pub mod leaf;
pub mod module;

use crate::types::{Signature, SignatureDescription};

pub use controlflow::ControlFlowOp;
pub use custom::{OpaqueOp, CustomOp, OpDef};
pub use function::FunctionOp;
pub use leaf::LeafOp;
pub use module::{ConstValue, ModuleOp};

/// A generic node operation
pub trait Op {
    /// The name of the operation.
    fn name(&self) -> &str;
    /// The description of the operation.
    fn description(&self) -> &str {
        ""
    }
    /// The signature of the operation.
    ///
    /// TODO: Return a reference? It'll need some lazy_statics to make it work.
    fn signature(&self) -> Signature;
    /// Optional description of the ports in the signature.
    ///
    /// TODO: Implement where possible
    fn signature_desc(&self) -> Option<&SignatureDescription> {
        None
    }
}

/// The concrete operation types for a node in the HUGR.
///
/// TODO: Flatten the enum? It improves efficiency, but makes it harder to read.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum OpType {
    /// A module region node.
    Module(ModuleOp),
    /// A control flow node
    ControlFlow(ControlFlowOp),
    /// A function manipulation node
    Function(FunctionOp),
    /// A quantum circuit operation
    Leaf(LeafOp),
}

impl Op for OpType {
    fn name(&self) -> &str {
        match self {
            OpType::Module(op) => op.name(),
            OpType::ControlFlow(op) => op.name(),
            OpType::Function(op) => op.name(),
            OpType::Leaf(op) => op.name(),
        }
    }

    fn signature(&self) -> Signature {
        match self {
            OpType::Module(op) => op.signature(),
            OpType::ControlFlow(op) => op.signature(),
            OpType::Function(op) => op.signature(),
            OpType::Leaf(op) => op.signature(),
        }
    }
}

impl Default for OpType {
    fn default() -> Self {
        Self::Leaf(Default::default())
    }
}
