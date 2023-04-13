pub mod controlflow;
pub mod custom;
pub mod function;
pub mod leaf;
pub mod module;

use crate::types::{EdgeKind, Signature, SignatureDescription};

pub use controlflow::BasicBlockOp;
pub use custom::{CustomOp, OpDef, OpaqueOp};
pub use function::FunctionOp;
pub use leaf::LeafOp;
pub use module::{ConstValue, ModuleOp};
use smol_str::SmolStr;

/// A generic node operation
pub trait Op {
    /// The name of the operation.
    fn name(&self) -> SmolStr;
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
    fn signature_desc(&self) -> Option<SignatureDescription> {
        None
    }
}

/// The concrete operation types for a node in the HUGR.
///
/// TODO: Flatten the enum? It improves efficiency, but makes it harder to read.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum OpType {
    /// A module region node - parent will be the Root (or the node itself is the Root)
    Module(ModuleOp),
    /// A basic block in a control flow graph - parent will be a kappa node
    BasicBlock(BasicBlockOp),
    /// A function manipulation node - parent will be a dataflow-graph container
    /// (delta, gamma, theta, def, beta)
    Function(FunctionOp),
}

impl OpType {
    /// If None, there will be no other input edges.
    /// Otherwise, all other input edges will be of that kind.
    pub fn other_inputs(&self) -> Option<EdgeKind> {
        match self {
            OpType::Module(op) => op.other_inputs(),
            OpType::Function(op) => op.other_inputs(),
            OpType::BasicBlock(op) => op.other_edges(),
        }
    }

    /// Like "other_inputs" but describes any other output edges
    pub fn other_outputs(&self) -> Option<EdgeKind> {
        match self {
            OpType::Module(op) => op.other_inputs(),
            OpType::Function(op) => op.other_outputs(),
            OpType::BasicBlock(op) => op.other_edges(),
        }
    }
}

impl Op for OpType {
    fn name(&self) -> SmolStr {
        match self {
            OpType::Module(op) => op.name(),
            OpType::BasicBlock(op) => op.name(),
            OpType::Function(op) => op.name(),
        }
    }

    fn signature(&self) -> Signature {
        match self {
            OpType::Module(op) => op.signature(),
            OpType::BasicBlock(op) => op.signature(),
            OpType::Function(op) => op.signature(),
        }
    }
}

impl Default for OpType {
    fn default() -> Self {
        Self::Function(Default::default())
    }
}

impl From<ModuleOp> for OpType {
    fn from(op: ModuleOp) -> Self {
        Self::Module(op)
    }
}

impl From<BasicBlockOp> for OpType {
    fn from(op: BasicBlockOp) -> Self {
        Self::BasicBlock(op)
    }
}

impl<T> From<T> for OpType
where
    T: Into<FunctionOp>,
{
    fn from(op: T) -> Self {
        Self::Function(op.into())
    }
}
