pub mod controlflow;
pub mod custom;
pub mod dataflow;
pub mod leaf;
pub mod module;

use crate::types::{EdgeKind, Signature, SignatureDescription};

pub use controlflow::BasicBlockOp;
pub use custom::{CustomOp, OpDef, OpaqueOp};
pub use dataflow::DataflowOp;
pub use leaf::LeafOp;
pub use module::{ConstValue, ModuleOp};
use smol_str::SmolStr;

/// The concrete operation types for a node in the HUGR.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum OpType {
    /// A module region node - parent will be the Root (or the node itself is the Root)
    Module(ModuleOp),
    /// A basic block in a control flow graph - parent will be a kappa node
    BasicBlock(BasicBlockOp),
    /// A function manipulation node - parent will be a dataflow-graph container
    /// (delta, gamma, theta, def, beta)
    Function(DataflowOp),
}

impl OpType {
    /// The name of the operation.
    pub fn name(&self) -> SmolStr {
        match self {
            OpType::Module(op) => op.name(),
            OpType::BasicBlock(op) => op.name(),
            OpType::Function(op) => op.name(),
        }
    }

    /// The description of the operation.
    pub fn description(&self) -> &str {
        match self {
            OpType::Module(op) => op.description(),
            OpType::BasicBlock(op) => op.description(),
            OpType::Function(op) => op.description(),
        }
    }

    /// The signature of the operation.
    ///
    /// Only dataflow operations have a non-empty signature.
    pub fn signature(&self) -> Signature {
        match self {
            OpType::Function(op) => op.signature(),
            _ => Default::default(),
        }
    }

    /// Optional description of the ports in the signature.
    ///
    /// Only dataflow operations have a non-empty signature.
    pub fn signature_desc(&self) -> SignatureDescription {
        match self {
            OpType::Function(op) => op.signature_desc(),
            _ => Default::default(),
        }
    }

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
            OpType::Module(op) => op.other_outputs(),
            OpType::Function(op) => op.other_outputs(),
            OpType::BasicBlock(op) => op.other_edges(),
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
    T: Into<DataflowOp>,
{
    fn from(op: T) -> Self {
        Self::Function(op.into())
    }
}
