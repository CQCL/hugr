pub mod controlflow;
pub mod custom;
pub mod dataflow;
pub mod leaf;
pub mod module;
pub mod validate;

use crate::types::{EdgeKind, Signature, SignatureDescription};

pub use controlflow::{BasicBlockOp, BranchOp, ControlFlowOp};
pub use custom::{CustomOp, OpDef, OpaqueOp};
pub use dataflow::DataflowOp;
pub use leaf::LeafOp;
pub use module::{ConstValue, ModuleOp};
use portgraph::{Direction, PortOffset};
use smol_str::SmolStr;

/// The concrete operation types for a node in the HUGR.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum OpType {
    /// A module region node - parent will be the Root (or the node itself is the Root)
    Module(ModuleOp),
    /// A basic block in a control flow graph - parent will be a kappa node
    BasicBlock(BasicBlockOp),
    /// A branch in a dataflow graph - parent will be a gamma node
    Branch(BranchOp),
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
            OpType::Branch(op) => op.name(),
            OpType::Function(op) => op.name(),
        }
    }

    /// The description of the operation.
    pub fn description(&self) -> &str {
        match self {
            OpType::Module(op) => op.description(),
            OpType::BasicBlock(op) => op.description(),
            OpType::Branch(op) => op.description(),
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
            OpType::Branch(op) => op.other_edges(),
        }
    }

    /// Like "other_inputs" but describes any other output edges
    pub fn other_outputs(&self) -> Option<EdgeKind> {
        match self {
            OpType::Module(op) => op.other_outputs(),
            OpType::Function(op) => op.other_outputs(),
            OpType::BasicBlock(op) => op.other_edges(),
            OpType::Branch(op) => op.other_edges(),
        }
    }

    /// Returns the edge kind for the given port offset
    pub fn port_kind(&self, offset: PortOffset) -> Option<EdgeKind> {
        let signature = self.signature();
        if let Some(port_kind) = signature.get(offset) {
            Some(port_kind)
        } else if offset.direction() == Direction::Incoming {
            self.other_inputs()
        } else {
            self.other_outputs()
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

impl From<BranchOp> for OpType {
    fn from(op: BranchOp) -> Self {
        OpType::Branch(op)
    }
}

/// Implementations of TryFrom for OpType and &'a OpType for each variant.
macro_rules! impl_try_from_optype {
    ($target:ident, $matcher:pat, $unpack:expr) => {
        impl TryFrom<OpType> for $target {
            type Error = ();

            fn try_from(op: OpType) -> Result<Self, Self::Error> {
                match op {
                    $matcher => Ok($unpack),
                    _ => Err(()),
                }
            }
        }

        impl<'a> TryFrom<&'a OpType> for &'a $target {
            type Error = ();

            fn try_from(op: &'a OpType) -> Result<Self, Self::Error> {
                match op {
                    $matcher => Ok($unpack),
                    _ => Err(()),
                }
            }
        }
    };
}
impl_try_from_optype!(ModuleOp, OpType::Module(op), op);
impl_try_from_optype!(BasicBlockOp, OpType::BasicBlock(op), op);
impl_try_from_optype!(BranchOp, OpType::Branch(op), op);
impl_try_from_optype!(DataflowOp, OpType::Function(op), op);
impl_try_from_optype!(
    ControlFlowOp,
    OpType::Function(DataflowOp::ControlFlow { op }),
    op
);
impl_try_from_optype!(LeafOp, OpType::Function(DataflowOp::Leaf { op }), op);
