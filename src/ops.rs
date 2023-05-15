//! The operation types for the HUGR.

pub mod controlflow;
pub mod custom;
pub mod dataflow;
pub mod handle;
pub mod leaf;
pub mod module;
pub mod validate;

use crate::types::{EdgeKind, Signature, SignatureDescription};

pub use controlflow::{BasicBlockOp, CaseOp, ControlFlowOp};
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
    /// A module region node - parent will be the Root (or the node itself is the Root).
    Module(ModuleOp),
    /// A basic block in a control flow graph - parent will be a CFG node.
    BasicBlock(BasicBlockOp),
    /// A branch in a dataflow graph - parent will be a Conditional node.
    Case(CaseOp),
    /// Nodes used inside dataflow containers
    /// (DFG, Conditional, TailLoop, def, BasicBlock).
    Dataflow(DataflowOp),
}

impl OpType {
    /// The name of the operation.
    pub fn name(&self) -> SmolStr {
        match self {
            OpType::Module(op) => op.name(),
            OpType::BasicBlock(op) => op.name(),
            OpType::Case(op) => op.name(),
            OpType::Dataflow(op) => op.name(),
        }
    }

    /// A human-readable description of the operation.
    pub fn description(&self) -> &str {
        match self {
            OpType::Module(op) => op.description(),
            OpType::BasicBlock(op) => op.description(),
            OpType::Case(op) => op.description(),
            OpType::Dataflow(op) => op.description(),
        }
    }

    /// The signature of the operation.
    ///
    /// Only dataflow operations have a non-empty signature.
    pub fn signature(&self) -> Signature {
        match self {
            OpType::Dataflow(op) => op.signature(),
            _ => Default::default(),
        }
    }

    /// Optional description of the ports in the signature.
    ///
    /// Only dataflow operations have a non-empty signature.
    pub fn signature_desc(&self) -> SignatureDescription {
        match self {
            OpType::Dataflow(op) => op.signature_desc(),
            _ => Default::default(),
        }
    }

    /// The edge kind for the inputs of the operation not described by the
    /// signature.
    ///
    /// If None, there will be no other input edges. Otherwise, all other input
    /// edges will be of that kind.
    pub fn other_inputs(&self) -> Option<EdgeKind> {
        match self {
            OpType::Module(op) => op.other_inputs(),
            OpType::Dataflow(op) => op.other_inputs(),
            OpType::BasicBlock(op) => op.other_edges(),
            OpType::Case(op) => op.other_edges(),
        }
    }

    /// The edge kind for the outputs of the operation not described by the
    /// signature.
    ///
    /// If None, there will be no other output edges. Otherwise, all other
    /// output edges will be of that kind.
    pub fn other_outputs(&self) -> Option<EdgeKind> {
        match self {
            OpType::Module(op) => op.other_outputs(),
            OpType::Dataflow(op) => op.other_outputs(),
            OpType::BasicBlock(op) => op.other_edges(),
            OpType::Case(op) => op.other_edges(),
        }
    }

    /// Returns the edge kind for the given port offset.
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
        Self::Dataflow(Default::default())
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
        Self::Dataflow(op.into())
    }
}

impl From<CaseOp> for OpType {
    fn from(op: CaseOp) -> Self {
        OpType::Case(op)
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
impl_try_from_optype!(CaseOp, OpType::Case(op), op);
impl_try_from_optype!(DataflowOp, OpType::Dataflow(op), op);
impl_try_from_optype!(
    ControlFlowOp,
    OpType::Dataflow(DataflowOp::ControlFlow { op }),
    op
);
impl_try_from_optype!(LeafOp, OpType::Dataflow(DataflowOp::Leaf { op }), op);
