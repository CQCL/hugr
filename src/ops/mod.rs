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

/// A trait defining validity properties of an operation type.
pub trait OpTypeValidator {
    /// Returns whether the given operation is allowed as a parent.
    fn is_valid_parent(&self, parent: &OpType) -> bool;

    /// Whether the operation can have children
    fn is_container(&self) -> bool {
        false
    }

    /// Whether the operation must have children
    fn requires_children(&self) -> bool {
        self.is_container()
    }

    /// Whether the operation contains dataflow children
    fn is_df_container(&self) -> bool {
        false
    }

    /// A restriction on the operation type of the first child
    fn first_child_valid(&self, _child: OpType) -> bool {
        true
    }

    /// A restriction on the operation type of the last child
    fn last_child_valid(&self, _child: OpType) -> bool {
        true
    }

    /// Whether the children must form a DAG (no cycles)
    fn require_dag(&self) -> bool {
        false
    }

    /// Whether the first/last child must dominate/post-dominate all other children
    fn require_dominators(&self) -> bool {
        false
    }
}

impl OpTypeValidator for OpType {
    fn is_valid_parent(&self, parent: &OpType) -> bool {
        match self {
            OpType::Module(op) => op.is_valid_parent(parent),
            OpType::Function(op) => op.is_valid_parent(parent),
            OpType::BasicBlock(op) => op.is_valid_parent(parent),
        }
    }

    fn is_container(&self) -> bool {
        match self {
            OpType::Module(op) => op.is_container(),
            OpType::Function(op) => op.is_container(),
            OpType::BasicBlock(op) => op.is_container(),
        }
    }

    fn requires_children(&self) -> bool {
        match self {
            OpType::Module(op) => op.requires_children(),
            OpType::Function(op) => op.requires_children(),
            OpType::BasicBlock(op) => op.requires_children(),
        }
    }

    fn is_df_container(&self) -> bool {
        match self {
            OpType::Module(op) => op.is_df_container(),
            OpType::Function(op) => op.is_df_container(),
            OpType::BasicBlock(op) => op.is_df_container(),
        }
    }

    fn first_child_valid(&self, child: OpType) -> bool {
        match self {
            OpType::Module(op) => op.first_child_valid(child),
            OpType::Function(op) => op.first_child_valid(child),
            OpType::BasicBlock(op) => op.first_child_valid(child),
        }
    }

    fn last_child_valid(&self, child: OpType) -> bool {
        match self {
            OpType::Module(op) => op.last_child_valid(child),
            OpType::Function(op) => op.last_child_valid(child),
            OpType::BasicBlock(op) => op.last_child_valid(child),
        }
    }

    fn require_dag(&self) -> bool {
        match self {
            OpType::Module(op) => op.require_dag(),
            OpType::Function(op) => op.require_dag(),
            OpType::BasicBlock(op) => op.require_dag(),
        }
    }

    fn require_dominators(&self) -> bool {
        match self {
            OpType::Module(op) => op.require_dominators(),
            OpType::Function(op) => op.require_dominators(),
            OpType::BasicBlock(op) => op.require_dominators(),
        }
    }
}
