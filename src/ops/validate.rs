//! Methods for validating hugr nodes according to their operation type.

use super::{BasicBlockOp, ControlFlowOp, DataflowOp, ModuleOp, OpType};

/// A trait defining validity properties of an operation type.
pub trait OpTypeValidator {
    /// Returns the set of valid parent operation types.
    fn valid_parents(&self) -> OpTypeSet;

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
    fn validate_first_child(&self, child: &OpType) -> bool {
        let _ = child;
        true
    }

    /// A restriction on the operation type of the last child
    fn validate_last_child(&self, child: &OpType) -> bool {
        let _ = child;
        true
    }

    /// Validates the complete set of children
    fn validate_children<'a>(&self, children: impl DoubleEndedIterator<Item = &'a OpType>) -> bool {
        // TODO: Use this to validate the children of branches and loops
        // TODO: Probably merge the first/last child validation into this, defining a custom ChildrenValidationError to throw
        let _ = children;
        true
    }

    /// Whether the children must form a DAG (no cycles)
    fn require_dag(&self) -> bool {
        false
    }
}

/// Sets of operation kinds.
///
/// Used to validate the allowed parent operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpTypeSet {
    /// No valid operation types.
    None,
    /// Only the module root operation.
    ModuleRoot,
    /// Any dataflow container operation.
    DataflowContainers,
    /// A control flow container operation.
    CfgNode,
}

impl OpTypeSet {
    /// Returns true if the set contains the given operation type.
    pub fn contains(&self, optype: &OpType) -> bool {
        match self {
            OpTypeSet::None => false,
            OpTypeSet::ModuleRoot => matches!(optype, OpType::Module(ModuleOp::Root)),
            OpTypeSet::DataflowContainers => optype.is_df_container(),
            OpTypeSet::CfgNode => matches!(
                optype,
                OpType::Function(DataflowOp::ControlFlow {
                    op: ControlFlowOp::CFG { .. }
                })
            ),
        }
    }

    /// Returns a user-friendly description of the set.
    pub fn set_description(&self) -> String {
        match self {
            OpTypeSet::None => "None",
            OpTypeSet::ModuleRoot => "ModuleOp::Root",
            OpTypeSet::DataflowContainers => "Dataflow containers",
            OpTypeSet::CfgNode => "ControlFlowOp::CFG",
        }
        .into()
    }
}

impl OpTypeValidator for OpType {
    fn valid_parents(&self) -> OpTypeSet {
        match self {
            OpType::Module(op) => op.valid_parents(),
            OpType::Function(op) => op.valid_parents(),
            OpType::BasicBlock(op) => op.valid_parents(),
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

    fn validate_first_child(&self, child: &OpType) -> bool {
        match self {
            OpType::Module(op) => op.validate_first_child(child),
            OpType::Function(op) => op.validate_first_child(child),
            OpType::BasicBlock(op) => op.validate_first_child(child),
        }
    }

    fn validate_last_child(&self, child: &OpType) -> bool {
        match self {
            OpType::Module(op) => op.validate_last_child(child),
            OpType::Function(op) => op.validate_last_child(child),
            OpType::BasicBlock(op) => op.validate_last_child(child),
        }
    }

    fn require_dag(&self) -> bool {
        match self {
            OpType::Module(op) => op.require_dag(),
            OpType::Function(op) => op.require_dag(),
            OpType::BasicBlock(op) => op.require_dag(),
        }
    }
}

impl OpTypeValidator for ModuleOp {
    fn valid_parents(&self) -> OpTypeSet {
        match self {
            ModuleOp::Root => OpTypeSet::None,
            _ => OpTypeSet::ModuleRoot,
        }
    }

    fn is_container(&self) -> bool {
        matches!(self, ModuleOp::Root | ModuleOp::Def { .. })
    }

    fn requires_children(&self) -> bool {
        // Allow empty modules roots for non-runnable hugrs
        matches!(self, ModuleOp::Def { .. })
    }

    fn is_df_container(&self) -> bool {
        matches!(self, ModuleOp::Def { .. })
    }

    fn validate_first_child(&self, child: &OpType) -> bool {
        match self {
            ModuleOp::Root { .. } => matches!(child, OpType::Module(ModuleOp::Def { .. })),
            ModuleOp::Def { .. } => matches!(child, OpType::Function(DataflowOp::Input { .. })),
            _ => true,
        }
    }

    fn validate_last_child(&self, child: &OpType) -> bool {
        match self {
            ModuleOp::Def { .. } => matches!(child, OpType::Function(DataflowOp::Output { .. })),
            _ => true,
        }
    }

    fn require_dag(&self) -> bool {
        matches!(self, ModuleOp::Def { .. })
    }
}

impl OpTypeValidator for ControlFlowOp {
    // TODO: CFG nodes require checking the internal signature of pairs of
    // BasicBlocks connected by ControlFlow edges. This is not currently
    // implemented, and should probably go outside of the OpTypeValidator trait.

    fn valid_parents(&self) -> OpTypeSet {
        OpTypeSet::DataflowContainers
    }

    fn is_container(&self) -> bool {
        true
    }

    fn is_df_container(&self) -> bool {
        matches!(
            self,
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. }
        )
    }

    fn validate_first_child(&self, child: &OpType) -> bool {
        // TODO: check signatures
        match self {
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. } => {
                matches!(child, OpType::Function(DataflowOp::Input { .. }))
            }
            ControlFlowOp::CFG { .. } => matches!(child, OpType::BasicBlock(_)),
        }
    }

    fn validate_last_child(&self, child: &OpType) -> bool {
        // TODO: check signatures
        match self {
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. } => {
                matches!(child, OpType::Function(DataflowOp::Output { .. }))
            }
            ControlFlowOp::CFG { .. } => matches!(child, OpType::BasicBlock(_)),
        }
    }

    fn require_dag(&self) -> bool {
        matches!(
            self,
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. }
        )
    }
}

impl OpTypeValidator for BasicBlockOp {
    fn valid_parents(&self) -> OpTypeSet {
        OpTypeSet::CfgNode
    }

    fn is_container(&self) -> bool {
        true
    }

    fn is_df_container(&self) -> bool {
        true
    }

    fn require_dag(&self) -> bool {
        true
    }
}

impl OpTypeValidator for DataflowOp {
    fn valid_parents(&self) -> OpTypeSet {
        OpTypeSet::DataflowContainers
    }

    fn is_container(&self) -> bool {
        match self {
            DataflowOp::ControlFlow { op } => op.is_df_container(),
            DataflowOp::Nested { .. } => true,
            _ => false,
        }
    }

    fn is_df_container(&self) -> bool {
        match self {
            DataflowOp::ControlFlow { op } => op.is_df_container(),
            DataflowOp::Nested { .. } => true,
            _ => false,
        }
    }

    fn validate_first_child(&self, child: &OpType) -> bool {
        match self {
            DataflowOp::ControlFlow { op } => op.validate_first_child(child),
            DataflowOp::Nested { .. } => {
                matches!(child, OpType::Function(DataflowOp::Input { .. }))
            }
            _ => true,
        }
    }

    fn require_dag(&self) -> bool {
        match self {
            DataflowOp::ControlFlow { op } => op.require_dag(),
            DataflowOp::Nested { .. } => true,
            _ => false,
        }
    }
}
