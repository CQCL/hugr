//! Methods for validating hugr nodes according to their operation type.

use thiserror::Error;

use super::{BasicBlockOp, ControlFlowOp, DataflowOp, ModuleOp, OpType};

/// A set of property flags required for an operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpValidityFlags {
    /// The set of valid parent operation types
    pub allowed_parents: ValidParentSet,
    /// Whether the operation can have children
    pub is_container: bool,
    /// Whether the operation contains dataflow children
    pub is_df_container: bool,
    /// Whether the operation must have children
    pub requires_children: bool,
    /// Whether the children must form a DAG (no cycles)
    pub require_dag: bool,
}

impl Default for OpValidityFlags {
    fn default() -> Self {
        Self {
            allowed_parents: ValidParentSet::None,
            is_container: false,
            is_df_container: false,
            requires_children: false,
            require_dag: false,
        }
    }
}

/// Sets of operation kinds.
///
/// Used to validate the allowed parent operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ValidParentSet {
    /// No valid operation types.
    #[default]
    None,
    /// Only the module root operation.
    ModuleRoot,
    /// Any dataflow container operation.
    DataflowContainers,
    /// A control flow container operation.
    CfgNode,
}

impl ValidParentSet {
    /// Returns true if the set contains the given operation type.
    pub fn contains(&self, optype: &OpType) -> bool {
        match self {
            ValidParentSet::None => false,
            ValidParentSet::ModuleRoot => matches!(optype, OpType::Module(ModuleOp::Root)),
            ValidParentSet::DataflowContainers => optype.validity_flags().is_df_container,
            ValidParentSet::CfgNode => matches!(
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
            ValidParentSet::None => "None",
            ValidParentSet::ModuleRoot => "ModuleOp::Root",
            ValidParentSet::DataflowContainers => "Dataflow containers",
            ValidParentSet::CfgNode => "ControlFlowOp::CFG",
        }
        .into()
    }
}

impl OpType {
    /// Returns a set of flags describing the validity predicates for this operation.
    pub fn validity_flags(&self) -> OpValidityFlags {
        match self {
            OpType::Module(op) => op.validity_flags(),
            OpType::Function(op) => op.validity_flags(),
            OpType::BasicBlock(op) => op.validity_flags(),
        }
    }

    /// Validates the complete set of children
    pub fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = &'a OpType>,
    ) -> bool {
        let _ = children;
        true

        // TODO: Boundary children conditions

        // TODO: No repeated Input or output nodes
    }
}

/// Errors that can occur while checking the children of a node.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChildrenValidationError {}

impl ModuleOp {
    /// Returns the set of allowed parent operation types.
    pub fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_parents: match self {
                ModuleOp::Root => ValidParentSet::None,
                _ => ValidParentSet::ModuleRoot,
            },
            is_container: matches!(self, ModuleOp::Root | ModuleOp::Def { .. }),
            is_df_container: matches!(self, ModuleOp::Def { .. }),
            requires_children: matches!(self, ModuleOp::Def { .. }),
            require_dag: matches!(self, ModuleOp::Def { .. }),
        }
    }

    pub fn validate_first_child(&self, child: &OpType) -> bool {
        match self {
            ModuleOp::Root { .. } => matches!(child, OpType::Module(ModuleOp::Def { .. })),
            ModuleOp::Def { .. } => matches!(child, OpType::Function(DataflowOp::Input { .. })),
            _ => true,
        }
    }

    pub fn validate_last_child(&self, child: &OpType) -> bool {
        match self {
            ModuleOp::Def { .. } => matches!(child, OpType::Function(DataflowOp::Output { .. })),
            _ => true,
        }
    }
}

impl ControlFlowOp {
    // TODO: CFG nodes require checking the internal signature of pairs of
    // BasicBlocks connected by ControlFlow edges. This is not currently
    // implemented, and should probably go outside of the OpTypeValidator trait.

    /// Returns the set of allowed parent operation types.
    pub fn validity_flags(&self) -> OpValidityFlags {
        let is_df_container = matches!(
            self,
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. }
        );

        OpValidityFlags {
            allowed_parents: ValidParentSet::DataflowContainers,
            is_container: true,
            is_df_container,
            requires_children: true,
            require_dag: is_df_container,
        }
    }

    pub fn validate_first_child(&self, child: &OpType) -> bool {
        // TODO: check signatures
        match self {
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. } => {
                matches!(child, OpType::Function(DataflowOp::Input { .. }))
            }
            ControlFlowOp::CFG { .. } => matches!(child, OpType::BasicBlock(_)),
        }
    }

    pub fn validate_last_child(&self, child: &OpType) -> bool {
        // TODO: check signatures
        match self {
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. } => {
                matches!(child, OpType::Function(DataflowOp::Output { .. }))
            }
            ControlFlowOp::CFG { .. } => matches!(child, OpType::BasicBlock(_)),
        }
    }
}

impl BasicBlockOp {
    /// Returns the set of allowed parent operation types.
    pub fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_parents: ValidParentSet::CfgNode,
            is_container: true,
            is_df_container: true,
            requires_children: true,
            require_dag: true,
        }
    }
}

impl DataflowOp {
    /// Returns the set of allowed parent operation types.
    pub fn validity_flags(&self) -> OpValidityFlags {
        match self {
            DataflowOp::ControlFlow { op } => op.validity_flags(),
            DataflowOp::Nested { .. } => OpValidityFlags {
                allowed_parents: ValidParentSet::DataflowContainers,
                is_container: true,
                is_df_container: true,
                requires_children: true,
                require_dag: true,
            },
            _ => OpValidityFlags {
                allowed_parents: ValidParentSet::DataflowContainers,
                ..Default::default()
            },
        }
    }

    pub fn validate_first_child(&self, child: &OpType) -> bool {
        match self {
            DataflowOp::ControlFlow { op } => op.validate_first_child(child),
            DataflowOp::Nested { .. } => {
                matches!(child, OpType::Function(DataflowOp::Input { .. }))
            }
            _ => true,
        }
    }
}
