//! Definitions for validating hugr nodes according to their operation type.
//!
//! Adds a `validity_flags` method to [`OpType`] that returns a series of flags
//! used by the [`crate::hugr::validate`] module.
//!
//! It also defines a `validate_children` method for more complex tests that
//! require traversing the children.

use std::fmt::Display;

use thiserror::Error;

use super::{BasicBlockOp, ControlFlowOp, DataflowOp, ModuleOp, OpType};

/// A set of property flags required for an operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpValidityFlags {
    /// The set of valid parent operation types
    pub allowed_parents: ValidOpSet,
    /// Additional restrictions on the first child operation.
    ///
    /// This is checked in addition to the child allowing the parent optype.
    pub allowed_first_child: ValidOpSet,
    /// Additional restrictions on the last child operation
    ///
    /// This is checked in addition to the child allowing the parent optype.
    pub allowed_last_child: ValidOpSet,
    /// Whether the operation can have children
    pub is_container: bool,
    /// Whether the operation contains dataflow children
    pub is_df_container: bool,
    /// Whether the operation must have children
    pub requires_children: bool,
    /// Whether the children must form a DAG (no cycles)
    pub requires_dag: bool,
}

impl Default for OpValidityFlags {
    fn default() -> Self {
        Self {
            allowed_parents: ValidOpSet::Any,
            allowed_first_child: ValidOpSet::Any,
            allowed_last_child: ValidOpSet::Any,
            is_container: false,
            is_df_container: false,
            requires_children: false,
            requires_dag: false,
        }
    }
}

/// Sets of operation kinds.
///
/// Used to validate the allowed parent and children operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ValidOpSet {
    /// All operations allowed
    #[default]
    Any,
    /// No valid operation types.
    None,
    /// Only the module root operation.
    ModuleRoot,
    /// Any dataflow container operation.
    DataflowContainers,
    /// A control flow container operation.
    CfgNode,
    /// A dataflow input
    Input,
    /// A dataflow output
    Output,
    /// A function definition
    Def,
    /// A control flow basic block
    BasicBlock,
}

impl ValidOpSet {
    /// Returns true if the set contains the given operation type.
    pub fn contains(&self, optype: &OpType) -> bool {
        match self {
            ValidOpSet::Any => true,
            ValidOpSet::None => false,
            ValidOpSet::ModuleRoot => matches!(optype, OpType::Module(ModuleOp::Root)),
            ValidOpSet::DataflowContainers => optype.validity_flags().is_df_container,
            ValidOpSet::CfgNode => matches!(
                optype,
                OpType::Function(DataflowOp::ControlFlow {
                    op: ControlFlowOp::CFG { .. }
                })
            ),
            ValidOpSet::Input => matches!(optype, OpType::Function(DataflowOp::Input { .. })),
            ValidOpSet::Output => matches!(optype, OpType::Function(DataflowOp::Output { .. })),
            ValidOpSet::Def => matches!(optype, OpType::Module(ModuleOp::Def { .. })),
            ValidOpSet::BasicBlock => matches!(optype, OpType::BasicBlock(_)),
        }
    }

    /// Returns a user-friendly description of the set.
    pub fn set_description(&self) -> &str {
        match self {
            ValidOpSet::Any => "Any",
            ValidOpSet::None => "None",
            ValidOpSet::ModuleRoot => "Module roots",
            ValidOpSet::DataflowContainers => "Dataflow containers",
            ValidOpSet::CfgNode => "ControlFlowOp::CFG containers",
            ValidOpSet::Input => "Input node",
            ValidOpSet::Output => "Output node",
            ValidOpSet::Def => "Function definition",
            ValidOpSet::BasicBlock => "Basic block",
        }
    }
}

impl Display for ValidOpSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.set_description())
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

    /// Validate the ordered list of children
    pub fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = &'a OpType>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            OpType::Module(_) => Ok(()),
            OpType::Function(op) => op.validate_children(children),
            OpType::BasicBlock(_) => Ok(()),
        }
    }
}

/// Errors that can occur while checking the children of a node.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChildrenValidationError {}

impl ModuleOp {
    /// Returns the set of allowed parent operation types.
    pub fn validity_flags(&self) -> OpValidityFlags {
        let flags = OpValidityFlags {
            allowed_parents: ValidOpSet::ModuleRoot,
            is_container: matches!(self, ModuleOp::Root | ModuleOp::Def { .. }),
            is_df_container: matches!(self, ModuleOp::Def { .. }),
            requires_children: matches!(self, ModuleOp::Def { .. }),
            requires_dag: matches!(self, ModuleOp::Def { .. }),
            ..Default::default()
        };
        match self {
            ModuleOp::Root { .. } => OpValidityFlags {
                allowed_parents: ValidOpSet::None,
                allowed_first_child: ValidOpSet::Def,
                ..flags
            },
            ModuleOp::Def { .. } => OpValidityFlags {
                allowed_first_child: ValidOpSet::Input,
                allowed_last_child: ValidOpSet::Output,
                ..flags
            },
            _ => flags,
        }
    }
}

impl ControlFlowOp {
    /// Returns the set of allowed parent operation types.
    pub fn validity_flags(&self) -> OpValidityFlags {
        let is_df_container = matches!(
            self,
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. }
        );

        let flags = OpValidityFlags {
            allowed_parents: ValidOpSet::DataflowContainers,
            is_container: true,
            requires_children: true,
            requires_dag: is_df_container,
            ..Default::default()
        };
        match self {
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. } => OpValidityFlags {
                is_df_container: true,
                ..flags
            },
            ControlFlowOp::CFG { .. } => OpValidityFlags {
                allowed_first_child: ValidOpSet::BasicBlock,
                allowed_last_child: ValidOpSet::BasicBlock,
                is_df_container: false,
                ..flags
            },
        }
    }

    /// Validate the ordered list of children
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = &'a OpType>,
    ) -> Result<(), ChildrenValidationError> {
        let _ = children;
        Ok(())

        // TODO: For Conditional and loop, all blocks must be `DataFlowOp::Nested`, with matching signatures.

        // TODO: CFG nodes require checking the internal signature of pairs of
        // BasicBlocks connected by ControlFlow edges. This should probably go
        // outside of the OpTypeValidator trait, as we don't have access to the
        // graph edges from here.
    }
}

impl BasicBlockOp {
    /// Returns the set of allowed parent operation types.
    pub fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_parents: ValidOpSet::CfgNode,
            allowed_first_child: ValidOpSet::Any,
            allowed_last_child: ValidOpSet::Any,
            is_container: true,
            is_df_container: true,
            requires_children: true,
            requires_dag: true,
        }
    }
}

impl DataflowOp {
    /// Returns the set of allowed parent operation types.
    pub fn validity_flags(&self) -> OpValidityFlags {
        match self {
            DataflowOp::ControlFlow { op } => op.validity_flags(),
            DataflowOp::Nested { .. } => OpValidityFlags {
                allowed_parents: ValidOpSet::DataflowContainers,
                allowed_first_child: ValidOpSet::Input,
                allowed_last_child: ValidOpSet::Output,
                is_container: true,
                is_df_container: true,
                requires_children: true,
                requires_dag: true,
            },
            _ => OpValidityFlags {
                allowed_parents: ValidOpSet::DataflowContainers,
                ..Default::default()
            },
        }
    }

    /// Validate the ordered list of children
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = &'a OpType>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            DataflowOp::ControlFlow { op } => op.validate_children(children),
            DataflowOp::Nested { .. } => {
                // TODO: Don't allow non-edge Input or Output nodes
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
