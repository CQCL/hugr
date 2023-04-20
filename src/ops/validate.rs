//! Definitions for validating hugr nodes according to their operation type.
//!
//! Adds a `validity_flags` method to [`OpType`] that returns a series of flags
//! used by the [`crate::hugr::validate`] module.
//!
//! It also defines a `validate_children` method for more complex tests that
//! require traversing the children.

use std::fmt::Display;

use portgraph::NodeIndex;
use thiserror::Error;

use crate::types::TypeRow;

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
            ValidOpSet::CfgNode => "CFG containers",
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
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            OpType::Module(op) => op.validate_children(children),
            OpType::Function(op) => op.validate_children(children),
            OpType::BasicBlock(op) => op.validate_children(children),
        }
    }
}

/// Errors that can occur while checking the children of a node.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChildrenValidationError {
    /// An operation only allowed as the first/last child was found as an intermediate child.
    #[error("A {optype:?} operation is only allowed as a {expected_position} child")]
    NonEdgeChildren {
        child: NodeIndex,
        optype: OpType,
        expected_position: &'static str,
    },
    /// The signature of the tail-controlled loop graph does not match the one of the container.
    #[error("The {node_desc} node of a {container_desc} has a signature of {actual:?}, which differs from the expected type row {expected:?}")]
    IOSignatureMismatch {
        child: NodeIndex,
        actual: TypeRow,
        expected: TypeRow,
        node_desc: &'static str,
        container_desc: &'static str,
    },
    /// The signature of a children branch in a conditional operation does not match the container's signature.
    #[error("A conditional branch has optype {optype:?}, which differs from the signature of Conditional container")]
    ConditionalBranchSignature { child: NodeIndex, optype: OpType },
}

impl ChildrenValidationError {
    /// Returns the node index of the child that caused the error.
    pub fn child(&self) -> NodeIndex {
        match self {
            ChildrenValidationError::NonEdgeChildren { child, .. } => *child,
            ChildrenValidationError::ConditionalBranchSignature { child, .. } => *child,
            ChildrenValidationError::IOSignatureMismatch { child, .. } => *child,
        }
    }
}

impl ModuleOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        let flags = OpValidityFlags {
            allowed_parents: ValidOpSet::ModuleRoot,
            ..Default::default()
        };
        match self {
            ModuleOp::Root { .. } => OpValidityFlags {
                allowed_parents: ValidOpSet::None,
                allowed_first_child: ValidOpSet::Def,
                is_container: true,
                requires_children: false,
                ..flags
            },
            ModuleOp::Def { .. } => OpValidityFlags {
                allowed_first_child: ValidOpSet::Input,
                allowed_last_child: ValidOpSet::Output,
                is_container: true,
                is_df_container: true,
                requires_children: true,
                requires_dag: true,
                ..flags
            },
            _ => flags,
        }
    }

    /// Validate the ordered list of children
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            ModuleOp::Def { signature } => validate_io_nodes(
                &signature.input,
                Some(&signature.output),
                "function definition",
                children,
            ),
            _ => Ok(()),
        }
    }
}

impl BasicBlockOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
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

    /// Validate the ordered list of children
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        // TODO: The output signature of a basic block should be a sum of the different possible outputs.
        // This is not yet implemented in the type system.
        validate_io_nodes(&self.inputs, None, "basic block graph", children)
    }
}

impl DataflowOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
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
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            DataflowOp::ControlFlow { op } => op.validate_children(children),
            DataflowOp::Nested { signature } => validate_io_nodes(
                &signature.input,
                Some(&signature.output),
                "nested graph",
                children,
            ),
            _ => Ok(()),
        }
    }
}

impl ControlFlowOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        let flags = OpValidityFlags {
            allowed_parents: ValidOpSet::DataflowContainers,
            is_container: true,
            requires_children: true,
            ..Default::default()
        };
        match self {
            ControlFlowOp::Conditional { .. } => OpValidityFlags {
                is_df_container: true,
                requires_dag: false,
                ..flags
            },
            ControlFlowOp::Loop { .. } => OpValidityFlags {
                allowed_first_child: ValidOpSet::Input,
                allowed_last_child: ValidOpSet::Output,
                is_df_container: true,
                requires_dag: true,
                ..flags
            },
            ControlFlowOp::CFG { .. } => OpValidityFlags {
                is_df_container: false,
                requires_dag: false,
                ..flags
            },
        }
    }

    /// Validate the ordered list of children
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            ControlFlowOp::Conditional {
                predicate,
                inputs,
                outputs,
            } => {
                // TODO: "The first input to the ɣ-node is a predicate of Sum type, whose arity matches the number of children of the ɣ-node."
                let _ = predicate;

                // Every children must have the specified signature.
                for (child, optype) in children {
                    let sig = optype.signature();
                    if sig.input != *inputs || sig.output != *outputs {
                        return Err(ChildrenValidationError::ConditionalBranchSignature {
                            child,
                            optype: optype.clone(),
                        });
                    }
                }
            }
            ControlFlowOp::Loop { vars } => {
                // TODO: Check the output node signature. "the DDG within the
                // θ-node computes a value of 2-ary Sum type; the first variant
                // means to repeat the loop with those values “fed” in at at the
                // top; the second variant means to exit the loop with those
                // values."
                validate_io_nodes(vars, None, "tail-controlled loop graph", children)?;
            }
            ControlFlowOp::CFG { .. } => {
                // TODO: CFG nodes require checking the internal signature of pairs of
                // BasicBlocks connected by ControlFlow edges. This should probably go
                // outside of the OpTypeValidator trait, as we don't have access to the
                // graph edges from here.
            }
        }
        Ok(())
    }
}

/// Checks a that the list of children nodes does not contain Input and Output
/// nodes outside of the first and last elements respectively, and that those
/// have the correct signature.
fn validate_io_nodes<'a>(
    expected_input: &TypeRow,
    expected_output: Option<&TypeRow>, // TODO: This should be non-optional, but we allow it for not-yet-implemented checks.
    container_desc: &'static str,
    mut children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
) -> Result<(), ChildrenValidationError> {
    // Check that the signature matches with the Input and Output rows.
    let (first, first_optype) = children.next().unwrap();
    let (last, last_optype) = children.next_back().unwrap();

    if &first_optype.signature().output != expected_input {
        return Err(ChildrenValidationError::IOSignatureMismatch {
            child: first,
            actual: first_optype.signature().output,
            expected: expected_input.clone(),
            node_desc: "Input",
            container_desc,
        });
    }
    if let Some(expected_output) = expected_output {
        if &last_optype.signature().input != expected_output {
            return Err(ChildrenValidationError::IOSignatureMismatch {
                child: last,
                actual: last_optype.signature().input,
                expected: expected_output.clone(),
                node_desc: "Output",
                container_desc,
            });
        }
    }

    // The first and last children have already been popped from the iterator.
    for (child, optype) in children {
        match optype {
            OpType::Function(DataflowOp::Input { .. }) => {
                return Err(ChildrenValidationError::NonEdgeChildren {
                    child,
                    optype: optype.clone(),
                    expected_position: "first",
                })
            }
            OpType::Function(DataflowOp::Output { .. }) => {
                return Err(ChildrenValidationError::NonEdgeChildren {
                    child,
                    optype: optype.clone(),
                    expected_position: "last",
                })
            }
            _ => {}
        }
    }
    Ok(())
}
