//! Definitions for validating hugr nodes according to their operation type.
//!
//! Adds a `validity_flags` method to [`OpType`] that returns a series of flags
//! used by the [`crate::hugr::validate`] module.
//!
//! It also defines a `validate_children` method for more complex tests that
//! require traversing the children.

use std::fmt::Display;

use itertools::Itertools;
use portgraph::{NodeIndex, PortOffset};
use thiserror::Error;

use crate::types::{SimpleType, TypeRow};

use super::{BasicBlockOp, ControlFlowOp, DataflowOp, ModuleOp, OpType};

/// A set of property flags required for an operation
#[non_exhaustive]
pub struct OpValidityFlags {
    /// The set of valid children operation types
    pub allowed_children: ValidOpSet,
    /// Additional restrictions on the first child operation.
    ///
    /// This is checked in addition to the child allowing the parent optype.
    pub allowed_first_child: ValidOpSet,
    /// Additional restrictions on the last child operation
    ///
    /// This is checked in addition to the child allowing the parent optype.
    pub allowed_last_child: ValidOpSet,
    /// Whether the operation must have children
    pub requires_children: bool,
    /// Whether the children must form a DAG (no cycles)
    pub requires_dag: bool,
    /// A strict requirement on the number of non-dataflow input and output wires
    pub non_df_ports: (Option<usize>, Option<usize>),
    /// A validation check for edges between children
    pub edge_check: Option<fn(ChildrenEdgeData) -> Result<(), EdgeValidationError>>,
}

impl Default for OpValidityFlags {
    fn default() -> Self {
        // Defaults to flags valid for non-container operations
        Self {
            allowed_children: ValidOpSet::None,
            allowed_first_child: ValidOpSet::Any,
            allowed_last_child: ValidOpSet::Any,
            requires_children: false,
            requires_dag: false,
            non_df_ports: (None, None),
            edge_check: None,
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
    /// Non-root module operations
    ModuleOps,
    /// Any dataflow operation.
    DataflowOps,
    /// A dataflow input
    Input,
    /// A dataflow output
    Output,
    /// A function definition
    Def,
    /// A control flow basic block
    BasicBlock,
    /// A control flow exit node
    BasicBlockExit,
}

impl ValidOpSet {
    /// Returns true if the set contains the given operation type.
    pub fn contains(&self, optype: &OpType) -> bool {
        match self {
            ValidOpSet::Any => true,
            ValidOpSet::None => false,
            ValidOpSet::ModuleOps => {
                if let OpType::Module(op) = optype {
                    op != &ModuleOp::Root
                } else {
                    false
                }
            }
            ValidOpSet::DataflowOps => matches!(optype, OpType::Function(_)),
            ValidOpSet::Input => matches!(optype, OpType::Function(DataflowOp::Input { .. })),
            ValidOpSet::Output => matches!(optype, OpType::Function(DataflowOp::Output { .. })),
            ValidOpSet::Def => matches!(optype, OpType::Module(ModuleOp::Def { .. })),
            ValidOpSet::BasicBlock => matches!(optype, OpType::BasicBlock(_)),
            ValidOpSet::BasicBlockExit => {
                matches!(optype, OpType::BasicBlock(BasicBlockOp::Exit { .. }))
            }
        }
    }

    /// Returns a user-friendly description of the set.
    pub fn set_description(&self) -> &str {
        match self {
            ValidOpSet::Any => "Any",
            ValidOpSet::None => "None",
            ValidOpSet::ModuleOps => "Module operations",
            ValidOpSet::DataflowOps => "Dataflow operations",
            ValidOpSet::Input => "Input node",
            ValidOpSet::Output => "Output node",
            ValidOpSet::Def => "Function definition",
            ValidOpSet::BasicBlock => "Basic block",
            ValidOpSet::BasicBlockExit => "Exit basic block node",
        }
    }

    /// Returns whether the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, ValidOpSet::None)
    }
}

impl Display for ValidOpSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.set_description())
    }
}

impl OpType {
    /// Returns a set of flags describing the validity predicates for this operation.
    #[inline]
    pub fn validity_flags(&self) -> OpValidityFlags {
        match self {
            OpType::Module(op) => op.validity_flags(),
            OpType::Function(op) => op.validity_flags(),
            OpType::BasicBlock(op) => op.validity_flags(),
        }
    }

    /// Validate the ordered list of children
    #[inline]
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
    /// An CFG graph has an exit operation as a non-last child.
    #[error("Exit basic blocks are only allowed as the last child in a CFG graph")]
    InternalExitChildren { child: NodeIndex },
    /// An operation only allowed as the first/last child was found as an intermediate child.
    #[error("A {optype:?} operation is only allowed as a {expected_position} child")]
    InternalIOChildren {
        child: NodeIndex,
        optype: OpType,
        expected_position: &'static str,
    },
    /// The signature of the contained dataflow graph does not match the one of the container.
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
    /// The conditional container's branch predicate does not match the number of children
    #[error("The conditional container's branch predicate input should be a sum with {expected_count} elements, but it had {actual_count} elements. Predicate type: {actual_predicate:?} ")]
    InvalidConditionalPredicate {
        child: NodeIndex,
        expected_count: usize,
        actual_count: usize,
        actual_predicate: TypeRow,
    },
}

impl ChildrenValidationError {
    /// Returns the node index of the child that caused the error.
    pub fn child(&self) -> NodeIndex {
        match self {
            ChildrenValidationError::InternalIOChildren { child, .. } => *child,
            ChildrenValidationError::InternalExitChildren { child, .. } => *child,
            ChildrenValidationError::ConditionalBranchSignature { child, .. } => *child,
            ChildrenValidationError::IOSignatureMismatch { child, .. } => *child,
            ChildrenValidationError::InvalidConditionalPredicate { child, .. } => *child,
        }
    }
}

/// Errors that can occur while checking the edges between children of a node.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EdgeValidationError {
    /// The dataflow signature of two connected basic blocks does not match.
    #[error("The dataflow signature of two connected basic blocks does not match. Output signature: {source_op:?}, input signature: {target_op:?}",
        source_op = edge.source_op,
        target_op = edge.target_op
    )]
    CFGEdgeSignatureMismatch { edge: ChildrenEdgeData },
}

impl EdgeValidationError {
    /// Returns the node index of the child that caused the error.
    pub fn edge(&self) -> &ChildrenEdgeData {
        match self {
            EdgeValidationError::CFGEdgeSignatureMismatch { edge } => edge,
        }
    }
}

/// Auxiliary structure passed as data in the [`validate_children_edges`] method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildrenEdgeData {
    /// Source child
    pub source: NodeIndex,
    /// Target child
    pub target: NodeIndex,
    /// Operation type of the source child
    pub source_op: OpType,
    /// Operation type of the target child
    pub target_op: OpType,
    /// Source port
    pub source_port: PortOffset,
    /// Target port
    pub target_port: PortOffset,
}

impl ModuleOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        match self {
            ModuleOp::Root { .. } => OpValidityFlags {
                allowed_children: ValidOpSet::ModuleOps,
                requires_children: false,
                ..Default::default()
            },
            ModuleOp::Def { .. } => OpValidityFlags {
                allowed_children: ValidOpSet::DataflowOps,
                allowed_first_child: ValidOpSet::Input,
                allowed_last_child: ValidOpSet::Output,
                requires_children: true,
                requires_dag: true,
                ..Default::default()
            },
            // Default flags are valid for non-container operations
            _ => Default::default(),
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
                &signature.output,
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
        match self {
            BasicBlockOp::Beta { n_branches, .. } => OpValidityFlags {
                allowed_children: ValidOpSet::DataflowOps,
                allowed_first_child: ValidOpSet::Input,
                allowed_last_child: ValidOpSet::Output,
                requires_children: true,
                requires_dag: true,
                non_df_ports: (None, Some(*n_branches)),
                ..Default::default()
            },
            // Default flags are valid for non-container operations
            BasicBlockOp::Exit { .. } => Default::default(),
        }
    }

    /// Validate the ordered list of children
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            BasicBlockOp::Beta {
                inputs,
                outputs,
                n_branches,
            } => {
                let predicate_type = SimpleType::new_predicate(*n_branches);
                let node_outputs: TypeRow = [&[predicate_type], outputs.as_ref()].concat().into();
                validate_io_nodes(inputs, &node_outputs, "basic block graph", children)
            }
            // Exit nodes do not have children
            BasicBlockOp::Exit { .. } => Ok(()),
        }
    }
}

impl DataflowOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        match self {
            DataflowOp::ControlFlow { op } => op.validity_flags(),
            DataflowOp::Nested { .. } => OpValidityFlags {
                allowed_children: ValidOpSet::DataflowOps,
                allowed_first_child: ValidOpSet::Input,
                allowed_last_child: ValidOpSet::Output,
                requires_children: true,
                requires_dag: true,
                ..Default::default()
            },
            // Default flags are valid for non-container operations
            _ => Default::default(),
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
                &signature.output,
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
        match self {
            ControlFlowOp::Conditional { .. } => OpValidityFlags {
                allowed_children: ValidOpSet::DataflowOps,
                requires_children: true,
                requires_dag: false,
                ..Default::default()
            },
            ControlFlowOp::Loop { .. } => OpValidityFlags {
                allowed_children: ValidOpSet::DataflowOps,
                allowed_first_child: ValidOpSet::Input,
                allowed_last_child: ValidOpSet::Output,
                requires_children: true,
                requires_dag: true,
                ..Default::default()
            },
            ControlFlowOp::CFG { .. } => OpValidityFlags {
                allowed_children: ValidOpSet::BasicBlock,
                allowed_last_child: ValidOpSet::BasicBlockExit,
                requires_children: true,
                requires_dag: false,
                edge_check: Some(validate_cfg_edge),
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
            ControlFlowOp::Conditional {
                predicate_inputs,
                inputs,
                outputs,
            } => {
                let children = children.collect_vec();
                // The first input to the ɣ-node is a predicate of Sum type,
                // whose arity matches the number of children of the ɣ-node.
                if predicate_inputs.len() != children.len() {
                    return Err(ChildrenValidationError::InvalidConditionalPredicate {
                        child: children[0].0, // Pass an arbitrary child
                        expected_count: children.len(),
                        actual_count: predicate_inputs.len(),
                        actual_predicate: predicate_inputs.clone(),
                    });
                }

                // Each child must have it's predicate variant and the rest of `inputs` as input,
                // and matching output
                for (i, (child, optype)) in children.into_iter().enumerate() {
                    let sig = optype.signature();
                    let predicate_value = &predicate_inputs[i];
                    if sig.input[0] != *predicate_value
                        || sig.input[1..] != inputs[..]
                        || sig.output != *outputs
                    {
                        return Err(ChildrenValidationError::ConditionalBranchSignature {
                            child,
                            optype: optype.clone(),
                        });
                    }
                }
            }
            ControlFlowOp::Loop { inputs, outputs } => {
                let expected_output = SimpleType::new_sum(vec![
                    SimpleType::new_tuple(inputs.clone()),
                    SimpleType::new_tuple(outputs.clone()),
                ]);
                let expected_output: TypeRow = vec![expected_output].into();
                validate_io_nodes(
                    inputs,
                    &expected_output,
                    "tail-controlled loop graph",
                    children,
                )?;
            }
            ControlFlowOp::CFG { .. } => {
                // TODO: CFG nodes require checking the internal signature of pairs of
                // BasicBlocks connected by ControlFlow edges. This should probably go
                // outside of the OpTypeValidator trait, as we don't have access to the
                // graph edges from here.

                // Only the last child can be an exit node
                for (child, optype) in children.dropping_back(1) {
                    if matches!(optype, OpType::BasicBlock(BasicBlockOp::Exit { .. })) {
                        return Err(ChildrenValidationError::InternalExitChildren { child });
                    }
                }
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
    expected_output: &TypeRow,
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
    if &last_optype.signature().input != expected_output {
        return Err(ChildrenValidationError::IOSignatureMismatch {
            child: last,
            actual: last_optype.signature().input,
            expected: expected_output.clone(),
            node_desc: "Output",
            container_desc,
        });
    }

    // The first and last children have already been popped from the iterator.
    for (child, optype) in children {
        match optype {
            OpType::Function(DataflowOp::Input { .. }) => {
                return Err(ChildrenValidationError::InternalIOChildren {
                    child,
                    optype: optype.clone(),
                    expected_position: "first",
                })
            }
            OpType::Function(DataflowOp::Output { .. }) => {
                return Err(ChildrenValidationError::InternalIOChildren {
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

/// Validate the ordered list of children
fn validate_cfg_edge(_edges: ChildrenEdgeData) -> Result<(), EdgeValidationError> {
    // Basic blocks connected by control flow wires must have matching
    // input/output types.

    // TODO: Matching number of connections

    Ok(())
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{
        ops::LeafOp,
        type_row,
        types::{ClassicType, SimpleType},
    };

    use super::*;

    #[test]
    fn test_validate_io_nodes() {
        const B: SimpleType = SimpleType::Classic(ClassicType::bit());

        let in_types = type_row![B];
        let out_types = type_row![B, B];

        let input_node = OpType::Function(DataflowOp::Input {
            types: in_types.clone(),
        });
        let output_node = OpType::Function(DataflowOp::Output {
            types: out_types.clone(),
        });
        let leaf_node = OpType::Function(DataflowOp::Leaf {
            op: LeafOp::Copy {
                n_copies: 2,
                typ: ClassicType::bit(),
            },
        });

        // Well-formed dataflow sibling nodes. Check the input and output node signatures.
        let children = vec![
            (0, &input_node),
            (1, &leaf_node),
            (2, &leaf_node),
            (3, &output_node),
        ];
        assert_eq!(
            validate_io_nodes(&in_types, &out_types, "test", make_iter(&children)),
            Ok(())
        );
        assert_matches!(
            validate_io_nodes(&out_types, &out_types, "test", make_iter(&children)),
            Err(ChildrenValidationError::IOSignatureMismatch { child, .. }) if child.index() == 0
        );
        assert_matches!(
            validate_io_nodes(&in_types, &in_types, "test", make_iter(&children)),
            Err(ChildrenValidationError::IOSignatureMismatch { child, .. }) if child.index() == 3
        );

        // Internal I/O nodes
        let children = vec![
            (0, &input_node),
            (1, &leaf_node),
            (42, &output_node),
            (2, &leaf_node),
            (3, &output_node),
        ];
        assert_matches!(
            validate_io_nodes(&in_types, &out_types, "test", make_iter(&children)),
            Err(ChildrenValidationError::InternalIOChildren { child, .. }) if child.index() == 42
        );
    }

    fn make_iter<'a>(
        children: &'a [(usize, &OpType)],
    ) -> impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)> {
        children.iter().map(|(n, op)| (NodeIndex::new(*n), *op))
    }
}
