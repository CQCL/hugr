//! Definitions for validating hugr nodes according to their operation type.
//!
//! Adds a `validity_flags` method to [`OpType`] that returns a series of flags
//! used by the [`crate::hugr::validate`] module.
//!
//! It also defines a `validate_op_children` method for more complex tests that
//! require traversing the children.

use itertools::Itertools;
use portgraph::{NodeIndex, PortOffset};
use thiserror::Error;

use crate::types::TypeRow;

use super::dataflow::{DataflowOpTrait, DataflowParent};
use super::{impl_validate_op, BasicBlock, ExitBlock, OpTag, OpTrait, OpType, ValidateOp};

/// A set of property flags required for an operation.
#[non_exhaustive]
pub struct OpValidityFlags {
    /// The set of valid children operation types.
    pub allowed_children: OpTag,
    /// Additional restrictions on the first child operation.
    ///
    /// This is checked in addition to the child allowing the parent optype.
    pub allowed_first_child: OpTag,
    /// Additional restrictions on the second child operation
    ///
    /// This is checked in addition to the child allowing the parent optype.
    pub allowed_second_child: OpTag,
    /// Whether the operation must have children.
    pub requires_children: bool,
    /// Whether the children must form a DAG (no cycles).
    pub requires_dag: bool,
    /// A validation check for edges between children
    ///
    // Enclosed in an `Option` to avoid iterating over the edges if not needed.
    pub edge_check: Option<fn(ChildrenEdgeData) -> Result<(), EdgeValidationError>>,
}

impl Default for OpValidityFlags {
    fn default() -> Self {
        // Defaults to flags valid for non-container operations
        Self {
            allowed_children: OpTag::None,
            allowed_first_child: OpTag::Any,
            allowed_second_child: OpTag::Any,
            requires_children: false,
            requires_dag: false,
            edge_check: None,
        }
    }
}

impl ValidateOp for super::Module {
    fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_children: OpTag::ModuleOp,
            requires_children: false,
            ..Default::default()
        }
    }
}

impl ValidateOp for super::Conditional {
    fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_children: OpTag::Case,
            requires_children: true,
            requires_dag: false,
            ..Default::default()
        }
    }

    fn validate_op_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        let children = children.collect_vec();
        // The first input to the ɣ-node is a value of Sum type,
        // whose arity matches the number of children of the ɣ-node.
        if self.sum_rows.len() != children.len() {
            return Err(ChildrenValidationError::InvalidConditionalSum {
                child: children[0].0, // Pass an arbitrary child
                expected_count: children.len(),
                actual_sum_rows: self.sum_rows.clone(),
            });
        }

        // Each child must have its variant's row and the rest of `inputs` as input,
        // and matching output
        for (i, (child, optype)) in children.into_iter().enumerate() {
            let case_op = optype
                .as_case()
                .expect("Child check should have already checked valid ops.");
            let sig = &case_op.inner_signature();
            if sig.input != self.case_input_row(i).unwrap() || sig.output != self.outputs {
                return Err(ChildrenValidationError::ConditionalCaseSignature {
                    child,
                    optype: optype.clone(),
                });
            }
        }

        Ok(())
    }
}

impl ValidateOp for super::CFG {
    fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_children: OpTag::ControlFlowChild,
            allowed_first_child: OpTag::DataflowBlock,
            allowed_second_child: OpTag::BasicBlockExit,
            requires_children: true,
            requires_dag: false,
            edge_check: Some(validate_cfg_edge),
            ..Default::default()
        }
    }

    fn validate_op_children<'a>(
        &self,
        mut children: impl Iterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        let (entry, entry_op) = children.next().unwrap();
        let (exit, exit_op) = children.next().unwrap();
        let entry_op = entry_op
            .as_dataflow_block()
            .expect("Child check should have already checked valid ops.");
        let exit_op = exit_op
            .as_exit_block()
            .expect("Child check should have already checked valid ops.");

        let sig = self.signature();
        if entry_op.inner_signature().input() != sig.input() {
            return Err(ChildrenValidationError::IOSignatureMismatch {
                child: entry,
                actual: entry_op.inner_signature().input().clone(),
                expected: sig.input().clone(),
                node_desc: "BasicBlock Input",
                container_desc: "CFG",
            });
        }
        if &exit_op.cfg_outputs != sig.output() {
            return Err(ChildrenValidationError::IOSignatureMismatch {
                child: exit,
                actual: exit_op.cfg_outputs.clone(),
                expected: sig.output().clone(),
                node_desc: "BasicBlockExit Output",
                container_desc: "CFG",
            });
        }
        for (child, optype) in children {
            if optype.tag() == OpTag::BasicBlockExit {
                return Err(ChildrenValidationError::InternalExitChildren { child });
            }
        }
        Ok(())
    }
}
/// Errors that can occur while checking the children of a node.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum ChildrenValidationError {
    /// An CFG graph has an exit operation as a non-second child.
    #[error("Exit basic blocks are only allowed as the second child in a CFG graph")]
    InternalExitChildren { child: NodeIndex },
    /// An operation only allowed as the first/second child was found as an intermediate child.
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
    /// The signature of a child case in a conditional operation does not match the container's signature.
    #[error("A conditional case has optype {optype:?}, which differs from the signature of Conditional container")]
    ConditionalCaseSignature { child: NodeIndex, optype: OpType },
    /// The conditional container's branching value does not match the number of children.
    #[error("The conditional container's branch Sum input should be a sum with {expected_count} elements, but it had {} elements. Sum rows: {actual_sum_rows:?}",
        actual_sum_rows.len())]
    InvalidConditionalSum {
        child: NodeIndex,
        expected_count: usize,
        actual_sum_rows: Vec<TypeRow>,
    },
}

impl ChildrenValidationError {
    /// Returns the node index of the child that caused the error.
    pub fn child(&self) -> NodeIndex {
        match self {
            ChildrenValidationError::InternalIOChildren { child, .. } => *child,
            ChildrenValidationError::InternalExitChildren { child, .. } => *child,
            ChildrenValidationError::ConditionalCaseSignature { child, .. } => *child,
            ChildrenValidationError::IOSignatureMismatch { child, .. } => *child,
            ChildrenValidationError::InvalidConditionalSum { child, .. } => *child,
        }
    }
}

/// Errors that can occur while checking the edges between children of a node.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum EdgeValidationError {
    /// The dataflow signature of two connected basic blocks does not match.
    #[error("The dataflow signature of two connected basic blocks does not match. Output signature: {source_op:?}, input signature: {target_op:?}",
        source_op = edge.source_op,
        target_op = edge.target_op
    )]
    CFGEdgeSignatureMismatch { edge: ChildrenEdgeData },
}

impl EdgeValidationError {
    /// Returns information on the edge that caused the error.
    pub fn edge(&self) -> &ChildrenEdgeData {
        match self {
            EdgeValidationError::CFGEdgeSignatureMismatch { edge } => edge,
        }
    }
}

/// Auxiliary structure passed as data to [`OpValidityFlags::edge_check`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChildrenEdgeData {
    /// Source child.
    pub source: NodeIndex,
    /// Target child.
    pub target: NodeIndex,
    /// Operation type of the source child.
    pub source_op: OpType,
    /// Operation type of the target child.
    pub target_op: OpType,
    /// Source port.
    pub source_port: PortOffset,
    /// Target port.
    pub target_port: PortOffset,
}

impl<T: DataflowParent> ValidateOp for T {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_children: OpTag::DataflowChild,
            allowed_first_child: OpTag::Input,
            allowed_second_child: OpTag::Output,
            requires_children: true,
            requires_dag: true,
            ..Default::default()
        }
    }

    /// Validate the ordered list of children.
    fn validate_op_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        let sig = self.inner_signature();
        validate_io_nodes(&sig.input, &sig.output, "DataflowParent", children)
    }
}

/// Checks a that the list of children nodes does not contain Input and Output
/// nodes outside of the first and second elements respectively, and that those
/// have the correct signature.
fn validate_io_nodes<'a>(
    expected_input: &TypeRow,
    expected_output: &TypeRow,
    container_desc: &'static str,
    mut children: impl Iterator<Item = (NodeIndex, &'a OpType)>,
) -> Result<(), ChildrenValidationError> {
    // Check that the signature matches with the Input and Output rows.
    let (first, first_optype) = children.next().unwrap();
    let (second, second_optype) = children.next().unwrap();

    let first_sig = first_optype.dataflow_signature().unwrap_or_default();
    if &first_sig.output != expected_input {
        return Err(ChildrenValidationError::IOSignatureMismatch {
            child: first,
            actual: first_sig.output,
            expected: expected_input.clone(),
            node_desc: "Input",
            container_desc,
        });
    }
    let second_sig = second_optype.dataflow_signature().unwrap_or_default();

    if &second_sig.input != expected_output {
        return Err(ChildrenValidationError::IOSignatureMismatch {
            child: second,
            actual: second_sig.input,
            expected: expected_output.clone(),
            node_desc: "Output",
            container_desc,
        });
    }

    // The first and second children have already been popped from the iterator.
    for (child, optype) in children {
        match optype.tag() {
            OpTag::Input => {
                return Err(ChildrenValidationError::InternalIOChildren {
                    child,
                    optype: optype.clone(),
                    expected_position: "first",
                })
            }
            OpTag::Output => {
                return Err(ChildrenValidationError::InternalIOChildren {
                    child,
                    optype: optype.clone(),
                    expected_position: "second",
                })
            }
            _ => {}
        }
    }
    Ok(())
}

/// Validate an edge between two basic blocks in a CFG sibling graph.
fn validate_cfg_edge(edge: ChildrenEdgeData) -> Result<(), EdgeValidationError> {
    let source = &edge
        .source_op
        .as_dataflow_block()
        .expect("CFG sibling graphs can only contain basic block operations.");

    let target_input = match &edge.target_op {
        OpType::DataflowBlock(dfb) => dfb.dataflow_input(),
        OpType::ExitBlock(exit) => exit.dataflow_input(),
        _ => panic!("CFG sibling graphs can only contain basic block operations."),
    };

    if source.successor_input(edge.source_port.index()).as_ref() != Some(target_input) {
        return Err(EdgeValidationError::CFGEdgeSignatureMismatch { edge });
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::extension::prelude::USIZE_T;
    use crate::ops::dataflow::IOTrait;
    use crate::{ops, type_row};
    use cool_asserts::assert_matches;

    use super::*;

    #[test]
    fn test_validate_io_nodes() {
        let in_types: TypeRow = type_row![USIZE_T];
        let out_types: TypeRow = type_row![USIZE_T, USIZE_T];

        let input_node: OpType = ops::Input::new(in_types.clone()).into();
        let output_node = ops::Output::new(out_types.clone()).into();
        let leaf_node = ops::Noop { ty: USIZE_T }.into();

        // Well-formed dataflow sibling nodes. Check the input and output node signatures.
        let children = vec![
            (0, &input_node),
            (1, &output_node),
            (2, &leaf_node),
            (3, &leaf_node),
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
            Err(ChildrenValidationError::IOSignatureMismatch { child, .. }) if child.index() == 1
        );

        // Internal I/O nodes
        let children = vec![
            (0, &input_node),
            (1, &output_node),
            (42, &leaf_node),
            (2, &leaf_node),
            (3, &output_node),
        ];
        assert_matches!(
            validate_io_nodes(&in_types, &out_types, "test", make_iter(&children)),
            Err(ChildrenValidationError::InternalIOChildren { child, .. }) if child.index() == 3
        );
    }

    fn make_iter<'a>(
        children: &'a [(usize, &OpType)],
    ) -> impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)> {
        children.iter().map(|(n, op)| (NodeIndex::new(*n), *op))
    }
}

use super::{
    AliasDecl, AliasDefn, Call, CallIndirect, Const, ExtensionOp, FuncDecl, Input, Lift,
    LoadConstant, LoadFunction, Noop, OpaqueOp, Output, Tag, UnpackTuple,
};
impl_validate_op!(FuncDecl);
impl_validate_op!(AliasDecl);
impl_validate_op!(AliasDefn);
impl_validate_op!(Input);
impl_validate_op!(Output);
impl_validate_op!(Const);
impl_validate_op!(Call);
impl_validate_op!(LoadConstant);
impl_validate_op!(LoadFunction);
impl_validate_op!(CallIndirect);
impl_validate_op!(ExtensionOp);
impl_validate_op!(OpaqueOp);
impl_validate_op!(Noop);
// impl_validate_op!(MakeTuple);
impl_validate_op!(UnpackTuple);
impl_validate_op!(Tag);
impl_validate_op!(Lift);
impl_validate_op!(ExitBlock);
