//! Definitions for validating hugr nodes according to their operation type.
//!
//! Adds a `validity_flags` method to [`OpType`] that returns a series of flags
//! used by the [`crate::hugr::validate`] module.
//!
//! It also defines a `validate_children` method for more complex tests that
//! require traversing the children.

use itertools::Itertools;
use portgraph::{NodeIndex, PortOffset};
use thiserror::Error;

use crate::types::{ClassicType, SimpleType, TypeRow};
use crate::Direction;

use super::{impl_validate_op, tag::OpTag, BasicBlock, OpTrait, OpType, ValidateOp};

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
    /// A strict requirement on the number of non-dataflow multiports.
    ///
    /// If not specified, the operation must have exactly one non-dataflow port
    /// if the operation type has other_edges, or zero otherwise.
    pub non_df_ports: (Option<usize>, Option<usize>),
    /// A validation check for edges between children
    ///
    // Enclosed in an `Option` to avoid iterating over the edges if not needed.
    pub edge_check: Option<fn(ChildrenEdgeData) -> Result<(), EdgeValidationError>>,
}

impl OpValidityFlags {
    /// Get the number of non-dataflow multiports.
    ///
    /// If None, the operation must have exactly one non-dataflow port
    /// if the operation type has other_edges, or zero otherwise.
    pub fn non_df_port_count(&self, dir: Direction) -> Option<usize> {
        match dir {
            Direction::Incoming => self.non_df_ports.0,
            Direction::Outgoing => self.non_df_ports.1,
        }
    }
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
            non_df_ports: (None, None),
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

impl ValidateOp for super::FuncDefn {
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

    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        validate_io_nodes(
            &self.signature.input,
            &self.signature.output,
            "function definition",
            children,
        )
    }
}

impl ValidateOp for super::DFG {
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

    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        validate_io_nodes(
            &self.signature.input,
            &self.signature.output,
            "nested graph",
            children,
        )
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

    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        let children = children.collect_vec();
        // The first input to the ɣ-node is a predicate of Sum type,
        // whose arity matches the number of children of the ɣ-node.
        if self.predicate_inputs.len() != children.len() {
            return Err(ChildrenValidationError::InvalidConditionalPredicate {
                child: children[0].0, // Pass an arbitrary child
                expected_count: children.len(),
                actual_count: self.predicate_inputs.len(),
                actual_predicate_rows: self.predicate_inputs.clone(),
            });
        }

        // Each child must have its predicate variant's row and the rest of `inputs` as input,
        // and matching output
        for (i, (child, optype)) in children.into_iter().enumerate() {
            let OpType::Case(case_op) = optype else {panic!("Child check should have already checked valid ops.")};
            let sig = &case_op.signature;
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

impl ValidateOp for super::TailLoop {
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

    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        validate_io_nodes(
            &self.body_input_row(),
            &self.body_output_row(),
            "tail-controlled loop graph",
            children,
        )
    }
}

impl ValidateOp for super::CFG {
    fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_children: OpTag::BasicBlock,
            allowed_second_child: OpTag::BasicBlockExit,
            requires_children: true,
            requires_dag: false,
            edge_check: Some(validate_cfg_edge),
            ..Default::default()
        }
    }

    fn validate_children<'a>(
        &self,
        children: impl Iterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        for (child, optype) in children.dropping(2) {
            if optype.tag() == OpTag::BasicBlockExit {
                return Err(ChildrenValidationError::InternalExitChildren { child });
            }
        }
        Ok(())
    }
}
/// Errors that can occur while checking the children of a node.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[allow(missing_docs)]
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
        actual: TypeRow<SimpleType>,
        expected: TypeRow<SimpleType>,
        node_desc: &'static str,
        container_desc: &'static str,
    },
    /// The signature of a child case in a conditional operation does not match the container's signature.
    #[error("A conditional case has optype {optype:?}, which differs from the signature of Conditional container")]
    ConditionalCaseSignature { child: NodeIndex, optype: OpType },
    /// The conditional container's branch predicate does not match the number of children.
    #[error("The conditional container's branch predicate input should be a sum with {expected_count} elements, but it had {actual_count} elements. Predicate rows: {actual_predicate_rows:?} ")]
    InvalidConditionalPredicate {
        child: NodeIndex,
        expected_count: usize,
        actual_count: usize,
        actual_predicate_rows: Vec<TypeRow<ClassicType>>,
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
            ChildrenValidationError::InvalidConditionalPredicate { child, .. } => *child,
        }
    }
}

/// Errors that can occur while checking the edges between children of a node.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[allow(missing_docs)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl ValidateOp for BasicBlock {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        match self {
            BasicBlock::DFB {
                predicate_variants, ..
            } => OpValidityFlags {
                allowed_children: OpTag::DataflowChild,
                allowed_first_child: OpTag::Input,
                allowed_second_child: OpTag::Output,
                requires_children: true,
                requires_dag: true,
                non_df_ports: (None, Some(predicate_variants.len())),
                ..Default::default()
            },
            // Default flags are valid for non-container operations
            BasicBlock::Exit { .. } => Default::default(),
        }
    }

    /// Validate the ordered list of children.
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            BasicBlock::DFB {
                inputs,
                predicate_variants,
                other_outputs: outputs,
            } => {
                let predicate_type = SimpleType::new_predicate(predicate_variants.clone());
                let node_outputs: TypeRow<SimpleType> =
                    [&[predicate_type], outputs.as_ref()].concat().into();
                validate_io_nodes(inputs, &node_outputs, "basic block graph", children)
            }
            // Exit nodes do not have children
            BasicBlock::Exit { .. } => Ok(()),
        }
    }
}

impl ValidateOp for super::Case {
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
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        validate_io_nodes(
            &self.signature.input,
            &self.signature.output,
            "Conditional",
            children,
        )
    }
}

/// Checks a that the list of children nodes does not contain Input and Output
/// nodes outside of the first and second elements respectively, and that those
/// have the correct signature.
fn validate_io_nodes<'a>(
    expected_input: &TypeRow<SimpleType>,
    expected_output: &TypeRow<SimpleType>,
    container_desc: &'static str,
    mut children: impl Iterator<Item = (NodeIndex, &'a OpType)>,
) -> Result<(), ChildrenValidationError> {
    // Check that the signature matches with the Input and Output rows.
    let (first, first_optype) = children.next().unwrap();
    let (second, second_optype) = children.next().unwrap();

    if &first_optype.signature().output != expected_input {
        return Err(ChildrenValidationError::IOSignatureMismatch {
            child: first,
            actual: first_optype.signature().output,
            expected: expected_input.clone(),
            node_desc: "Input",
            container_desc,
        });
    }
    if &second_optype.signature().input != expected_output {
        return Err(ChildrenValidationError::IOSignatureMismatch {
            child: second,
            actual: second_optype.signature().input,
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
    let [source, target]: [&BasicBlock; 2] = [&edge.source_op, &edge.target_op].map(|op| {
        let OpType::BasicBlock(block_op) = op else {panic!("CFG sibling graphs can only contain basic block operations.")};
        block_op

    });

    if source.successor_input(edge.source_port.index()).as_ref() != Some(target.dataflow_input()) {
        return Err(EdgeValidationError::CFGEdgeSignatureMismatch { edge });
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::ops;
    use crate::{
        ops::dataflow::IOTrait,
        ops::LeafOp,
        type_row,
        types::{ClassicType, SimpleType},
    };
    use cool_asserts::assert_matches;

    use super::*;

    #[test]
    fn test_validate_io_nodes() {
        const B: SimpleType = SimpleType::Classic(ClassicType::bit());

        let in_types = type_row![B];
        let out_types = type_row![B, B];

        let input_node: OpType = ops::Input::new(in_types.clone()).into();
        let output_node = ops::Output::new(out_types.clone()).into();
        let leaf_node = LeafOp::Noop {
            ty: ClassicType::bit().into(),
        }
        .into();

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
    AliasDecl, AliasDefn, Call, CallIndirect, Const, FuncDecl, Input, LeafOp, LoadConstant, Output,
};
impl_validate_op!(FuncDecl);
impl_validate_op!(AliasDecl);
impl_validate_op!(AliasDefn);
impl_validate_op!(Input);
impl_validate_op!(Output);
impl_validate_op!(Const);
impl_validate_op!(Call);
impl_validate_op!(CallIndirect);
impl_validate_op!(LoadConstant);
impl_validate_op!(LeafOp);
