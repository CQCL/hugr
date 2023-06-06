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

use crate::types::{SimpleType, TypeRow};

use super::{
    controlflow::{CaseOp, ConditionalSignature, TailLoopSignature},
    tag::OpTag,
    BasicBlockOp, ControlFlowOp, DataflowOp, OpType,
};

/// A set of property flags required for an operation.
#[non_exhaustive]
pub struct OpValidityFlags {
    /// The set of valid children operation types.
    pub allowed_children: OpTag,
    /// Additional restrictions on the first child operation.
    ///
    /// This is checked in addition to the child allowing the parent optype.
    pub allowed_first_child: OpTag,
    /// Additional restrictions on the last child operation
    ///
    /// This is checked in addition to the child allowing the parent optype.
    pub allowed_last_child: OpTag,
    /// Whether the operation must have children.
    pub requires_children: bool,
    /// Whether the children must form a DAG (no cycles).
    pub requires_dag: bool,
    /// A strict requirement on the number of non-dataflow input and output wires.
    pub non_df_ports: (Option<usize>, Option<usize>),
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
            allowed_last_child: OpTag::Any,
            requires_children: false,
            requires_dag: false,
            non_df_ports: (None, None),
            edge_check: None,
        }
    }
}

impl OpType {
    /// Returns a set of flags describing the validity predicates for this operation.
    #[inline]
    pub fn validity_flags(&self) -> OpValidityFlags {
        match self {
            OpType::Root { .. } => OpValidityFlags {
                allowed_children: OpTag::ModuleOp,
                requires_children: false,
                ..Default::default()
            },
            OpType::Def { .. } => OpValidityFlags {
                allowed_children: OpTag::DataflowOp,
                allowed_first_child: OpTag::Input,
                allowed_last_child: OpTag::Output,
                requires_children: true,
                requires_dag: true,
                ..Default::default()
            },
            // Default flags are valid for non-container operations
            // _ => Default::default(),
            OpType::Dataflow(op) => op.validity_flags(),
            OpType::BasicBlock(op) => op.validity_flags(),
            OpType::Case(op) => op.validity_flags(),
            OpType::Declare { .. }
            | OpType::AliasDeclare { .. }
            | OpType::AliasDef { .. }
            | OpType::Const(_) => Default::default(),
        }
    }

    /// Validate the ordered list of children.
    #[inline]
    pub fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            OpType::Def { signature } => validate_io_nodes(
                &signature.input,
                &signature.output,
                "function definition",
                children,
            ),
            OpType::Dataflow(op) => op.validate_children(children),
            OpType::BasicBlock(op) => op.validate_children(children),
            OpType::Case(op) => op.validate_children(children),
            _ => Ok(()),
        }
    }
}

/// Errors that can occur while checking the children of a node.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[allow(missing_docs)]
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
    /// The signature of a child case in a conditional operation does not match the container's signature.
    #[error("A conditional case has optype {optype:?}, which differs from the signature of Conditional container")]
    ConditionalCaseSignature { child: NodeIndex, optype: OpType },
    /// The conditional container's branch predicate does not match the number of children.
    #[error("The conditional container's branch predicate input should be a sum with {expected_count} elements, but it had {actual_count} elements. Predicate rows: {actual_predicate_rows:?} ")]
    InvalidConditionalPredicate {
        child: NodeIndex,
        expected_count: usize,
        actual_count: usize,
        actual_predicate_rows: Vec<TypeRow>,
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

impl BasicBlockOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        match self {
            BasicBlockOp::Block {
                predicate_variants, ..
            } => OpValidityFlags {
                allowed_children: OpTag::DataflowOp,
                allowed_first_child: OpTag::Input,
                allowed_last_child: OpTag::Output,
                requires_children: true,
                requires_dag: true,
                non_df_ports: (None, Some(predicate_variants.len())),
                ..Default::default()
            },
            // Default flags are valid for non-container operations
            BasicBlockOp::Exit { .. } => Default::default(),
        }
    }

    /// Validate the ordered list of children.
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            BasicBlockOp::Block {
                inputs,
                predicate_variants,
                other_outputs: outputs,
            } => {
                let predicate_type = SimpleType::new_predicate(predicate_variants.clone());
                let node_outputs: TypeRow = [&[predicate_type], outputs.as_ref()].concat().into();
                validate_io_nodes(inputs, &node_outputs, "basic block graph", children)
            }
            // Exit nodes do not have children
            BasicBlockOp::Exit { .. } => Ok(()),
        }
    }
}

impl CaseOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        OpValidityFlags {
            allowed_children: OpTag::DataflowOp,
            allowed_first_child: OpTag::Input,
            allowed_last_child: OpTag::Output,
            requires_children: true,
            requires_dag: true,
            non_df_ports: (Some(0), Some(0)),
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

impl DataflowOp {
    /// Returns the set of allowed parent operation types.
    fn validity_flags(&self) -> OpValidityFlags {
        match self {
            DataflowOp::ControlFlow { op } => op.validity_flags(),
            DataflowOp::DFG { .. } => OpValidityFlags {
                allowed_children: OpTag::DataflowOp,
                allowed_first_child: OpTag::Input,
                allowed_last_child: OpTag::Output,
                requires_children: true,
                requires_dag: true,
                ..Default::default()
            },
            // Default flags are valid for non-container operations
            _ => Default::default(),
        }
    }

    /// Validate the ordered list of children.
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            DataflowOp::ControlFlow { op } => op.validate_children(children),
            DataflowOp::DFG { signature } => validate_io_nodes(
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
                allowed_children: OpTag::Case,
                requires_children: true,
                requires_dag: false,
                ..Default::default()
            },
            ControlFlowOp::TailLoop { .. } => OpValidityFlags {
                allowed_children: OpTag::DataflowOp,
                allowed_first_child: OpTag::Input,
                allowed_last_child: OpTag::Output,
                requires_children: true,
                requires_dag: true,
                ..Default::default()
            },
            ControlFlowOp::CFG { .. } => OpValidityFlags {
                allowed_children: OpTag::BasicBlock,
                allowed_last_child: OpTag::BasicBlockExit,
                requires_children: true,
                requires_dag: false,
                edge_check: Some(validate_cfg_edge),
                ..Default::default()
            },
        }
    }

    /// Validate the ordered list of children.
    fn validate_children<'a>(
        &self,
        children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), ChildrenValidationError> {
        match self {
            ControlFlowOp::Conditional(ConditionalSignature {
                predicate_inputs,
                other_inputs,
                outputs,
            }) => {
                let children = children.collect_vec();
                // The first input to the ɣ-node is a predicate of Sum type,
                // whose arity matches the number of children of the ɣ-node.
                if predicate_inputs.len() != children.len() {
                    return Err(ChildrenValidationError::InvalidConditionalPredicate {
                        child: children[0].0, // Pass an arbitrary child
                        expected_count: children.len(),
                        actual_count: predicate_inputs.len(),
                        actual_predicate_rows: predicate_inputs.clone(),
                    });
                }

                // Each child must have its predicate variant's row and the rest of `inputs` as input,
                // and matching output
                for (i, (child, optype)) in children.into_iter().enumerate() {
                    let case_op: &CaseOp = optype
                        .try_into()
                        .expect("Child check should have already checked valid ops.");
                    let sig = &case_op.signature;
                    let predicate_value = &predicate_inputs[i];
                    if sig.input[0..predicate_value.len()] != predicate_value[..]
                        || sig.input[predicate_value.len()..] != other_inputs[..]
                        || sig.output != *outputs
                    {
                        return Err(ChildrenValidationError::ConditionalCaseSignature {
                            child,
                            optype: optype.clone(),
                        });
                    }
                }
            }
            ControlFlowOp::TailLoop(TailLoopSignature {
                just_inputs,
                just_outputs,
                rest,
            }) => {
                let expected_output = SimpleType::new_sum(vec![
                    SimpleType::new_tuple(just_inputs.clone()),
                    SimpleType::new_tuple(just_outputs.clone()),
                ]);
                let mut expected_output = vec![expected_output];
                expected_output.extend_from_slice(rest);
                let expected_output: TypeRow = expected_output.into();

                let mut expected_input = just_inputs.clone();
                expected_input.to_mut().extend_from_slice(rest);

                validate_io_nodes(
                    &expected_input,
                    &expected_output,
                    "tail-controlled loop graph",
                    children,
                )?;
            }
            ControlFlowOp::CFG { .. } => {
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
            OpType::Dataflow(DataflowOp::Input { .. }) => {
                return Err(ChildrenValidationError::InternalIOChildren {
                    child,
                    optype: optype.clone(),
                    expected_position: "first",
                })
            }
            OpType::Dataflow(DataflowOp::Output { .. }) => {
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

/// Validate an edge between two basic blocks in a CFG sibling graph.
fn validate_cfg_edge(edge: ChildrenEdgeData) -> Result<(), EdgeValidationError> {
    let [source, target]: [&BasicBlockOp; 2] = [&edge.source_op, &edge.target_op].map(|op| {
        op.try_into()
            .expect("CFG sibling graphs can only contain basic block operations.")
    });

    if source.successor_input(edge.source_port.index()).as_ref() != Some(target.dataflow_input()) {
        return Err(EdgeValidationError::CFGEdgeSignatureMismatch { edge });
    }

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

        let input_node = OpType::Dataflow(DataflowOp::Input {
            types: in_types.clone(),
        });
        let output_node = OpType::Dataflow(DataflowOp::Output {
            types: out_types.clone(),
        });
        let leaf_node = OpType::Dataflow(DataflowOp::Leaf {
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
