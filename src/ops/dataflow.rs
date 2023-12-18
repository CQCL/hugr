//! Dataflow operations.

use super::{impl_op_name, OpTag, OpTrait};

use crate::extension::ExtensionSet;
use crate::ops::StaticTag;
use crate::types::{EdgeKind, FunctionType, Type, TypeRow};
use crate::IncomingPort;

pub(crate) trait DataflowOpTrait {
    const TAG: OpTag;
    fn description(&self) -> &str;
    fn signature(&self) -> FunctionType;

    /// The edge kind for the non-dataflow or constant inputs of the operation,
    /// not described by the signature.
    ///
    /// If not None, a single extra output multiport of that kind will be
    /// present.
    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
    /// The edge kind for the non-dataflow outputs of the operation, not
    /// described by the signature.
    ///
    /// If not None, a single extra output multiport of that kind will be
    /// present.
    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}

/// Helpers to construct input and output nodes
pub trait IOTrait {
    /// Construct a new I/O node from a type row with no extension requirements
    fn new(types: impl Into<TypeRow>) -> Self;
}

/// An input node.
/// The outputs of this node are the inputs to the function.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Input {
    /// Input value types
    pub types: TypeRow,
}

impl_op_name!(Input);

impl IOTrait for Input {
    fn new(types: impl Into<TypeRow>) -> Self {
        Input {
            types: types.into(),
        }
    }
}

/// An output node. The inputs are the outputs of the function.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Output {
    /// Output value types
    pub types: TypeRow,
}

impl_op_name!(Output);

impl IOTrait for Output {
    fn new(types: impl Into<TypeRow>) -> Self {
        Output {
            types: types.into(),
        }
    }
}

impl DataflowOpTrait for Input {
    const TAG: OpTag = OpTag::Input;

    fn description(&self) -> &str {
        "The input node for this dataflow subgraph"
    }

    fn other_input(&self) -> Option<EdgeKind> {
        None
    }

    fn signature(&self) -> FunctionType {
        FunctionType::new(TypeRow::new(), self.types.clone())
            .with_extension_delta(&ExtensionSet::new())
    }
}
impl DataflowOpTrait for Output {
    const TAG: OpTag = OpTag::Output;

    fn description(&self) -> &str {
        "The output node for this dataflow subgraph"
    }

    // Note: We know what the input extensions should be, so we *could* give an
    // instantiated Signature instead
    fn signature(&self) -> FunctionType {
        FunctionType::new(self.types.clone(), TypeRow::new())
    }

    fn other_output(&self) -> Option<EdgeKind> {
        None
    }
}

impl<T: DataflowOpTrait> OpTrait for T {
    fn description(&self) -> &str {
        DataflowOpTrait::description(self)
    }
    fn tag(&self) -> OpTag {
        T::TAG
    }
    fn dataflow_signature(&self) -> Option<FunctionType> {
        Some(DataflowOpTrait::signature(self))
    }
    fn extension_delta(&self) -> ExtensionSet {
        DataflowOpTrait::signature(self).extension_reqs.clone()
    }
    fn other_input(&self) -> Option<EdgeKind> {
        DataflowOpTrait::other_input(self)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        DataflowOpTrait::other_output(self)
    }
}
impl<T: DataflowOpTrait> StaticTag for T {
    const TAG: OpTag = T::TAG;
}

/// Call a function directly.
///
/// The first ports correspond to the signature of the function being
/// called. Immediately following those ports, the first input port is
/// connected to the def/declare block with a `ConstE<Graph>` edge.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Call {
    /// Signature of function being called
    pub signature: FunctionType,
}
impl_op_name!(Call);

impl DataflowOpTrait for Call {
    const TAG: OpTag = OpTag::FnCall;

    fn description(&self) -> &str {
        "Call a function directly"
    }

    fn signature(&self) -> FunctionType {
        self.signature.clone()
    }
}
impl Call {
    #[inline]
    /// Return the signature of the function called by this op.
    pub fn called_function_type(&self) -> &FunctionType {
        &self.signature
    }

    /// The IncomingPort which links to the function being called.
    #[inline]
    pub fn called_function_port(&self) -> IncomingPort {
        self.called_function_type().input_count().into()
    }
}

/// Call a function indirectly. Like call, but the first input is a standard dataflow graph type.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CallIndirect {
    /// Signature of function being called
    pub signature: FunctionType,
}
impl_op_name!(CallIndirect);

impl DataflowOpTrait for CallIndirect {
    const TAG: OpTag = OpTag::FnCall;

    fn description(&self) -> &str {
        "Call a function indirectly"
    }

    fn signature(&self) -> FunctionType {
        let mut s = self.signature.clone();
        s.input
            .to_mut()
            .insert(0, Type::new_function(self.signature.clone()));
        s
    }
}

/// Load a static constant in to the local dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LoadConstant {
    /// Constant type
    pub datatype: Type,
}
impl_op_name!(LoadConstant);
impl DataflowOpTrait for LoadConstant {
    const TAG: OpTag = OpTag::LoadConst;

    fn description(&self) -> &str {
        "Load a static constant in to the local dataflow graph"
    }

    fn signature(&self) -> FunctionType {
        FunctionType::new(TypeRow::new(), vec![self.datatype.clone()])
    }
}
impl LoadConstant {
    #[inline]
    /// The type of the constant loaded by this op.
    pub fn constant_type(&self) -> &Type {
        &self.datatype
    }

    /// The IncomingPort which links to the loaded constant.
    #[inline]
    pub fn constant_port(&self) -> IncomingPort {
        0.into()
    }
}

/// A simply nested dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DFG {
    /// Signature of DFG node
    pub signature: FunctionType,
}

impl_op_name!(DFG);
impl DataflowOpTrait for DFG {
    const TAG: OpTag = OpTag::Dfg;

    fn description(&self) -> &str {
        "A simply nested dataflow graph"
    }

    fn signature(&self) -> FunctionType {
        self.signature.clone()
    }
}
