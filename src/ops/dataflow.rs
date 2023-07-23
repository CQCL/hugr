//! Dataflow operations.

use super::StaticTag;
use super::{impl_op_name, OpTag, OpTrait};

use crate::resource::ResourceSet;
use crate::types::{ClassicType, EdgeKind, Signature, SimpleType, TypeRow};

pub(super) trait DataflowOpTrait {
    const TAG: OpTag;
    fn description(&self) -> &str;
    fn signature(&self) -> Signature;
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
    /// Construct a new I/O node from a type row with no resource requirements
    fn new(types: impl Into<TypeRow>) -> Self;
    /// Helper method to add resource requirements to an I/O node
    fn with_resources(self, rs: ResourceSet) -> Self;
}

/// An input node.
/// The outputs of this node are the inputs to the function.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Input {
    /// Input value types
    pub types: TypeRow,
    /// Resources attached to output wires
    pub resources: ResourceSet,
}

impl_op_name!(Input);

impl IOTrait for Input {
    fn new(types: impl Into<TypeRow>) -> Self {
        Input {
            types: types.into(),
            resources: ResourceSet::new(),
        }
    }

    fn with_resources(mut self, resources: ResourceSet) -> Self {
        self.resources = resources;
        self
    }
}

/// An output node. The inputs are the outputs of the function.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Output {
    /// Output value types
    pub types: TypeRow,
    /// Resources expected from input wires
    pub resources: ResourceSet,
}

impl_op_name!(Output);

impl IOTrait for Output {
    fn new(types: impl Into<TypeRow>) -> Self {
        Output {
            types: types.into(),
            resources: ResourceSet::new(),
        }
    }

    fn with_resources(mut self, resources: ResourceSet) -> Self {
        self.resources = resources;
        self
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

    fn signature(&self) -> Signature {
        let mut sig = Signature::new_df(TypeRow::new(), self.types.clone());
        sig.output_resources = self.resources.clone();
        sig
    }
}
impl DataflowOpTrait for Output {
    const TAG: OpTag = OpTag::Output;

    fn description(&self) -> &str {
        "The output node for this dataflow subgraph"
    }

    fn signature(&self) -> Signature {
        let mut sig = Signature::new_df(self.types.clone(), TypeRow::new());
        sig.input_resources = self.resources.clone();
        sig
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
    fn signature(&self) -> Signature {
        DataflowOpTrait::signature(self)
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
    pub signature: Signature,
}
impl_op_name!(Call);

impl DataflowOpTrait for Call {
    const TAG: OpTag = OpTag::FnCall;

    fn description(&self) -> &str {
        "Call a function directly"
    }

    fn signature(&self) -> Signature {
        Signature {
            static_input: vec![ClassicType::graph_from_sig(self.signature.clone()).into()].into(),
            ..self.signature.clone()
        }
    }
}

/// Call a function indirectly. Like call, but the first input is a standard dataflow graph type.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CallIndirect {
    /// Signature of function being called
    pub signature: Signature,
}
impl_op_name!(CallIndirect);

impl DataflowOpTrait for CallIndirect {
    const TAG: OpTag = OpTag::FnCall;

    fn description(&self) -> &str {
        "Call a function indirectly"
    }

    fn signature(&self) -> Signature {
        let mut s = self.signature.clone();
        s.input.to_mut().insert(
            0,
            ClassicType::graph_from_sig(self.signature.clone()).into(),
        );
        s
    }
}

/// Load a static constant in to the local dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LoadConstant {
    /// Constant type
    pub datatype: ClassicType,
}
impl_op_name!(LoadConstant);
impl DataflowOpTrait for LoadConstant {
    const TAG: OpTag = OpTag::LoadConst;

    fn description(&self) -> &str {
        "Load a static constant in to the local dataflow graph"
    }

    fn signature(&self) -> Signature {
        Signature::new(
            TypeRow::new(),
            vec![SimpleType::Classic(self.datatype.clone())],
            vec![SimpleType::Classic(self.datatype.clone())],
        )
    }
}

/// A simply nested dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DFG {
    /// Signature of DFG node
    pub signature: Signature,
}

impl_op_name!(DFG);
impl DataflowOpTrait for DFG {
    const TAG: OpTag = OpTag::Dfg;

    fn description(&self) -> &str {
        "A simply nested dataflow graph"
    }

    fn signature(&self) -> Signature {
        self.signature.clone()
    }
}
