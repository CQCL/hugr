//! Dataflow operations.

use super::{impl_op_name, tag::OpTag, OpTrait};

use crate::resource::ResourceSet;
use crate::types::{
    AbstractSignature, ClassicType, EdgeKind, SignatureTrait, SimpleType, TypeRow,
};

pub(super) trait DataflowOpTrait {
    fn description(&self) -> &str;
    fn tag(&self) -> OpTag;
    fn signature(&self) -> AbstractSignature;
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
    fn description(&self) -> &str {
        "The input node for this dataflow subgraph"
    }

    fn tag(&self) -> super::tag::OpTag {
        OpTag::Input
    }

    fn other_input(&self) -> Option<EdgeKind> {
        None
    }

    fn signature(&self) -> AbstractSignature {
        AbstractSignature::new_df(TypeRow::new(), self.types.clone())
            .with_resource_delta(&self.resources)
    }
}
impl DataflowOpTrait for Output {
    fn description(&self) -> &str {
        "The output node for this dataflow subgraph"
    }

    fn tag(&self) -> super::tag::OpTag {
        OpTag::Output
    }

    // TODO: We know what the input resources should be, so we *could* give an
    // instantiated Signature instead
    fn signature(&self) -> AbstractSignature {
        AbstractSignature::new_df(self.types.clone(), TypeRow::new())
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
        DataflowOpTrait::tag(self)
    }
    fn op_signature(&self) -> AbstractSignature {
        DataflowOpTrait::signature(self)
    }
    fn other_input(&self) -> Option<EdgeKind> {
        DataflowOpTrait::other_input(self)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        DataflowOpTrait::other_output(self)
    }
}

/// Call a function directly.
///
/// The first ports correspond to the signature of the function being
/// called. Immediately following those ports, the first input port is
/// connected to the def/declare block with a `ConstE<Graph>` edge.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Call {
    /// Signature of function being called
    pub signature: AbstractSignature,
}
impl_op_name!(Call);

impl DataflowOpTrait for Call {
    fn description(&self) -> &str {
        "Call a function directly"
    }

    fn tag(&self) -> OpTag {
        OpTag::FnCall
    }

    fn signature(&self) -> AbstractSignature {
        AbstractSignature {
            static_input: vec![ClassicType::graph_from_sig(self.signature.clone()).into()].into(),
            ..self.signature.clone()
        }
    }
}

/// Call a function indirectly. Like call, but the first input is a standard dataflow graph type.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CallIndirect {
    /// Signature of function being called
    pub signature: AbstractSignature,
}
impl_op_name!(CallIndirect);

impl DataflowOpTrait for CallIndirect {
    fn description(&self) -> &str {
        "Call a function indirectly"
    }

    fn tag(&self) -> OpTag {
        OpTag::FnCall
    }

    fn signature(&self) -> AbstractSignature {
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
    fn description(&self) -> &str {
        "Load a static constant in to the local dataflow graph"
    }

    fn tag(&self) -> OpTag {
        OpTag::LoadConst
    }

    fn signature(&self) -> AbstractSignature {
        AbstractSignature::new(
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
    pub signature: AbstractSignature,
}

impl_op_name!(DFG);
impl DataflowOpTrait for DFG {
    fn description(&self) -> &str {
        "A simply nested dataflow graph"
    }

    fn tag(&self) -> OpTag {
        OpTag::Dfg
    }

    fn signature(&self) -> AbstractSignature {
        self.signature.clone()
    }
}
