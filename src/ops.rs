//! The operation types for the HUGR.

pub mod controlflow;
pub mod custom;
pub mod dataflow;
pub mod handle;
pub mod leaf;
pub mod module;
pub mod tag;
pub mod validate;

use crate::types::{ClassicType, EdgeKind, Signature, SignatureDescription, SimpleType, TypeRow};
use crate::{Direction, Port};

pub use controlflow::{BasicBlockOp, CaseOp, ControlFlowOp};
pub use custom::{CustomOp, OpDef, OpaqueOp};
pub use dataflow::DataflowOp;
pub use leaf::LeafOp;
pub use module::ConstValue;

use smol_str::SmolStr;

use self::tag::OpTag;

/// The concrete operation types for a node in the HUGR.
// TODO: Link the NodeHandles to the OpType.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum OpType {
    #[default]
    /// The root of a module, parent of all other `OpType`s.
    Root,
    /// A function definition.
    ///
    /// Children nodes are the body of the definition.
    Def {
        signature: Signature,
    },
    /// External function declaration, linked at runtime.
    Declare {
        signature: Signature,
    },
    /// A type alias declaration. Resolved at link time.
    AliasDeclare {
        name: SmolStr,
        linear: bool,
    },
    /// A type alias definition, used only for debug/metadata.
    AliasDef {
        name: SmolStr,
        definition: SimpleType,
    },
    // A constant value definition.
    Const(ConstValue),
    /// A module region node - parent will be the Root (or the node itself is the Root).
    // Module(OpType),
    /// A basic block in a control flow graph - parent will be a CFG node.
    BasicBlock(BasicBlockOp),
    /// A branch in a dataflow graph - parent will be a Conditional node.
    Case(CaseOp),
    /// Nodes used inside dataflow containers
    /// (DFG, Conditional, TailLoop, def, BasicBlock).
    Dataflow(DataflowOp),
}

impl OpType {
    /// The name of the operation.
    pub fn name(&self) -> SmolStr {
        match self {
            OpType::Root => "module".into(),
            OpType::Def { .. } => "def".into(),
            OpType::Declare { .. } => "declare".into(),
            OpType::AliasDeclare { .. } => "alias_declare".into(),
            OpType::AliasDef { .. } => "alias_def".into(),
            OpType::Const(val) => return val.name(),

            OpType::BasicBlock(op) => {
                let ref this = op;
                match this {
                    BasicBlockOp::Block { .. } => "BasicBlock".into(),
                    BasicBlockOp::Exit { .. } => "ExitBlock".into(),
                }
            }
            OpType::Case(op) => {
                let ref this = op;
                "Case".into()
            }
            OpType::Dataflow(op) => {
                let ref this = op;
                match this {
                    DataflowOp::Input { .. } => "input",
                    DataflowOp::Output { .. } => "output",
                    DataflowOp::Call { .. } => "call",
                    DataflowOp::CallIndirect { .. } => "call_indirect",
                    DataflowOp::LoadConstant { .. } => "load",
                    DataflowOp::Leaf { op } => return op.name(),
                    DataflowOp::DFG { .. } => "DFG",
                    DataflowOp::ControlFlow { op } => return op.name(),
                }
                .into()
            }
        }
    }

    /// A human-readable description of the operation.
    pub fn description(&self) -> &str {
        match self {
            OpType::Root => "The root of a module, parent of all other `OpType`s",
            OpType::Def { .. } => "A function definition",
            OpType::Declare { .. } => "External function declaration, linked at runtime",
            OpType::AliasDeclare { .. } => "A type alias declaration",
            OpType::AliasDef { .. } => "A type alias definition",
            OpType::Const(val) => val.description(),

            OpType::BasicBlock(op) => {
                let ref this = op;
                match this {
                    BasicBlockOp::Block { .. } => "A CFG basic block node",
                    BasicBlockOp::Exit { .. } => "A CFG exit block node",
                }
            }
            OpType::Case(op) => {
                let ref this = op;
                "A case node inside a conditional"
            }
            OpType::Dataflow(op) => {
                let ref this = op;
                match this {
                    DataflowOp::Input { .. } => "The input node for this dataflow subgraph",
                    DataflowOp::Output { .. } => "The output node for this dataflow subgraph",
                    DataflowOp::Call { .. } => "Call a function directly",
                    DataflowOp::CallIndirect { .. } => "Call a function indirectly",
                    DataflowOp::LoadConstant { .. } => {
                        "Load a static constant in to the local dataflow graph"
                    }
                    DataflowOp::Leaf { op } => return op.description(),
                    DataflowOp::DFG { .. } => "A simply nested dataflow graph",
                    DataflowOp::ControlFlow { op } => return op.description(),
                }
            }
        }
    }

    /// Tag identifying the operation.
    pub fn tag(&self) -> OpTag {
        match self {
            OpType::Root => OpTag::ModuleRoot,
            OpType::Def { .. } => OpTag::Def,
            OpType::Declare { .. } => OpTag::Function,
            OpType::AliasDeclare { .. } => OpTag::Alias,
            OpType::AliasDef { .. } => OpTag::Alias,
            OpType::Const { .. } => OpTag::Const,

            OpType::BasicBlock(op) => {
                let ref this = op;
                match this {
                    BasicBlockOp::Block { .. } => OpTag::BasicBlock,
                    BasicBlockOp::Exit { .. } => OpTag::BasicBlockExit,
                }
            }
            OpType::Case(op) => {
                let ref this = op;
                OpTag::Case
            }
            OpType::Dataflow(op) => {
                let ref this = op;
                match this {
                    DataflowOp::Input { .. } => OpTag::Input,
                    DataflowOp::Output { .. } => OpTag::Output,
                    DataflowOp::Call { .. } | DataflowOp::CallIndirect { .. } => OpTag::FnCall,
                    DataflowOp::LoadConstant { .. } => OpTag::LoadConst,
                    DataflowOp::Leaf { .. } => OpTag::Leaf,
                    DataflowOp::DFG { .. } => OpTag::Dfg,
                    DataflowOp::ControlFlow { op } => op.tag(),
                }
            }
        }
    }

    /// The signature of the operation.
    ///
    /// Only dataflow operations have a non-empty signature.
    pub fn signature(&self) -> Signature {
        match self {
            OpType::Dataflow(op) => {
                let ref this = op;
                match this {
                    DataflowOp::Input { types } => Signature::new_df(TypeRow::new(), types.clone()),
                    DataflowOp::Output { types } => {
                        Signature::new_df(types.clone(), TypeRow::new())
                    }
                    DataflowOp::Call { signature } => Signature {
                        const_input: vec![ClassicType::graph_from_sig(signature.clone()).into()]
                            .into(),
                        ..signature.clone()
                    },
                    DataflowOp::CallIndirect { signature } => {
                        let mut s = signature.clone();
                        s.input
                            .to_mut()
                            .insert(0, ClassicType::graph_from_sig(signature.clone()).into());
                        s
                    }
                    DataflowOp::LoadConstant { datatype } => Signature::new(
                        TypeRow::new(),
                        vec![SimpleType::Classic(datatype.clone())],
                        vec![SimpleType::Classic(datatype.clone())],
                    ),
                    DataflowOp::Leaf { op } => op.signature(),
                    DataflowOp::DFG { signature } => signature.clone(),
                    DataflowOp::ControlFlow { op } => op.signature(),
                }
            }
            _ => Default::default(),
        }
    }

    /// Optional description of the ports in the signature.
    ///
    /// Only dataflow operations have a non-empty signature.
    pub fn signature_desc(&self) -> SignatureDescription {
        match self {
            OpType::Dataflow(op) => op.signature_desc(),
            _ => Default::default(),
        }
    }

    /// The edge kind for the inputs of the operation not described by the
    /// signature.
    ///
    /// If None, there will be no other input edges. Otherwise, all other input
    /// edges will be of that kind.
    pub fn other_inputs(&self) -> Option<EdgeKind> {
        match self {
            OpType::Dataflow(op) => {
                let ref this = op;
                if let DataflowOp::Input { .. } = this {
                    None
                } else {
                    Some(EdgeKind::StateOrder)
                }
            }
            OpType::BasicBlock(op) => {
                let ref this = op;
                Some(EdgeKind::ControlFlow)
            }
            OpType::Case(op) => {
                let ref this = op;
                None
            }
            _ => None,
        }
    }

    /// The edge kind for the outputs of the operation not described by the
    /// signature.
    ///
    /// If None, there will be no other output edges. Otherwise, all other
    /// output edges will be of that kind.
    pub fn other_outputs(&self) -> Option<EdgeKind> {
        match self {
            OpType::Root | OpType::AliasDeclare { .. } | OpType::AliasDef { .. } => None,
            OpType::Def { signature } | OpType::Declare { signature } => Some(EdgeKind::Const(
                ClassicType::graph_from_sig(signature.clone()),
            )),
            OpType::Const(v) => Some(EdgeKind::Const(v.const_type())),

            OpType::Dataflow(op) => {
                let ref this = op;
                if let DataflowOp::Output { .. } = this {
                    None
                } else {
                    Some(EdgeKind::StateOrder)
                }
            }
            OpType::BasicBlock(op) => {
                let ref this = op;
                Some(EdgeKind::ControlFlow)
            }
            OpType::Case(op) => {
                let ref this = op;
                None
            }
        }
    }

    /// Returns the edge kind for the given port.
    pub fn port_kind(&self, port: impl Into<Port>) -> Option<EdgeKind> {
        let signature = self.signature();
        let port = port.into();
        if let Some(port_kind) = signature.get(port) {
            Some(port_kind)
        } else if port.direction() == Direction::Incoming {
            self.other_inputs()
        } else {
            self.other_outputs()
        }
    }
}

// impl From<OpType> for OpType {
//     fn from(op: OpType) -> Self {
//         Self::Module(op)
//     }
// }

impl From<BasicBlockOp> for OpType {
    fn from(op: BasicBlockOp) -> Self {
        Self::BasicBlock(op)
    }
}

impl<T> From<T> for OpType
where
    T: Into<DataflowOp>,
{
    fn from(op: T) -> Self {
        Self::Dataflow(op.into())
    }
}

impl From<CaseOp> for OpType {
    fn from(op: CaseOp) -> Self {
        OpType::Case(op)
    }
}

/// Implementations of TryFrom for OpType and &'a OpType for each variant.
macro_rules! impl_try_from_optype {
    ($target:ident, $matcher:pat, $unpack:expr) => {
        impl TryFrom<OpType> for $target {
            type Error = ();

            fn try_from(op: OpType) -> Result<Self, Self::Error> {
                match op {
                    $matcher => Ok($unpack),
                    _ => Err(()),
                }
            }
        }

        impl<'a> TryFrom<&'a OpType> for &'a $target {
            type Error = ();

            fn try_from(op: &'a OpType) -> Result<Self, Self::Error> {
                match op {
                    $matcher => Ok($unpack),
                    _ => Err(()),
                }
            }
        }
    };
}
// impl_try_from_optype!(OpType, OpType::Module(op), op);
impl_try_from_optype!(BasicBlockOp, OpType::BasicBlock(op), op);
impl_try_from_optype!(CaseOp, OpType::Case(op), op);
impl_try_from_optype!(DataflowOp, OpType::Dataflow(op), op);
impl_try_from_optype!(
    ControlFlowOp,
    OpType::Dataflow(DataflowOp::ControlFlow { op }),
    op
);
impl_try_from_optype!(LeafOp, OpType::Dataflow(DataflowOp::Leaf { op }), op);
