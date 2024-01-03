//! The operation types for the HUGR.

pub mod constant;
pub mod controlflow;
pub mod custom;
pub mod dataflow;
pub mod handle;
pub mod leaf;
pub mod module;
pub mod tag;
pub mod validate;
use crate::extension::ExtensionSet;
use crate::types::{EdgeKind, FunctionType, Type};
use crate::{Direction, OutgoingPort, Port};
use crate::{IncomingPort, PortIndex};
use paste::paste;

use portgraph::NodeIndex;
use smol_str::SmolStr;

use enum_dispatch::enum_dispatch;

pub use constant::Const;
pub use controlflow::{Case, Conditional, Exit, TailLoop, CFG, DFB};
pub use dataflow::{Call, CallIndirect, DataflowParent, Input, LoadConstant, Output, DFG};
pub use leaf::LeafOp;
pub use module::{AliasDecl, AliasDefn, FuncDecl, FuncDefn, Module};
pub use tag::OpTag;

#[enum_dispatch(OpTrait, OpName, ValidateOp, OpParent)]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
/// The concrete operation types for a node in the HUGR.
// TODO: Link the NodeHandles to the OpType.
#[non_exhaustive]
#[allow(missing_docs)]
#[serde(tag = "op")]
pub enum OpType {
    Module,
    FuncDefn,
    FuncDecl,
    AliasDecl,
    AliasDefn,
    Const,
    Input,
    Output,
    Call,
    CallIndirect,
    LoadConstant,
    DFG,
    LeafOp,
    DFB,
    Exit,
    TailLoop,
    CFG,
    Conditional,
    Case,
}

macro_rules! impl_op_ref_try_into {
    ($Op: tt, $sname:ident) => {
        paste! {
            impl OpType {
                #[doc = "If is an instance of `" $Op "` return a reference to it."]
                pub fn [<as_ $sname:snake>](&self) -> Option<&$Op> {
                    if let OpType::$Op(l) = self {
                        Some(l)
                    } else {
                        None
                    }
                }

                #[doc = "If is an instance of `" $Op "`."]
                pub fn [<is_ $sname:snake>](&self) -> bool {
                    self.[<as_ $sname:snake>]().is_some()
                }
            }
        }
    };
    ($Op:tt) => {
        impl_op_ref_try_into!($Op, $Op);
    };
}

impl_op_ref_try_into!(Module);
impl_op_ref_try_into!(FuncDefn);
impl_op_ref_try_into!(FuncDecl);
impl_op_ref_try_into!(AliasDecl);
impl_op_ref_try_into!(AliasDefn);
impl_op_ref_try_into!(Const);
impl_op_ref_try_into!(Input);
impl_op_ref_try_into!(Output);
impl_op_ref_try_into!(Call);
impl_op_ref_try_into!(CallIndirect);
impl_op_ref_try_into!(LoadConstant);
impl_op_ref_try_into!(DFG, dfg);
impl_op_ref_try_into!(LeafOp);
impl_op_ref_try_into!(DFB, dfb);
impl_op_ref_try_into!(Exit);
impl_op_ref_try_into!(TailLoop);
impl_op_ref_try_into!(CFG, cfg);
impl_op_ref_try_into!(Conditional);
impl_op_ref_try_into!(Case);

/// The default OpType (as returned by [Default::default])
pub const DEFAULT_OPTYPE: OpType = OpType::Module(Module);

impl Default for OpType {
    fn default() -> Self {
        DEFAULT_OPTYPE
    }
}

impl OpType {
    /// The edge kind for the non-dataflow or constant ports of the
    /// operation, not described by the signature.
    ///
    /// If not None, a single extra multiport of that kind will be present on
    /// the given direction.
    pub fn other_port_kind(&self, dir: Direction) -> Option<EdgeKind> {
        match dir {
            Direction::Incoming => self.other_input(),
            Direction::Outgoing => self.other_output(),
        }
    }

    /// Returns the edge kind for the given port.
    pub fn port_kind(&self, port: impl Into<Port>) -> Option<EdgeKind> {
        let signature = self.dataflow_signature().unwrap_or_default();
        let port: Port = port.into();
        let port_as_in = port.as_incoming().ok();
        let dir = port.direction();

        let port_count = signature.port_count(dir);
        if port.index() < port_count {
            signature.port_type(port).cloned().map(EdgeKind::Value)
        } else if port_as_in.is_some() && port_as_in == self.static_input_port() {
            Some(EdgeKind::Static(static_in_type(self)))
        } else {
            self.other_port_kind(dir)
        }
    }

    /// The non-dataflow port for the operation, not described by the signature.
    /// See `[OpType::other_port_kind]`.
    ///
    /// Returns None if there is no such port, or if the operation defines multiple non-dataflow ports.
    pub fn other_port(&self, dir: Direction) -> Option<Port> {
        let non_df_count = self.non_df_port_count(dir);
        if self.other_port_kind(dir).is_some() && non_df_count == 1 {
            // if there is a static input it comes before the non_df_ports
            let static_input =
                (dir == Direction::Incoming && OpTag::StaticInput.is_superset(self.tag())) as usize;

            Some(Port::new(dir, self.value_port_count(dir) + static_input))
        } else {
            None
        }
    }

    /// The number of Value ports in given direction.
    pub fn value_port_count(&self, dir: portgraph::Direction) -> usize {
        self.dataflow_signature()
            .map(|sig| sig.port_count(dir))
            .unwrap_or(0)
    }

    /// The number of Value input ports.
    pub fn value_input_count(&self) -> usize {
        self.value_port_count(Direction::Incoming)
    }

    /// The number of Value output ports.
    pub fn value_output_count(&self) -> usize {
        self.value_port_count(Direction::Outgoing)
    }

    /// The non-dataflow input port for the operation, not described by the signature.
    /// See `[OpType::other_port]`.
    pub fn other_input_port(&self) -> Option<Port> {
        self.other_port(Direction::Incoming)
    }

    /// The non-dataflow input port for the operation, not described by the signature.
    /// See `[OpType::other_port]`.
    pub fn other_output_port(&self) -> Option<Port> {
        self.other_port(Direction::Outgoing)
    }

    /// If the op has a static input (Call and LoadConstant), the port of that input.
    pub fn static_input_port(&self) -> Option<IncomingPort> {
        match self {
            OpType::Call(call) => Some(call.called_function_port()),
            OpType::LoadConstant(l) => Some(l.constant_port()),
            _ => None,
        }
    }

    /// If the op has a static output (Const, FuncDefn, FuncDecl), the port of that output.
    pub fn static_output_port(&self) -> Option<OutgoingPort> {
        OpTag::StaticOutput
            .is_superset(self.tag())
            .then_some(0.into())
    }

    /// Returns the number of ports for the given direction.
    pub fn port_count(&self, dir: Direction) -> usize {
        let non_df_count = self.non_df_port_count(dir);
        // if there is a static input it comes before the non_df_ports
        let static_input =
            (dir == Direction::Incoming && OpTag::StaticInput.is_superset(self.tag())) as usize;
        self.value_port_count(dir) + non_df_count + static_input
    }

    /// Returns the number of inputs ports for the operation.
    pub fn input_count(&self) -> usize {
        self.port_count(Direction::Incoming)
    }

    /// Returns the number of outputs ports for the operation.
    pub fn output_count(&self) -> usize {
        self.port_count(Direction::Outgoing)
    }

    /// Checks whether the operation can contain children nodes.
    pub fn is_container(&self) -> bool {
        self.validity_flags().allowed_children != OpTag::None
    }
}

fn static_in_type(op: &OpType) -> Type {
    match op {
        OpType::Call(call) => Type::new_function(call.called_function_type().clone()),
        OpType::LoadConstant(load) => load.constant_type().clone(),
        _ => panic!("this function should not be called if the optype is not known to be Call or LoadConst.")
    }
}

/// Macro used by operations that want their
/// name to be the same as their type name
macro_rules! impl_op_name {
    ($i: ident) => {
        impl $crate::ops::OpName for $i {
            fn name(&self) -> smol_str::SmolStr {
                stringify!($i).into()
            }
        }
    };
}

use impl_op_name;

#[enum_dispatch]
/// Trait for setting name of OpType variants.
// Separate to OpTrait to allow simple definition via impl_op_name
pub trait OpName {
    /// The name of the operation.
    fn name(&self) -> SmolStr;
}

/// Trait statically querying the tag of an operation.
///
/// This is implemented by all OpType variants, and always contains the dynamic
/// tag returned by `OpType::tag(&self)`.
pub trait StaticTag {
    /// The name of the operation.
    const TAG: OpTag;
}

#[enum_dispatch]
/// Trait implemented by all OpType variants.
pub trait OpTrait {
    /// A human-readable description of the operation.
    fn description(&self) -> &str;

    /// Tag identifying the operation.
    fn tag(&self) -> OpTag;

    /// The signature of the operation.
    ///
    /// Only dataflow operations have a signature, otherwise returns None.
    fn dataflow_signature(&self) -> Option<FunctionType> {
        None
    }

    /// The delta between the input extensions specified for a node,
    /// and the output extensions calculated for that node
    fn extension_delta(&self) -> ExtensionSet {
        ExtensionSet::new()
    }

    /// The edge kind for the non-dataflow or constant inputs of the operation,
    /// not described by the signature.
    ///
    /// If not None, a single extra output multiport of that kind will be
    /// present.
    fn other_input(&self) -> Option<EdgeKind> {
        None
    }

    /// The edge kind for the non-dataflow outputs of the operation, not
    /// described by the signature.
    ///
    /// If not None, a single extra output multiport of that kind will be
    /// present.
    fn other_output(&self) -> Option<EdgeKind> {
        None
    }

    /// Get the number of non-dataflow multiports.
    fn non_df_port_count(&self, dir: Direction) -> usize {
        match dir {
            Direction::Incoming => self.other_input(),
            Direction::Outgoing => self.other_output(),
        }
        .is_some() as usize
    }
}

/// Properties of child graphs of ops, if the op has children.
#[enum_dispatch]
pub trait OpParent {
    /// The inner function type of the operation, if it has a child dataflow
    /// sibling graph.
    fn inner_function_type(&self) -> Option<FunctionType> {
        None
    }
}

impl<T: DataflowParent> OpParent for T {
    fn inner_function_type(&self) -> Option<FunctionType> {
        Some(DataflowParent::inner_signature(self))
    }
}

impl OpParent for Module {}
impl OpParent for AliasDecl {}
impl OpParent for AliasDefn {}
impl OpParent for Const {}
impl OpParent for Input {}
impl OpParent for Output {}
impl OpParent for Call {}
impl OpParent for CallIndirect {}
impl OpParent for LoadConstant {}
impl OpParent for LeafOp {}
impl OpParent for TailLoop {}
impl OpParent for CFG {}
impl OpParent for Conditional {}
impl OpParent for FuncDecl {}
impl OpParent for Exit {}

#[enum_dispatch]
/// Methods for Ops to validate themselves and children
pub trait ValidateOp {
    /// Returns a set of flags describing the validity predicates for this operation.
    #[inline]
    fn validity_flags(&self) -> validate::OpValidityFlags {
        Default::default()
    }

    /// Validate the ordered list of children.
    #[inline]
    fn validate_op_children<'a>(
        &self,
        _children: impl DoubleEndedIterator<Item = (NodeIndex, &'a OpType)>,
    ) -> Result<(), validate::ChildrenValidationError> {
        Ok(())
    }
}

/// Macro used for default implementation of ValidateOp
macro_rules! impl_validate_op {
    ($i: ident) => {
        impl $crate::ops::ValidateOp for $i {}
    };
}

use impl_validate_op;
