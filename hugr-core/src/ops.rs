//! The operation types for the HUGR.

pub mod constant;
pub mod controlflow;
pub mod custom;
pub mod dataflow;
pub mod handle;
pub mod module;
pub mod sum;
pub mod tag;
pub mod validate;
use crate::extension::simple_op::MakeExtensionOp;
use crate::extension::ExtensionSet;
use crate::types::{EdgeKind, Signature};
use crate::{Direction, OutgoingPort, Port};
use crate::{IncomingPort, PortIndex};
use derive_more::Display;
use paste::paste;
use portgraph::NodeIndex;

use enum_dispatch::enum_dispatch;

pub use constant::{Const, Value};
pub use controlflow::{BasicBlock, Case, Conditional, DataflowBlock, ExitBlock, TailLoop, CFG};
pub use custom::{ExtensionOp, OpaqueOp};
pub use dataflow::{
    Call, CallIndirect, DataflowOpTrait, DataflowParent, Input, LoadConstant, LoadFunction, Output,
    DFG,
};
pub use module::{AliasDecl, AliasDefn, FuncDecl, FuncDefn, Module};
use smol_str::SmolStr;
pub use sum::Tag;
pub use tag::OpTag;

#[enum_dispatch(OpTrait, NamedOp, ValidateOp, OpParent)]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
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
    LoadFunction,
    DFG,
    #[serde(skip_deserializing, rename = "Extension")]
    ExtensionOp,
    #[serde(rename = "Extension")]
    OpaqueOp,
    Tag,
    DataflowBlock,
    ExitBlock,
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
                    TryInto::<&$Op>::try_into(self).ok()
                }

                #[doc = "Returns `true` if the operation is an instance of `" $Op "`."]
                pub fn [<is_ $sname:snake>](&self) -> bool {
                    self.[<as_ $sname:snake>]().is_some()
                }
            }

            impl<'a> TryFrom<&'a OpType> for &'a $Op {
                type Error = ();
                fn try_from(optype: &'a OpType) -> Result<Self, Self::Error> {
                    if let OpType::$Op(l) = optype {
                        Ok(l)
                    } else {
                        Err(())
                    }
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
impl_op_ref_try_into!(LoadFunction);
impl_op_ref_try_into!(DFG, dfg);
impl_op_ref_try_into!(ExtensionOp);
impl_op_ref_try_into!(Tag);
impl_op_ref_try_into!(DataflowBlock);
impl_op_ref_try_into!(ExitBlock);
impl_op_ref_try_into!(TailLoop);
impl_op_ref_try_into!(CFG, cfg);
impl_op_ref_try_into!(Conditional);
impl_op_ref_try_into!(Case);

/// The default OpType (as returned by [Default::default])
pub const DEFAULT_OPTYPE: OpType = OpType::Module(Module::new());

impl Default for OpType {
    fn default() -> Self {
        DEFAULT_OPTYPE
    }
}

impl Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl OpType {
    /// The edge kind for the non-dataflow ports of the operation, not described
    /// by the signature.
    ///
    /// If not None, a single extra port of that kind will be present on
    /// the given direction after any dataflow or constant ports.
    #[inline]
    pub fn other_port_kind(&self, dir: Direction) -> Option<EdgeKind> {
        match dir {
            Direction::Incoming => self.other_input(),
            Direction::Outgoing => self.other_output(),
        }
    }

    /// The edge kind for the static ports of the operation, not described by
    /// the dataflow signature.
    ///
    /// If not None, an extra input port of that kind will be present on the
    /// given direction after any dataflow ports and before any
    /// [`OpType::other_port_kind`] ports.
    #[inline]
    pub fn static_port_kind(&self, dir: Direction) -> Option<EdgeKind> {
        match dir {
            Direction::Incoming => self.static_input(),
            Direction::Outgoing => self.static_output(),
        }
    }

    /// Returns the edge kind for the given port.
    ///
    /// The result may be a value port, a static port, or a non-dataflow port.
    /// See [`OpType::dataflow_signature`], [`OpType::static_port_kind`], and
    /// [`OpType::other_port_kind`].
    pub fn port_kind(&self, port: impl Into<Port>) -> Option<EdgeKind> {
        let signature = self.dataflow_signature().unwrap_or_default();
        let port: Port = port.into();
        let dir = port.direction();
        let port_count = signature.port_count(dir);

        // Dataflow ports
        if port.index() < port_count {
            return signature.port_type(port).cloned().map(EdgeKind::Value);
        }

        // Constant port
        let static_kind = self.static_port_kind(dir);
        if port.index() == port_count {
            if let Some(kind) = static_kind {
                return Some(kind);
            }
        }

        // Non-dataflow ports
        self.other_port_kind(dir)
    }

    /// The non-dataflow port for the operation, not described by the signature.
    /// See `[OpType::other_port_kind]`.
    ///
    /// Returns None if there is no such port, or if the operation defines multiple non-dataflow ports.
    pub fn other_port(&self, dir: Direction) -> Option<Port> {
        let df_count = self.value_port_count(dir);
        let non_df_count = self.non_df_port_count(dir);
        // if there is a static input it comes before the non_df_ports
        let static_input =
            (dir == Direction::Incoming && OpTag::StaticInput.is_superset(self.tag())) as usize;
        if self.other_port_kind(dir).is_some() && non_df_count >= 1 {
            Some(Port::new(dir, df_count + static_input))
        } else {
            None
        }
    }

    /// The non-dataflow input port for the operation, not described by the signature.
    /// See `[OpType::other_port]`.
    #[inline]
    pub fn other_input_port(&self) -> Option<IncomingPort> {
        self.other_port(Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// The non-dataflow output port for the operation, not described by the signature.
    /// See `[OpType::other_port]`.
    #[inline]
    pub fn other_output_port(&self) -> Option<OutgoingPort> {
        self.other_port(Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// If the op has a static port, the port of that input.
    ///
    /// See [`OpType::static_input_port`] and [`OpType::static_output_port`].
    #[inline]
    pub fn static_port(&self, dir: Direction) -> Option<Port> {
        self.static_port_kind(dir)?;
        Some(Port::new(dir, self.value_port_count(dir)))
    }

    /// If the op has a static input ([`Call`], [`LoadConstant`], and [`LoadFunction`]), the port of
    /// that input.
    #[inline]
    pub fn static_input_port(&self) -> Option<IncomingPort> {
        self.static_port(Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// If the op has a static output ([`Const`], [`FuncDefn`], [`FuncDecl`]), the port of that output.
    #[inline]
    pub fn static_output_port(&self) -> Option<OutgoingPort> {
        self.static_port(Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// The number of Value ports in given direction.
    #[inline]
    pub fn value_port_count(&self, dir: portgraph::Direction) -> usize {
        self.dataflow_signature()
            .map(|sig| sig.port_count(dir))
            .unwrap_or(0)
    }

    /// The number of Value input ports.
    #[inline]
    pub fn value_input_count(&self) -> usize {
        self.value_port_count(Direction::Incoming)
    }

    /// The number of Value output ports.
    #[inline]
    pub fn value_output_count(&self) -> usize {
        self.value_port_count(Direction::Outgoing)
    }

    /// Returns the number of ports for the given direction.
    #[inline]
    pub fn port_count(&self, dir: Direction) -> usize {
        let has_static_port = self.static_port_kind(dir).is_some();
        let non_df_count = self.non_df_port_count(dir);
        self.value_port_count(dir) + has_static_port as usize + non_df_count
    }

    /// Returns the number of inputs ports for the operation.
    #[inline]
    pub fn input_count(&self) -> usize {
        self.port_count(Direction::Incoming)
    }

    /// Returns the number of outputs ports for the operation.
    #[inline]
    pub fn output_count(&self) -> usize {
        self.port_count(Direction::Outgoing)
    }

    /// Checks whether the operation can contain children nodes.
    #[inline]
    pub fn is_container(&self) -> bool {
        self.validity_flags().allowed_children != OpTag::None
    }

    /// Cast to an extension operation.
    pub fn cast<T: MakeExtensionOp>(&self) -> Option<T> {
        self.as_extension_op()
            .and_then(|o| T::from_extension_op(o).ok())
    }
}

/// Macro used by operations that want their
/// name to be the same as their type name
macro_rules! impl_op_name {
    ($i: ident) => {
        impl $crate::ops::NamedOp for $i {
            fn name(&self) -> $crate::ops::OpName {
                stringify!($i).into()
            }
        }
    };
}

use impl_op_name;

/// A unique identifier for a operation.
pub type OpName = SmolStr;

/// Slice of a [`OpName`] operation identifier.
pub type OpNameRef = str;

#[enum_dispatch]
/// Trait for setting name of OpType variants.
// Separate to OpTrait to allow simple definition via impl_op_name
pub trait NamedOp {
    /// The name of the operation.
    fn name(&self) -> OpName;
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
    fn dataflow_signature(&self) -> Option<Signature> {
        None
    }

    /// The delta between the input extensions specified for a node,
    /// and the output extensions calculated for that node
    fn extension_delta(&self) -> ExtensionSet {
        ExtensionSet::new()
    }

    /// The edge kind for the non-dataflow inputs of the operation,
    /// not described by the signature.
    ///
    /// If not None, a single extra input port of that kind will be
    /// present.
    fn other_input(&self) -> Option<EdgeKind> {
        None
    }

    /// The edge kind for the non-dataflow outputs of the operation, not
    /// described by the signature.
    ///
    /// If not None, a single extra output port of that kind will be
    /// present.
    fn other_output(&self) -> Option<EdgeKind> {
        None
    }

    /// The edge kind for a single constant input of the operation, not
    /// described by the dataflow signature.
    ///
    /// If not None, an extra input port of that kind will be present after the
    /// dataflow input ports and before any [`OpTrait::other_input`] ports.
    fn static_input(&self) -> Option<EdgeKind> {
        None
    }

    /// The edge kind for a single constant output of the operation, not
    /// described by the dataflow signature.
    ///
    /// If not None, an extra output port of that kind will be present after the
    /// dataflow input ports and before any [`OpTrait::other_output`] ports.
    fn static_output(&self) -> Option<EdgeKind> {
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
    ///
    /// Non-container ops like `FuncDecl` return `None` even though they represent a function.
    fn inner_function_type(&self) -> Option<Signature> {
        None
    }
}

impl<T: DataflowParent> OpParent for T {
    fn inner_function_type(&self) -> Option<Signature> {
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
impl OpParent for LoadFunction {}
impl OpParent for ExtensionOp {}
impl OpParent for OpaqueOp {}
impl OpParent for Tag {}
impl OpParent for CFG {}
impl OpParent for Conditional {}
impl OpParent for FuncDecl {}
impl OpParent for ExitBlock {}

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
