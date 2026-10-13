//! The operation types for the HUGR.
//!
//! Operations represent the nodes in a HUGR graph. Each operation has a type
//! ([`OpType`]), which determines its behaviour, signature, and what kind of
//! edges can connect to it. We may have different operations:
//! - Unified representation: All operations are represented by the [`OpType`] enum
//! - Dataflow operations: Operations like [`DFG`], [`Call`], and [`Input`]/[`Output`]
//!   that process data
//! - Control flow operations: Operations like [`CFG`], [`Conditional`], and [`TailLoop`]
//!   for program control flow
//! - Module-level operations: Operations like [`FuncDefn`], [`FuncDecl`], and [`Module`]
//!   for program structure
//! - Extension operations: Custom operations via [`ExtensionOp`]
//!
//! # Example
//!
//! ```
//! use hugr::ops::{OpType, OpTrait, Input, Output, DFG, Call, FuncDefn, FuncDecl, LoadConstant};
//! use hugr::ops::{Module, CFG, TailLoop, Conditional};
//! use hugr::ops::dataflow::IOTrait;
//! use hugr::types::{Signature, PolyFuncType};
//! use hugr::extension::prelude::{bool_t, usize_t};
//! use hugr::{Direction};
//!
//! // Create a dataflow graph operation
//! let sig = Signature::new(vec![usize_t(), bool_t()], vec![bool_t()]);
//! let dfg_op: OpType = DFG { signature: sig.clone() }.into();
//!
//! // Query operation properties
//! assert_eq!(dfg_op.value_port_count(Direction::Incoming), 2);
//! assert_eq!(dfg_op.value_port_count(Direction::Outgoing), 1);
//! assert!(dfg_op.is_container()); // DFG can contain child nodes
//!
//! // Create input and output operations for the DFG's children
//! let input_op: OpType = Input::new(sig.input().clone()).into();
//! let output_op: OpType = Output::new(sig.output().clone()).into();
//!
//! // Check dataflow signatures
//! if let Some(out_sig) = output_op.dataflow_signature() {
//!     println!("Output takes {} inputs", out_sig.input_count());
//! }
//!
//! // Create a function call operation
//! let func_sig = PolyFuncType::from(Signature::new(vec![usize_t()], vec![bool_t()]));
//! let call_op: OpType = Call::try_new(func_sig.clone(), &[]).unwrap().into();
//!
//! // Access the call signature
//! let call_sig = call_op.dataflow_signature().unwrap();
//! assert_eq!(call_sig.input_count(), 1);
//! assert_eq!(call_sig.output_count(), 1);
//!
//! // Create a load constant operation
//! let load_op: OpType = LoadConstant { datatype: bool_t() }.into();
//! assert!(load_op.static_input_port().is_some()); // Has static input for constant
//!
//! // Create a control flow graph operation
//! let cfg_sig = Signature::new(vec![usize_t()], vec![bool_t()]);
//! let cfg_op: OpType = CFG { signature: cfg_sig }.into();
//! assert!(cfg_op.is_container()); // CFG contains basic blocks
//!
//! // Create a tail loop operation
//! let tail_loop = TailLoop {
//!     just_inputs: vec![usize_t()].into(),
//!     just_outputs: vec![bool_t()].into(),
//!     rest: vec![usize_t()].into(),
//! };
//! let loop_op: OpType = tail_loop.into();
//! assert!(loop_op.is_container()); // Contains loop body
//!
//! // Create a conditional operation
//! let conditional = Conditional {
//!     sum_rows: vec![[usize_t()].into(), [bool_t()].into()],
//!     other_inputs: [usize_t()].into(),
//!     outputs: [bool_t()].into(),
//! };
//! let cond_op: OpType = conditional.into();
//! assert!(cond_op.is_container()); // Contains case branches
//! let cond_sig = cond_op.dataflow_signature().unwrap();
//! assert_eq!(cond_sig.input_count(), 2); // Sum + other inputs
//!
//! // Create a module operation (root of HUGR)
//! let module_op: OpType = Module::new().into();
//! assert!(module_op.is_container()); // Contains functions and declarations
//!
//! // Create a function definition
//! let func_defn: OpType = FuncDefn::new("my_func", func_sig.clone()).into();
//! assert!(func_defn.static_output_port().is_some()); // Has static output
//! assert!(func_defn.is_container()); // Can contain the function body
//!
//! // Create a function declaration (forward declaration)
//! let func_decl: OpType = FuncDecl::new("external_func", func_sig).into();
//! assert!(func_decl.static_output_port().is_some()); // Also has static output
//! assert!(!func_decl.is_container()); // Declarations have no body
//!
//! ```

pub mod constant;
pub mod controlflow;
pub mod custom;
pub mod dataflow;
pub mod handle;
pub mod module;
pub mod sum;
pub mod tag;
pub mod validate;
use crate::core::HugrNode;
use crate::extension::resolution::{
    ExtensionCollectionError, collect_op_extension, collect_op_types_extensions,
};
use std::borrow::Cow;
use std::cmp::Ordering;

use crate::extension::simple_op::MakeExtensionOp;
use crate::extension::{ExtensionId, ExtensionRegistry};
use crate::hugr::views::render::RenderStringConfig;
use crate::types::{EdgeKind, Signature, Substitution, Type};
use crate::{Direction, Node, OutgoingPort, Port};
use crate::{IncomingPort, PortIndex};
use handle::NodeHandle;
use pastey::paste;

use enum_dispatch::enum_dispatch;

pub use constant::{Const, Value};
pub use controlflow::{BasicBlock, CFG, Case, Conditional, DataflowBlock, ExitBlock, TailLoop};
pub use custom::{ExtensionOp, OpaqueOp};
pub use dataflow::{
    Call, CallIndirect, DFG, DataflowOpTrait, DataflowParent, Input, LoadConstant, LoadFunction,
    Output,
};
pub use module::{AliasDecl, AliasDefn, FuncDecl, FuncDefn, Module};
use smol_str::SmolStr;
pub use sum::Tag;
pub use tag::OpTag;

#[enum_dispatch(OpTrait, NamedOp, ValidateOp, OpParent)]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
/// The concrete operation types for a node in the HUGR.
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

fn optype_id(optype: &OpType) -> usize {
    match optype {
        OpType::Module(_) => 0,
        OpType::FuncDefn(_) => 1,
        OpType::FuncDecl(_) => 2,
        OpType::AliasDecl(_) => 3,
        OpType::AliasDefn(_) => 4,
        OpType::Const(_) => 5,
        OpType::Input(_) => 6,
        OpType::Output(_) => 7,
        OpType::Call(_) => 8,
        OpType::CallIndirect(_) => 9,
        OpType::LoadConstant(_) => 10,
        OpType::LoadFunction(_) => 11,
        OpType::DFG(_) => 12,
        OpType::ExtensionOp(_) => 13,
        OpType::OpaqueOp(_) => 14,
        OpType::Tag(_) => 15,
        OpType::DataflowBlock(_) => 16,
        OpType::ExitBlock(_) => 17,
        OpType::TailLoop(_) => 18,
        OpType::CFG(_) => 19,
        OpType::Conditional(_) => 20,
        OpType::Case(_) => 21,
    }
}

impl PartialOrd for OpType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let a = optype_id(self);
        let b = optype_id(other);
        if a < b {
            Some(Ordering::Less)
        } else if a > b {
            Some(Ordering::Greater)
        } else {
            match format!("{:?}", self).cmp(&format!("{:?}", other)) {
                Ordering::Less => Some(Ordering::Less),
                Ordering::Greater => Some(Ordering::Greater),
                Ordering::Equal => None,
            }
        }
    }
}

macro_rules! impl_op_ref_try_into {
    ($Op: tt, $sname:ident) => {
        paste! {
            impl OpType {
                #[doc = "If is an instance of `" $Op "` return a reference to it."]
                #[must_use] pub fn [<as_ $sname:snake>](&self) -> Option<&$Op> {
                    TryInto::<&$Op>::try_into(self).ok()
                }

                #[doc = "Returns `true` if the operation is an instance of `" $Op "`."]
                #[must_use] pub fn [<is_ $sname:snake>](&self) -> bool {
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

/// The default `OpType` (as returned by [`Default::default`])
pub const DEFAULT_OPTYPE: OpType = OpType::Module(Module::new());

impl Default for OpType {
    fn default() -> Self {
        DEFAULT_OPTYPE
    }
}

impl std::fmt::Display for OpType {
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
    #[must_use]
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
    #[must_use]
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
        let port: Port = port.into();
        let dir = port.direction();
        let port_count = self.value_port_count(dir);

        // Dataflow ports
        if port.index() < port_count {
            return OpTrait::value_port_type(self, port).map(EdgeKind::Value);
        }

        // Constant port
        let static_kind = self.static_port_kind(dir);
        if port.index() == port_count
            && let Some(kind) = static_kind
        {
            return Some(kind);
        }

        // Non-dataflow ports
        self.other_port_kind(dir)
    }

    /// The non-dataflow port for the operation, not described by the signature.
    /// See `[OpType::other_port_kind]`.
    ///
    /// Returns None if there is no such port, or if the operation defines multiple non-dataflow ports.
    #[must_use]
    pub fn other_port(&self, dir: Direction) -> Option<Port> {
        let df_count = self.value_port_count(dir);
        let non_df_count = self.non_df_port_count(dir);
        // if there is a static input it comes before the non_df_ports
        let static_input =
            usize::from(dir == Direction::Incoming && OpTag::StaticInput.is_superset(self.tag()));
        if self.other_port_kind(dir).is_some() && non_df_count >= 1 {
            Some(Port::new(dir, df_count + static_input))
        } else {
            None
        }
    }

    /// The non-dataflow input port for the operation, not described by the signature.
    /// See `[OpType::other_port]`.
    #[inline]
    #[must_use]
    pub fn other_input_port(&self) -> Option<IncomingPort> {
        self.other_port(Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// The non-dataflow output port for the operation, not described by the signature.
    /// See `[OpType::other_port]`.
    #[inline]
    #[must_use]
    pub fn other_output_port(&self) -> Option<OutgoingPort> {
        self.other_port(Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// If the op has a static port, the port of that input.
    ///
    /// See [`OpType::static_input_port`] and [`OpType::static_output_port`].
    #[inline]
    #[must_use]
    pub fn static_port(&self, dir: Direction) -> Option<Port> {
        self.static_port_kind(dir)?;
        Some(Port::new(dir, self.value_port_count(dir)))
    }

    /// If the op has a static input ([`Call`], [`LoadConstant`], and [`LoadFunction`]), the port of
    /// that input.
    #[inline]
    #[must_use]
    pub fn static_input_port(&self) -> Option<IncomingPort> {
        self.static_port(Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// If the op has a static output ([`Const`], [`FuncDefn`], [`FuncDecl`]), the port of that output.
    #[inline]
    #[must_use]
    pub fn static_output_port(&self) -> Option<OutgoingPort> {
        self.static_port(Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// Return the dataflow value ports for the given direction.
    ///
    /// See [`OpType::value_input_ports`] and [`OpType::value_output_ports`].
    #[inline]
    #[must_use]
    pub fn value_ports(&self, dir: Direction) -> impl DoubleEndedIterator<Item = Port> {
        (0..self.value_port_count(dir)).map(move |i| Port::new(dir, i))
    }

    /// Return the dataflow value ports and their types for the given direction.
    #[inline]
    pub fn value_types(&self, dir: Direction) -> impl Iterator<Item = (Port, Type)> {
        self.value_ports(dir)
            .map(|port| (port, self.value_port_type(port).unwrap()))
    }

    /// Return the dataflow value input ports and their types.
    #[inline]
    pub fn value_input_types(&self) -> impl Iterator<Item = (IncomingPort, Type)> {
        self.value_types(Direction::Incoming)
            .map(|(port, typ)| (port.as_incoming().unwrap(), typ))
    }

    /// Return the dataflow value output ports and their types.
    #[inline]
    pub fn value_output_types(&self) -> impl Iterator<Item = (OutgoingPort, Type)> {
        self.value_types(Direction::Outgoing)
            .map(|(port, typ)| (port.as_outgoing().unwrap(), typ))
    }

    /// Return the dataflow value input ports for the given direction.
    #[inline]
    #[must_use]
    pub fn value_input_ports(&self) -> impl DoubleEndedIterator<Item = IncomingPort> {
        self.value_ports(Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// Return the dataflow value output ports for the given direction.
    #[inline]
    #[must_use]
    pub fn value_output_ports(&self) -> impl DoubleEndedIterator<Item = OutgoingPort> {
        self.value_ports(Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// The number of Value ports in given direction.
    #[inline]
    #[must_use]
    pub fn value_port_count(&self, dir: portgraph::Direction) -> usize {
        OpTrait::value_port_count(self, dir)
    }

    /// The number of Value input ports.
    #[inline]
    #[must_use]
    pub fn value_input_count(&self) -> usize {
        self.value_port_count(Direction::Incoming)
    }

    /// The number of Value output ports.
    #[inline]
    #[must_use]
    pub fn value_output_count(&self) -> usize {
        self.value_port_count(Direction::Outgoing)
    }

    /// Returns the number of ports for the given direction.
    #[inline]
    #[must_use]
    pub fn port_count(&self, dir: Direction) -> usize {
        let has_static_port = self.static_port_kind(dir).is_some();
        let non_df_count = self.non_df_port_count(dir);
        self.value_port_count(dir) + usize::from(has_static_port) + non_df_count
    }

    /// Returns the number of inputs ports for the operation.
    #[inline]
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.port_count(Direction::Incoming)
    }

    /// Returns the number of outputs ports for the operation.
    #[inline]
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.port_count(Direction::Outgoing)
    }

    /// Checks whether the operation can contain children nodes.
    #[inline]
    #[must_use]
    pub fn is_container(&self) -> bool {
        self.validity_flags::<Node>().allowed_children != OpTag::None
    }

    /// Cast to an extension operation.
    ///
    /// Returns `None` if the operation is not of the requested type.
    pub fn cast<T: MakeExtensionOp>(&self) -> Option<T> {
        self.as_extension_op().and_then(ExtensionOp::cast)
    }

    /// Returns the extension where the operation is defined, if any.
    #[must_use]
    pub fn extension_id(&self) -> Option<&ExtensionId> {
        match self {
            OpType::OpaqueOp(opaque) => Some(opaque.extension()),
            OpType::ExtensionOp(e) => Some(e.def().extension_id()),
            _ => None,
        }
    }

    /// Returns a registry with all the extensions required by the operation.
    ///
    /// This includes the operation extension in [`OpType::extension_id`], and any
    /// extension required by the operation's signature types.
    pub fn used_extensions(&self) -> Result<ExtensionRegistry, ExtensionCollectionError> {
        // Collect extensions on the types.
        let mut reg = collect_op_types_extensions(None, self)?;
        // And on the operation definition itself.
        if let Some(ext) = collect_op_extension(None, self)? {
            reg.register(ext);
        }
        reg.extend_with_dependencies()?;
        Ok(reg)
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
/// Trait for setting name of `OpType` variants.
// Separate to OpTrait to allow simple definition via impl_op_name
pub(crate) trait NamedOp {
    /// The name of the operation.
    fn name(&self) -> OpName;
}

/// Trait statically querying the tag of an operation.
///
/// This is implemented by all `OpType` variants, and always contains the dynamic
/// tag returned by `OpType::tag(&self)`.
pub trait StaticTag {
    /// The name of the operation.
    const TAG: OpTag;
}

#[enum_dispatch]
/// Trait implemented by all `OpType` variants.
pub trait OpTrait: Sized + Clone {
    /// A human-readable description of the operation.
    fn description(&self) -> &str;

    /// Returns an exhaustive string representation of the operation.
    fn render_str(&self, _config: RenderStringConfig) -> String {
        // TODO: Remove this default implementation with the next breaking release.
        unimplemented!(
            "This operation is relying on the default undefined render_str implementation, implement one."
        )
    }

    /// Tag identifying the operation.
    fn tag(&self) -> OpTag;

    /// Tries to create a specific [`NodeHandle`] for a node with this operation
    /// type.
    ///
    /// Fails if the operation's [`OpTrait::tag`] does not match the
    /// [`NodeHandle::TAG`] of the requested handle.
    fn try_node_handle<N, H>(&self, node: N) -> Option<H>
    where
        N: HugrNode,
        H: NodeHandle<N> + From<N>,
    {
        H::TAG.is_superset(self.tag()).then(|| node.into())
    }

    /// The signature of the operation.
    ///
    /// Only dataflow operations have a signature, otherwise returns None.
    fn dataflow_signature(&self) -> Option<Cow<'_, Signature>> {
        None
    }

    /// Returns the type of a value port.
    ///
    /// Implementations may override this to avoid constructing a complete
    /// [`Signature`] when only one port type is needed.
    fn value_port_type(&self, port: Port) -> Option<Type> {
        self.dataflow_signature()?.port_type(port).cloned()
    }

    /// Returns the type of an input value port.
    ///
    /// Shorthand for `value_port_type(port.into())`.
    fn value_input_type(&self, port: IncomingPort) -> Option<Type> {
        self.value_port_type(port.into())
    }

    /// Returns the type of an output value port.
    ///
    /// Shorthand for `value_port_type(port.into())`.
    fn value_output_type(&self, port: OutgoingPort) -> Option<Type> {
        self.value_port_type(port.into())
    }

    /// Returns the number of value ports in one direction.
    ///
    /// Implementations may override this to avoid constructing a complete
    /// [`Signature`] when only its size is needed.
    fn value_port_count(&self, dir: Direction) -> usize {
        self.dataflow_signature()
            .map_or(0, |signature| signature.port_count(dir))
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
        usize::from(
            match dir {
                Direction::Incoming => self.other_input(),
                Direction::Outgoing => self.other_output(),
            }
            .is_some(),
        )
    }

    /// Apply a type-level substitution to this `OpType`, i.e. replace
    /// [type variables](crate::types::TypeArg::new_var_use) with new types.
    fn substitute(&self, _subst: &Substitution) -> Self {
        self.clone()
    }
}

/// Properties of child graphs of ops, if the op has children.
#[enum_dispatch]
pub trait OpParent {
    /// The inner function type of the operation, if it has a child dataflow
    /// sibling graph.
    ///
    /// Non-container ops like `FuncDecl` return `None` even though they represent a function.
    fn inner_function_type(&self) -> Option<Cow<'_, Signature>> {
        None
    }
}

impl<T: DataflowParent> OpParent for T {
    fn inner_function_type(&self) -> Option<Cow<'_, Signature>> {
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
    fn validity_flags<N: HugrNode>(&self) -> validate::OpValidityFlags<N> {
        Default::default()
    }

    /// Validate the ordered list of children.
    #[inline]
    fn validate_op_children<'a, N: HugrNode>(
        &self,
        _children: impl DoubleEndedIterator<Item = (N, &'a OpType)>,
    ) -> Result<(), validate::ChildrenValidationError<N>> {
        Ok(())
    }
}

/// Macro used for default implementation of `ValidateOp`
macro_rules! impl_validate_op {
    ($i: ident) => {
        impl $crate::ops::ValidateOp for $i {}
    };
}

use impl_validate_op;

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{PolyFuncType, Type};
    use rstest::rstest;

    fn signature() -> Signature {
        Signature::new([Type::UNIT, Type::new_unit_sum(2)], [Type::new_unit_sum(3)])
    }

    fn input() -> OpType {
        Input {
            types: signature().input,
        }
        .into()
    }

    fn output() -> OpType {
        Output {
            types: signature().output,
        }
        .into()
    }

    fn call_indirect() -> OpType {
        CallIndirect {
            signature: signature(),
        }
        .into()
    }

    fn load_constant() -> OpType {
        LoadConstant {
            datatype: Type::new_unit_sum(2),
        }
        .into()
    }

    fn load_function() -> OpType {
        let instantiation = signature();
        LoadFunction {
            func_sig: PolyFuncType::new(Vec::new(), instantiation.clone()),
            type_args: Vec::new(),
            instantiation,
        }
        .into()
    }

    fn tag() -> OpType {
        Tag::new(1, vec![vec![Type::UNIT].into(), signature().input]).into()
    }

    fn tail_loop() -> OpType {
        TailLoop {
            just_inputs: vec![Type::UNIT].into(),
            just_outputs: vec![Type::new_unit_sum(2)].into(),
            rest: vec![Type::new_unit_sum(3)].into(),
        }
        .into()
    }

    fn conditional() -> OpType {
        Conditional {
            sum_rows: vec![vec![Type::UNIT].into(), vec![Type::new_unit_sum(2)].into()],
            other_inputs: signature().input,
            outputs: signature().output,
        }
        .into()
    }

    /// Borrow-first queries must remain equivalent to the public signature API.
    #[rstest]
    #[case::input(input())]
    #[case::output(output())]
    #[case::call_indirect(call_indirect())]
    #[case::load_constant(load_constant())]
    #[case::load_function(load_function())]
    #[case::tag(tag())]
    #[case::tail_loop(tail_loop())]
    #[case::conditional(conditional())]
    fn value_ports_match_signature(#[case] op: OpType) {
        let signature = op.dataflow_signature().expect("dataflow operation");

        for dir in [Direction::Incoming, Direction::Outgoing] {
            assert_eq!(op.value_port_count(dir), signature.port_count(dir));
            for index in 0..signature.port_count(dir) {
                let port = Port::new(dir, index);
                assert_eq!(
                    op.port_kind(port),
                    signature.port_type(port).cloned().map(EdgeKind::Value)
                );
            }
        }
    }
}
