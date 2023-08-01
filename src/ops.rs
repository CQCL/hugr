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
use crate::types::{AbstractSignature, EdgeKind, SignatureDescription};
use crate::{Direction, Port};

use portgraph::NodeIndex;
use smol_str::SmolStr;

use enum_dispatch::enum_dispatch;

pub use constant::{Const, ConstValue};
pub use controlflow::{BasicBlock, Case, Conditional, TailLoop, CFG};
pub use dataflow::{Call, CallIndirect, Input, LoadConstant, Output, DFG};
pub use leaf::LeafOp;
pub use module::{AliasDecl, AliasDefn, FuncDecl, FuncDefn, Module};
pub use tag::OpTag;

#[enum_dispatch(OpTrait, OpName, ValidateOp)]
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
    BasicBlock,
    TailLoop,
    CFG,
    Conditional,
    Case,
}

impl Default for OpType {
    fn default() -> Self {
        Module.into()
    }
}

impl OpType {
    /// The edge kind for the non-dataflow or constant-input ports of the
    /// operation, not described by the signature.
    ///
    /// If not None, a single extra multiport of that kind will be present on
    /// the given direction.
    pub fn other_port(&self, dir: Direction) -> Option<EdgeKind> {
        match dir {
            Direction::Incoming => self.other_input(),
            Direction::Outgoing => self.other_output(),
        }
    }

    /// Returns the edge kind for the given port.
    pub fn port_kind(&self, port: impl Into<Port>) -> Option<EdgeKind> {
        let signature = self.signature();
        let port = port.into();
        let dir = port.direction();
        match port.index() < signature.port_count(dir) {
            true => signature.get(port),
            false => self.other_port(dir),
        }
    }

    /// The non-dataflow port for the operation, not described by the signature.
    /// See `[OpType::other_port]`.
    ///
    /// Returns None if there is no such port, or if the operation defines multiple non-dataflow ports.
    pub fn other_port_index(&self, dir: Direction) -> Option<Port> {
        let non_df_count = self.validity_flags().non_df_port_count(dir).unwrap_or(1);
        if self.other_port(dir).is_some() && non_df_count == 1 {
            Some(Port::new(dir, self.signature().port_count(dir)))
        } else {
            None
        }
    }

    /// Returns the number of ports for the given direction.
    pub fn port_count(&self, dir: Direction) -> usize {
        let signature = self.signature();
        let has_other_ports = self.other_port(dir).is_some();
        let non_df_count = self
            .validity_flags()
            .non_df_port_count(dir)
            .unwrap_or(has_other_ports as usize);
        signature.port_count(dir) + non_df_count
    }

    /// Returns the number of inputs ports for the operation.
    pub fn input_count(&self) -> usize {
        self.port_count(Direction::Incoming)
    }

    /// Returns the number of outputs ports for the operation.
    pub fn output_count(&self) -> usize {
        self.port_count(Direction::Outgoing)
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
    /// Only dataflow operations have a non-empty signature.
    fn signature(&self) -> AbstractSignature {
        Default::default()
    }
    /// Optional description of the ports in the signature.
    ///
    /// Only dataflow operations have a non-empty signature.
    fn signature_desc(&self) -> SignatureDescription {
        Default::default()
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
}

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
    fn validate_children<'a>(
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
