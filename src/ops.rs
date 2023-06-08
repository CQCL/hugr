//! The operation types for the HUGR.
#![allow(missing_docs)]
pub mod constant;
pub mod controlflow;
pub mod custom;
pub mod dataflow;
pub mod handle;
pub mod leaf;
pub mod module;
pub mod tag;
pub mod validate;
use crate::types::{EdgeKind, Signature, SignatureDescription};
use crate::{Direction, Port};

pub use custom::{CustomOp, OpDef, OpaqueOp};

use portgraph::NodeIndex;
use smol_str::SmolStr;

use self::tag::OpTag;
use enum_dispatch::enum_dispatch;

pub use constant::{Const, ConstValue};
pub use controlflow::{BasicBlock, Case, Conditional, TailLoop, CFG};
pub use dataflow::{Call, CallIndirect, Input, LoadConstant, Output, DFG};
pub use leaf::LeafOp;
pub use module::{AliasDeclare, AliasDef, Declare, Def, Module};

#[enum_dispatch(OpTrait, OpName, ValidateOp)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// The concrete operation types for a node in the HUGR.
// TODO: Link the NodeHandles to the OpType.
#[non_exhaustive]
pub enum OpType {
    /// The root of a module, parent of all other `OpType`s.
    Module,
    Def,
    Declare,
    AliasDeclare,
    AliasDef,
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
pub trait OpName {
    fn name(&self) -> SmolStr;
}

#[enum_dispatch]
pub trait OpTrait {
    fn description(&self) -> &str;
    fn tag(&self) -> OpTag;
    fn signature(&self) -> Signature {
        Default::default()
    }
    fn signature_desc(&self) -> SignatureDescription {
        Default::default()
    }
    fn other_inputs(&self) -> Option<EdgeKind> {
        None
    }
    fn other_outputs(&self) -> Option<EdgeKind> {
        None
    }

    fn port_kind(&self, port: impl Into<Port>) -> Option<EdgeKind> {
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

#[enum_dispatch]
pub trait ValidateOp {
    fn validity_flags(&self) -> validate::OpValidityFlags {
        Default::default()
    }

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
