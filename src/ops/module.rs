use std::any::Any;

use super::Op;
use crate::{
    macros::impl_box_clone,
    types::{ClassicType, Signature, SignatureDescription, SimpleType},
};

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum ModuleOp {
    #[default]
    /// The root of a module, parent of all other `ModuleOp`s
    Root,
    /// A function definition.
    /// Children nodes are the body of the definition
    Def {
        signature: Signature,
    },
    /// External function declaration, linked at runtime
    Declare {
        signature: Signature,
    },
    /// Top level struct type definition
    #[non_exhaustive] // TODO
    Struct {},
    /// A type alias
    #[non_exhaustive] // TODO
    Alias {},
    // A constant value definition
    Const(ConstValue),
}

impl Op for ModuleOp {
    fn name(&self) -> SmolStr {
        // TODO: These should be unique names for each distinct op
        match self {
            ModuleOp::Root => "module",
            ModuleOp::Def { .. } => "def",
            ModuleOp::Declare { .. } => "declare",
            ModuleOp::Struct { .. } => "struct",
            ModuleOp::Alias { .. } => "alias",
            ModuleOp::Const(_) => "const",
        }
        .into()
    }

    fn signature(&self) -> Signature {
        match self {
            ModuleOp::Root => Signature::default(),
            ModuleOp::Def { signature } => signature.clone(),
            ModuleOp::Declare { signature } => signature.clone(),
            ModuleOp::Struct { .. } => todo!(),
            ModuleOp::Alias { .. } => todo!(),
            ModuleOp::Const(v) => Signature::new_const(v.const_type()),
        }
    }
}

/// Value constants
///
/// TODO: Add more constants
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ConstValue {
    Bit(bool),
    Int(i64),
    Opaque(SimpleType, Box<dyn CustomConst>),
}

impl PartialEq for ConstValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bit(l0), Self::Bit(r0)) => l0 == r0,
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Opaque(l0, l1), Self::Opaque(r0, r1)) => l0 == r0 && l1.eq(&**r1),
            _ => false,
        }
    }
}

impl Default for ConstValue {
    fn default() -> Self {
        Self::Bit(false)
    }
}

impl ConstValue {
    /// Returns the datatype of the constant
    pub fn const_type(&self) -> ClassicType {
        match self {
            Self::Bit(_) => ClassicType::Bit,
            Self::Int(_) => ClassicType::Int,
            Self::Opaque(_, b) => (*b).const_type(),
        }
    }
}

impl Op for ConstValue {
    fn name(&self) -> SmolStr {
        match self {
            Self::Bit(v) => format!("const:bit:{v}"),
            Self::Int(v) => format!("const:int:{v}"),
            Self::Opaque(_, v) => format!("const:{}", v.name()),
        }
        .into()
    }

    fn description(&self) -> &str {
        "Constant value"
    }

    fn signature(&self) -> Signature {
        Signature::new_const(self.const_type())
    }

    fn signature_desc(&self) -> Option<SignatureDescription> {
        Some(SignatureDescription::new([], [], None))
    }
}

impl<T: CustomConst> From<T> for ConstValue {
    fn from(v: T) -> Self {
        Self::Opaque(SimpleType::Classic(v.const_type()), Box::new(v))
    }
}

/// Constant value for opaque [`SimpleType`]s.
///
// When implementing this trait, include the `#[typetag::serde]` attribute to
/// enable serialization.
#[typetag::serde]
pub trait CustomConst:
    Send + Sync + std::fmt::Debug + CustomConstBoxClone + Any + Downcast
{
    fn name(&self) -> SmolStr;

    /// Returns the type of the constant.
    fn const_type(&self) -> ClassicType;

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    fn eq(&self, other: &dyn CustomConst) -> bool {
        let _ = other;
        false
    }
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);
