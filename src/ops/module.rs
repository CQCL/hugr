use super::Op;
use crate::types::{ClassicType, Signature, SimpleType};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

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
    fn name(&self) -> &str {
        match self {
            ModuleOp::Root => "module",
            ModuleOp::Def { .. } => "def",
            ModuleOp::Declare { .. } => "declare",
            ModuleOp::Struct { .. } => "struct",
            ModuleOp::Alias { .. } => "alias",
            ModuleOp::Const(_) => "const",
        }
    }

    fn signature(&self) -> Signature {
        match self {
            ModuleOp::Root => Signature::default(),
            ModuleOp::Def { signature } => signature.clone(),
            ModuleOp::Declare { signature } => signature.clone(),
            ModuleOp::Struct { .. } => todo!(),
            ModuleOp::Alias { .. } => todo!(),
            ModuleOp::Const(v) => Signature::new_const([v.constant_type()]),
        }
    }
}

/// Value constants
#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ConstValue {
    Bit(bool),
    Int(i64),
}

impl Default for ConstValue {
    fn default() -> Self {
        Self::Bit(false)
    }
}

impl ConstValue {
    /// Returns the datatype of the constant
    pub fn constant_type(&self) -> SimpleType {
        match self {
            Self::Bit(_) => ClassicType::Bit.into(),
            Self::Int(_) => ClassicType::Int.into(),
        }
    }
}
