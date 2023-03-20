//! Definition of the leaf operations.
//!
//! TODO: Better name than "leaf"?

use downcast_rs::{impl_downcast, Downcast};
use std::any::Any;

use super::Op;
use crate::hugr::Hugr;
use crate::macros::impl_box_clone;
use crate::types::{DataType, Signature};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LeafOp {
    /// A user-defined operation that can be downcasted by the extensions that define it.
    CustomOp(Box<dyn CustomOp>),
    /// An externally-defined operation that can be downcasted by the extensions that define it.
    Opdef(Box<dyn CustomOp>),

    H,
    T,
    S,
    X,
    Y,
    Z,
    Tadj,
    Sadj,
    CX,
    ZZMax,
    Reset,
    Noop(DataType),
    Measure,
    AngleAdd,
    AngleMul,
    AngleNeg,
    QuatMul,
    Copy {
        n_copies: u32,
        typ: DataType,
    },
    RxF64,
    RzF64,
    TK1,
    Rotation,
    ToRotation,
    Xor,
}

impl PartialEq for LeafOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::CustomOp(l0), Self::CustomOp(r0)) => l0.eq(&**r0),
            (Self::Opdef(l0), Self::Opdef(r0)) => l0.eq(&**r0),
            (Self::Noop(l0), Self::Noop(r0)) => l0 == r0,
            (
                Self::Copy {
                    n_copies: l_n_copies,
                    typ: l_typ,
                },
                Self::Copy {
                    n_copies: r_n_copies,
                    typ: r_typ,
                },
            ) => l_n_copies == r_n_copies && l_typ == r_typ,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Default for LeafOp {
    fn default() -> Self {
        Self::Noop(DataType::default())
    }
}

pub fn approx_eq(x: f64, y: f64, modulo: u32, tol: f64) -> bool {
    let modulo = f64::from(modulo);
    let x = (x - y) / modulo;

    let x = x - x.floor();

    let r = modulo * x;

    r < tol || r > modulo - tol
}

impl LeafOp {
    pub fn is_one_qb_gate(&self) -> bool {
        self.signature().linear().count() == 1
    }

    pub fn is_two_qb_gate(&self) -> bool {
        self.signature().linear().count() == 1
    }

    pub fn is_pure_classical(&self) -> bool {
        self.signature().purely_classical()
    }
}

impl Op for LeafOp {
    fn signature(&self) -> Signature {
        match self {
            LeafOp::Noop(typ) => Signature::new_df([typ.clone()], [typ.clone()]),
            LeafOp::H
            | LeafOp::Reset
            | LeafOp::T
            | LeafOp::S
            | LeafOp::Tadj
            | LeafOp::Sadj
            | LeafOp::X
            | LeafOp::Y
            | LeafOp::Z => Signature::new_linear([DataType::Qubit]),
            LeafOp::CX | LeafOp::ZZMax => Signature::new_linear([DataType::Qubit, DataType::Qubit]),
            LeafOp::Measure => Signature::new_linear([DataType::Qubit, DataType::Bool]),
            LeafOp::AngleAdd | LeafOp::AngleMul => Signature::new_linear([DataType::Angle]),
            LeafOp::QuatMul => Signature::new_linear([DataType::Quat64]),
            LeafOp::AngleNeg => Signature::new_linear([DataType::Angle]),
            LeafOp::Copy { n_copies, typ } => {
                Signature::new_df([typ.clone()], vec![typ.clone(); *n_copies as usize])
            }
            LeafOp::RxF64 | LeafOp::RzF64 => {
                Signature::new_df([DataType::Qubit], [DataType::Angle])
            }
            LeafOp::TK1 => Signature::new_df(vec![DataType::Qubit], vec![DataType::Angle; 3]),
            LeafOp::Rotation => Signature::new_df([DataType::Qubit], [DataType::Quat64]),
            LeafOp::ToRotation => Signature::new_df(
                [DataType::Angle, DataType::F64, DataType::F64, DataType::F64],
                [DataType::Quat64],
            ),
            LeafOp::Xor => Signature::new_df([DataType::Bool, DataType::Bool], [DataType::Bool]),
            LeafOp::CustomOp(op) => op.signature(),
            LeafOp::Opdef(op) => op.signature(),
        }
    }

    fn name(&self) -> &str {
        match self {
            LeafOp::CustomOp(op) => op.name(),
            LeafOp::Opdef(op) => op.name(),
            LeafOp::H => "H",
            LeafOp::T => "T",
            LeafOp::S => "S",
            LeafOp::X => "X",
            LeafOp::Y => "Y",
            LeafOp::Z => "Z",
            LeafOp::Tadj => "Tadj",
            LeafOp::Sadj => "Sadj",
            LeafOp::CX => "CX",
            LeafOp::ZZMax => "ZZMax",
            LeafOp::Reset => "Reset",
            LeafOp::Noop(_) => "Noop",
            LeafOp::Measure => "Measure",
            LeafOp::AngleAdd => "AngleAdd",
            LeafOp::AngleMul => "AngleMul",
            LeafOp::AngleNeg => "AngleNeg",
            LeafOp::QuatMul => "QuatMul",
            LeafOp::Copy { .. } => "Copy",
            LeafOp::RxF64 => "RxF64",
            LeafOp::RzF64 => "RzF64",
            LeafOp::TK1 => "TK1",
            LeafOp::Rotation => "Rotation",
            LeafOp::ToRotation => "ToRotation",
            LeafOp::Xor => "Xor",
        }
    }
}

#[derive(Debug)]
pub struct ToHugrFail;
pub trait CustomOp: Send + Sync + std::fmt::Debug + CustomOpBoxClone + Op + Any + Downcast {
    // TODO: Create a separate HUGR, or create a children subgraph in the HUGR?
    fn to_hugr(&self) -> Result<Hugr, ToHugrFail> {
        Err(ToHugrFail)
    }

    /// Check if two custom ops are equal, by downcasting and comparing the definitions.
    fn eq(&self, other: &dyn CustomOp) -> bool {
        let _ = other;
        false
    }
}

impl_downcast!(CustomOp);
impl_box_clone!(CustomOp, CustomOpBoxClone);
