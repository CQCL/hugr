//! Definition of the leaf operations.
//!
//! TODO: Better name than "leaf"?

use super::{Op, OpaqueOp};
use crate::types::{ClassicType, QuantumType, Signature, SimpleType};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum LeafOp {
    /// A user-defined operation that can be downcasted by the extensions that
    /// define it.
    CustomOp(OpaqueOp),
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
    Noop(SimpleType),
    Measure,
    Copy {
        n_copies: u32,
        typ: ClassicType,
    },
    Xor,
}

impl Default for LeafOp {
    fn default() -> Self {
        Self::Noop(SimpleType::default())
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
        // TODO: Missing [`DataType::Money`] inputs and outputs.

        // Static signatures. The `TypeRow`s in the `Signature` use a
        // copy-on-write strategy, so we can avoid unnecessary allocations.
        const Q: SimpleType = SimpleType::Quantum(QuantumType::Qubit);
        const B: SimpleType = SimpleType::Classic(ClassicType::Bit);
        static ROW_QUBIT: &[SimpleType] = &[Q];
        static ROW_2QUBIT: &[SimpleType] = &[Q, Q];
        static ROW_BIT: &[SimpleType] = &[B];
        static ROW_2BIT: &[SimpleType] = &[B, B];
        static ROW_QUBIT_BIT: &[SimpleType] = &[Q, B];

        match self {
            LeafOp::Noop(typ) => Signature::new_df(vec![typ.clone()], vec![typ.clone()]),
            LeafOp::H
            | LeafOp::Reset
            | LeafOp::T
            | LeafOp::S
            | LeafOp::Tadj
            | LeafOp::Sadj
            | LeafOp::X
            | LeafOp::Y
            | LeafOp::Z => Signature::new_linear(ROW_QUBIT),
            LeafOp::CX | LeafOp::ZZMax => Signature::new_linear(ROW_2QUBIT),
            LeafOp::Measure => Signature::new_linear(ROW_QUBIT_BIT),
            LeafOp::Copy { n_copies, typ } => {
                let typ: SimpleType = typ.clone().into();
                Signature::new_df(vec![typ.clone()], vec![typ; *n_copies as usize])
            }
            LeafOp::Xor => Signature::new_df(ROW_2BIT, ROW_BIT),
            LeafOp::CustomOp(opaque) => opaque.signature(),
        }
    }

    fn name(&self) -> &str {
        match self {
            LeafOp::CustomOp(opaque) => opaque.id.as_str(),
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
            LeafOp::Copy { .. } => "Copy",
            LeafOp::Xor => "Xor",
        }
    }
}
