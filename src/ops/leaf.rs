//! Definition of the leaf operations.
//!
//! TODO: Better name than "leaf"?

use smol_str::SmolStr;

use super::{CustomOp, Op};
use crate::types::{ClassicType, QuantumType, Signature, SimpleType};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum LeafOp {
    /// A user-defined operation that can be downcasted by the extensions that
    /// define it.
    ///
    /// TODO: We could replace the `Box` with an `Arc` to reduce memory usage,
    /// but it adds atomic ops and a serialization-deserialization roundtrip
    /// would still generate copies. Maybe it can be complemented with a custom
    /// deserializer that avoids the clones.
    CustomOp {
        id: SmolStr,
        custom_op: Box<dyn CustomOp>,
    },

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

impl PartialEq for LeafOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::CustomOp { id, .. }, Self::CustomOp { id: other_id, .. }) => id.eq(other_id),
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
            LeafOp::CustomOp { custom_op, .. } => custom_op.signature(),
        }
    }

    fn name(&self) -> &str {
        match self {
            LeafOp::CustomOp { id, .. } => id.as_str(),
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
