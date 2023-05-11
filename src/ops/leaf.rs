//! Definition of the leaf operations.

use smol_str::SmolStr;

use super::OpaqueOp;
use crate::{
    type_row,
    types::{ClassicType, LinearType, Signature, SignatureDescription, SimpleType, TypeRow},
};

/// Dataflow operations with no children.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum LeafOp {
    /// A user-defined operation that can be downcasted by the extensions that
    /// define it.
    CustomOp(OpaqueOp),
    /// A Hadamard gate.
    H,
    /// A T gate.
    T,
    /// An S gate.
    S,
    /// A Pauli X gate.
    X,
    /// A Pauli Y gate.
    Y,
    /// A Pauli Z gate.
    Z,
    /// An adjoint T gate.
    Tadj,
    /// An adjoint S gate.
    Sadj,
    /// A controlled X gate.
    CX,
    /// A maximally entangling ZZ phase gate.
    ZZMax,
    /// A qubit reset operation.
    Reset,
    /// A no-op operation.
    Noop(SimpleType),
    /// A qubit measurement operation.
    Measure,
    /// A copy operation for classical data.
    Copy {
        /// The number of copies to make.
        ///
        /// Note that a 0-ary copy acts as an explicit discard.
        /// Like any stateful operation with no dataflow outputs, such
        /// a copy should have a State output connecting it to the Output node.
        n_copies: u32,
        /// The type of the data to copy.
        typ: ClassicType,
    },
    /// A bitwise XOR operation.
    Xor,
    /// An operation that packs all its inputs into a tuple.
    MakeTuple(TypeRow),
    /// An operation that unpacks a tuple into its components.
    UnpackTuple(TypeRow),
    /// An operation that wraps a value into a new type.
    MakeNewType {
        /// The new type name.
        name: SmolStr,
        /// The wrapped type.
        typ: SimpleType,
    },
    /// An operation that creates a tagged sum value from one of its variants.
    Tag {
        /// The variant to create.
        tag: usize,
        /// The variants of the sum type.
        variants: TypeRow,
    },
}

impl Default for LeafOp {
    fn default() -> Self {
        Self::Noop(SimpleType::default())
    }
}

impl LeafOp {
    /// Returns the number of linear inputs (also outputs) of the operation
    pub fn linear_count(&self) -> usize {
        self.signature().linear().count()
    }

    /// Returns true if the operation has only classical inputs and outputs.
    pub fn is_pure_classical(&self) -> bool {
        self.signature().purely_classical()
    }

    /// The name of the operation.
    pub fn name(&self) -> SmolStr {
        match self {
            LeafOp::CustomOp(opaque) => return opaque.name(),
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
            LeafOp::MakeTuple(_) => "MakeTuple",
            LeafOp::UnpackTuple(_) => "UnpackTuple",
            LeafOp::MakeNewType { .. } => "MakeNewType",
            LeafOp::Tag { .. } => "Tag",
        }
        .into()
    }

    /// A human-readable description of the operation.
    pub fn description(&self) -> &str {
        match self {
            LeafOp::CustomOp(opaque) => opaque.description(),
            LeafOp::H => "Hadamard gate",
            LeafOp::T => "T gate",
            LeafOp::S => "S gate",
            LeafOp::X => "Pauli X gate",
            LeafOp::Y => "Pauli Y gate",
            LeafOp::Z => "Pauli Z gate",
            LeafOp::Tadj => "Adjoint T gate",
            LeafOp::Sadj => "Adjoint S gate",
            LeafOp::CX => "Controlled X gate",
            LeafOp::ZZMax => "Maximally entangling ZZPhase gate",
            LeafOp::Reset => "Qubit reset",
            LeafOp::Noop(_) => "Noop gate",
            LeafOp::Measure => "Qubit measurement gate",
            LeafOp::Copy { .. } => "Classical data copy",
            LeafOp::Xor => "Bitwise XOR",
            LeafOp::MakeTuple(_) => "MakeTuple operation",
            LeafOp::UnpackTuple(_) => "UnpackTuple operation",
            LeafOp::Tag { .. } => "Tag Sum operation",
            LeafOp::MakeNewType { .. } => {
                "Make a new type value from a value of the defining type."
            }
        }
    }

    /// The signature of the operation.
    pub fn signature(&self) -> Signature {
        // Static signatures. The `TypeRow`s in the `Signature` use a
        // copy-on-write strategy, so we can avoid unnecessary allocations.
        const Q: SimpleType = SimpleType::Linear(LinearType::Qubit);
        const B: SimpleType = SimpleType::Classic(ClassicType::bit());

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
            | LeafOp::Z => Signature::new_linear(type_row![Q]),
            LeafOp::CX | LeafOp::ZZMax => Signature::new_linear(type_row![Q, Q]),
            LeafOp::Measure => Signature::new_linear(type_row![Q, B]),
            LeafOp::Copy { n_copies, typ } => {
                let typ: SimpleType = typ.clone().into();
                Signature::new_df(vec![typ.clone()], vec![typ; *n_copies as usize])
            }
            LeafOp::Xor => Signature::new_df(type_row![B, B], type_row![B]),
            LeafOp::CustomOp(opaque) => opaque.signature(),
            LeafOp::MakeTuple(types) => {
                Signature::new_df(types.clone(), vec![SimpleType::new_tuple(types.clone())])
            }
            LeafOp::UnpackTuple(types) => {
                Signature::new_df(vec![SimpleType::new_tuple(types.clone())], types.clone())
            }
            LeafOp::Tag { tag, variants } => Signature::new_df(
                vec![variants.get(*tag).expect("Not a valid tag").clone()],
                vec![SimpleType::new_sum(variants.clone())],
            ),
            LeafOp::MakeNewType { typ, name } => Signature::new_df(
                vec![typ.clone()],
                vec![typ.clone().into_new_type(name.clone())],
            ),
        }
    }

    /// Optional description of the ports in the signature.
    pub fn signature_desc(&self) -> SignatureDescription {
        match self {
            LeafOp::CustomOp(opaque) => opaque.signature_desc(),
            // TODO: More port descriptions
            _ => Default::default(),
        }
    }
}
