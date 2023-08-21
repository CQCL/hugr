//! Definition of the leaf operations.

use smol_str::SmolStr;

use super::custom::ExternalOp;
use super::{OpName, OpTag, OpTrait, StaticTag};

use crate::{
    extension::{ExtensionId, ExtensionSet},
    types::{AbstractSignature, EdgeKind, SignatureDescription, Type, TypeRow},
};

/// Dataflow operations with no children.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(tag = "lop")]
pub enum LeafOp {
    /// A user-defined operation that can be downcasted by the extensions that
    /// define it.
    CustomOp(Box<ExternalOp>),

    /// A no-op operation.
    Noop {
        /// The type of edges connecting the Noop.
        ty: Type,
    },
    /// An operation that packs all its inputs into a tuple.
    MakeTuple {
        ///Tuple element types.
        tys: TypeRow,
    },
    /// An operation that unpacks a tuple into its components.
    UnpackTuple {
        ///Tuple element types.
        tys: TypeRow,
    },
    /// An operation that creates a tagged sum value from one of its variants.
    Tag {
        /// The variant to create.
        tag: usize,
        /// The variants of the sum type.
        variants: TypeRow,
    },
    /// A node which adds a extension req to the types of the wires it is passed
    /// It has no effect on the values passed along the edge
    Lift {
        /// The types of the edges
        type_row: TypeRow,
        /// The extensions which we're adding to the inputs
        new_extension: ExtensionId,
    },
}

impl Default for LeafOp {
    fn default() -> Self {
        Self::Noop { ty: Type::UNIT }
    }
}
impl OpName for LeafOp {
    /// The name of the operation.
    fn name(&self) -> SmolStr {
        match self {
            LeafOp::CustomOp(ext) => return ext.name(),
            LeafOp::Noop { ty: _ } => "Noop",
            LeafOp::MakeTuple { tys: _ } => "MakeTuple",
            LeafOp::UnpackTuple { tys: _ } => "UnpackTuple",
            LeafOp::Tag { .. } => "Tag",
            LeafOp::Lift { .. } => "Lift",
        }
        .into()
    }
}

impl StaticTag for LeafOp {
    const TAG: OpTag = OpTag::Leaf;
}

impl OpTrait for LeafOp {
    /// A human-readable description of the operation.
    fn description(&self) -> &str {
        match self {
            LeafOp::CustomOp(ext) => ext.description(),
            LeafOp::Noop { ty: _ } => "Noop gate",
            LeafOp::MakeTuple { tys: _ } => "MakeTuple operation",
            LeafOp::UnpackTuple { tys: _ } => "UnpackTuple operation",
            LeafOp::Tag { .. } => "Tag Sum operation",
            LeafOp::Lift { .. } => "Add a extension requirement to an edge",
        }
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    /// The signature of the operation.
    fn signature(&self) -> AbstractSignature {
        // Static signatures. The `TypeRow`s in the `AbstractSignature` use a
        // copy-on-write strategy, so we can avoid unnecessary allocations.

        match self {
            LeafOp::Noop { ty: typ } => {
                AbstractSignature::new(vec![typ.clone()], vec![typ.clone()])
            }
            LeafOp::CustomOp(ext) => ext.signature(),
            LeafOp::MakeTuple { tys: types } => {
                AbstractSignature::new(types.clone(), vec![Type::new_tuple(types.clone())])
            }
            LeafOp::UnpackTuple { tys: types } => {
                AbstractSignature::new(vec![Type::new_tuple(types.clone())], types.clone())
            }
            LeafOp::Tag { tag, variants } => AbstractSignature::new(
                vec![variants.get(*tag).expect("Not a valid tag").clone()],
                vec![Type::new_sum(variants.clone())],
            ),
            LeafOp::Lift {
                type_row,
                new_extension,
            } => AbstractSignature::new(type_row.clone(), type_row.clone())
                .with_extension_delta(&ExtensionSet::singleton(new_extension)),
        }
    }

    /// Optional description of the ports in the signature.
    fn signature_desc(&self) -> SignatureDescription {
        match self {
            LeafOp::CustomOp(ext) => ext.signature_desc(),
            // TODO: More port descriptions
            _ => Default::default(),
        }
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}
