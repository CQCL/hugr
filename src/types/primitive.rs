//! Primitive types which are leaves of the type tree

use crate::ops::AliasDecl;

use super::{AbstractSignature, CustomType, TypeBound};

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
pub(super) enum PrimType {
    E(Box<CustomType>),
    #[display(fmt = "Alias({})", "_0.name()")]
    A(AliasDecl),
    #[display(fmt = "Graph({})", "_0")]
    Graph(Box<AbstractSignature>),
}

impl PrimType {
    pub(super) fn bound(&self) -> Option<TypeBound> {
        // TODO update once inner types are updated to new TypeBound
        return None;
        match self {
            PrimType::E(_c) => todo!(),
            PrimType::A(_) => todo!(),
            PrimType::Graph(_) => Some(TypeBound::Copyable),
        }
    }
}
