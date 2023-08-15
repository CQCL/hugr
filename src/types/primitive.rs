//! Primitive types which are leaves of the type tree

use crate::ops::AliasDecl;

use super::{AbstractSignature, CustomType, TypeBound};

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
pub(super) enum PrimType {
    // TODO optimise with Box<CustomType> ?
    // or some static version of this?
    E(CustomType),
    #[display(fmt = "Alias({})", "_0.name()")]
    A(AliasDecl),
    #[display(fmt = "Graph({})", "_0")]
    Graph(Box<AbstractSignature>),
}

impl PrimType {
    pub(super) fn bound(&self) -> TypeBound {
        match self {
            PrimType::E(c) => c.bound(),
            PrimType::A(a) => a.bound,
            PrimType::Graph(_) => TypeBound::Copyable,
        }
    }
}
