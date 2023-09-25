//! Primitive types which are leaves of the type tree

use crate::ops::AliasDecl;

use super::{CustomType, PolyFuncType, TypeBound};

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
pub(super) enum PrimType {
    // TODO optimise with Box<CustomType> ?
    // or some static version of this?
    Extension(CustomType),
    #[display(fmt = "Alias({})", "_0.name()")]
    Alias(AliasDecl),
    #[display(fmt = "Function({})", "_0")]
    Function(Box<PolyFuncType>),
    // DeBruijn index, and cache of TypeBound (checked in validation)
    #[display(fmt = "Variable({})", _0)]
    Variable(usize, TypeBound),
}

impl PrimType {
    pub(super) fn bound(&self) -> TypeBound {
        match self {
            PrimType::Extension(c) => c.bound(),
            PrimType::Alias(a) => a.bound,
            PrimType::Function(_) => TypeBound::Copyable,
            PrimType::Variable(_, b) => *b,
        }
    }
}
