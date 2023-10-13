//! Primitive types which are leaves of the type tree

use crate::ops::AliasDecl;

use super::{CustomType, FunctionType, TypeBound};

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
/// Representation of a Primitive type, i.e. neither a Sum nor a Tuple.
pub enum PrimType {
    // TODO optimise with Box<CustomType> ?
    // or some static version of this?
    #[allow(missing_docs)]
    Extension(CustomType),
    #[allow(missing_docs)]
    #[display(fmt = "Alias({})", "_0.name()")]
    Alias(AliasDecl),
    #[allow(missing_docs)]
    #[display(fmt = "Function({})", "_0")]
    Function(Box<FunctionType>),
    #[allow(missing_docs)]
    // DeBruijn index, and cache of TypeBound (checked in validation)
    #[display(fmt = "Variable({})", _0)]
    Variable(usize, TypeBound),
}

impl PrimType {
    /// Returns the bound of this [`PrimType`].
    pub fn bound(&self) -> TypeBound {
        match self {
            PrimType::Extension(c) => c.bound(),
            PrimType::Alias(a) => a.bound,
            PrimType::Function(_) => TypeBound::Copyable,
            PrimType::Variable(_, b) => *b,
        }
    }
}
