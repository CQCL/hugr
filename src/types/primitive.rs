//! Primitive types which are leaves of the type tree

use crate::{ops::AliasDecl, utils::MaybeRef};

use super::{AbstractSignature, CustomType, TypeBound};

#[derive(Clone, PartialEq, Debug, Eq, derive_more::Display)]
pub(super) enum PrimType {
    // TODO optimise with Box<CustomType> ?
    // or some static version of this?
    #[display(fmt = "{}", "_0.as_ref()")]
    Extension(MaybeRef<'static, CustomType>),
    #[display(fmt = "Alias({})", "_0.name()")]
    Alias(AliasDecl),
    #[display(fmt = "Graph({})", "_0")]
    Graph(Box<AbstractSignature>),
}

impl PrimType {
    pub(super) fn bound(&self) -> TypeBound {
        match self {
            PrimType::Extension(c) => c.as_ref().bound(),
            PrimType::Alias(a) => a.bound,
            PrimType::Graph(_) => TypeBound::Copyable,
        }
    }
}
