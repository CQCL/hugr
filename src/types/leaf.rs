//! Primitive types which are leaves of the type tree
#![allow(missing_docs)]

use crate::ops::AliasDecl;

use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;

use super::CustomType;
#[derive(
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    Debug,
    derive_more::Display,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum TypeTag {
    Eq,
    Copyable,
}

impl TypeTag {
    /// Returns the smallest TypeTag containing both the receiver and argument.
    /// (This will be one of the receiver or the argument.)
    pub fn union(self, other: Self) -> Self {
        if self.contains(other) {
            self
        } else {
            debug_assert!(other.contains(self));
            other
        }
    }

    /// Report if this tag contains another.
    pub fn contains(&self, other: TypeTag) -> bool {
        use TypeTag::*;
        match (self, other) {
            (Copyable, Eq) => true,
            (Eq, Copyable) => false,
            _ => true,
        }
    }
}

pub fn containing_tag(mut tags: impl Iterator<Item = Option<TypeTag>>) -> Option<TypeTag> {
    tags.fold_while(Some(TypeTag::Eq), |acc, new| {
        if let (Some(acc), Some(new)) = (acc, new) {
            Continue(Some(acc.union(new)))
        } else {
            Done(None)
        }
    })
    .into_inner()
}

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
pub enum PrimType {
    E(CustomType),
    #[display(fmt = "Alias({})", "_0.name()")]
    A(AliasDecl),
}

impl PrimType {
    pub fn tag(&self) -> Option<TypeTag> {
        return None;
        match self {
            PrimType::E(_c) => todo!(),
            PrimType::A(_) => todo!(),
        }
    }
}
