//! Primitive types which are leaves of the type tree
#![allow(missing_docs)]

use crate::ops::AliasDecl;

use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::CustomType;
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, derive_more::Display, Serialize, Deserialize)]
pub enum TypeBound {
    #[serde(rename = "e")]
    Eq,
    #[serde(rename = "c")]
    Copyable,
}

impl TypeBound {
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

    /// Report if this bound contains another.
    pub fn contains(&self, other: TypeBound) -> bool {
        use TypeBound::*;
        match (self, other) {
            (Copyable, Eq) => true,
            (Eq, Copyable) => false,
            _ => true,
        }
    }
}

pub fn least_upper_bound(mut tags: impl Iterator<Item = Option<TypeBound>>) -> Option<TypeBound> {
    tags.fold_while(Some(TypeBound::Eq), |acc, new| {
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
    pub fn bound(&self) -> Option<TypeBound> {
        return None;
        match self {
            PrimType::E(_c) => todo!(),
            PrimType::A(_) => todo!(),
        }
    }
}
