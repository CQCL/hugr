//! Primitive types which are leaves of the type tree
#![allow(missing_docs)]

use std::marker::PhantomData;

use crate::ops::AliasDecl;

use super::{AbstractSignature, CustomType, TypeTag};
use derive_more::Display;
use thiserror::Error;

#[derive(Clone, PartialEq, Debug, Eq, Display)]
pub enum EqLeaf {
    USize,
}

#[derive(Clone, PartialEq, Debug, Eq, Display)]
pub enum CopyableLeaf {
    E(EqLeaf),
    #[display(fmt = "Graph")]
    Graph(Box<AbstractSignature>),
}

#[derive(Clone, PartialEq, Debug, Eq, Display)]
pub enum AnyLeaf {
    C(CopyableLeaf),
}

impl From<EqLeaf> for CopyableLeaf {
    fn from(value: EqLeaf) -> Self {
        CopyableLeaf::E(value)
    }
}

impl<T: Into<CopyableLeaf>> From<T> for AnyLeaf {
    fn from(value: T) -> Self {
        AnyLeaf::C(value.into())
    }
}

pub(crate) mod sealed {
    use super::{AnyLeaf, CopyableLeaf, EqLeaf};
    pub trait Sealed {}
    impl Sealed for AnyLeaf {}
    impl Sealed for CopyableLeaf {}
    impl Sealed for EqLeaf {}
}

pub trait TypeClass: sealed::Sealed + Clone + 'static {
    const BOUND_TAG: TypeTag;
}

impl TypeClass for EqLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Hashable;
}

impl TypeClass for CopyableLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Classic;
}

impl TypeClass for AnyLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Simple;
}

#[derive(Clone, PartialEq, Debug, Eq, Display)]
#[display(fmt = "{}", "_0")]
pub struct Tagged<I, T>(pub(super) I, pub(super) PhantomData<T>);

pub trait ActualTag {
    fn actual_tag(&self) -> TypeTag;
}

impl ActualTag for CustomType {
    fn actual_tag(&self) -> TypeTag {
        self.tag()
    }
}

impl ActualTag for AliasDecl {
    fn actual_tag(&self) -> TypeTag {
        self.tag
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
#[error("The tag reported by the object ({found:?} is not contained by the tag of the Type ({bound:?}).")]
pub struct InvalidBound {
    pub bound: TypeTag,
    pub found: TypeTag,
}

impl<T: ActualTag, C: TypeClass> Tagged<T, C> {
    pub fn new(inner: T) -> Result<Self, InvalidBound> {
        if C::BOUND_TAG.contains(inner.actual_tag()) {
            Ok(Self(inner, PhantomData))
        } else {
            Err(InvalidBound {
                bound: C::BOUND_TAG,
                found: inner.actual_tag(),
            })
        }
    }
    pub fn inner(&self) -> &T {
        &self.0
    }
}
