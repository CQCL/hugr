//! Primitive types which are leaves of the type tree
#![allow(missing_docs)]

use std::marker::PhantomData;

use crate::ops::AliasDecl;

use super::{AbstractSignature, CustomType, TypeTag};
use thiserror::Error;

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum EqLeaf {
    USize,
}

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum ClassicLeaf {
    E(EqLeaf),
    Graph(Box<AbstractSignature>),
}

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum AnyLeaf {
    C(ClassicLeaf),
}

impl From<EqLeaf> for ClassicLeaf {
    fn from(value: EqLeaf) -> Self {
        ClassicLeaf::E(value)
    }
}

impl<T: Into<ClassicLeaf>> From<T> for AnyLeaf {
    fn from(value: T) -> Self {
        AnyLeaf::C(value.into())
    }
}

pub(crate) mod sealed {
    use super::{AnyLeaf, ClassicLeaf, EqLeaf};
    pub trait Sealed {}
    impl Sealed for AnyLeaf {}
    impl Sealed for ClassicLeaf {}
    impl Sealed for EqLeaf {}
}

pub trait TypeClass: sealed::Sealed {
    const BOUND_TAG: TypeTag;
}

impl TypeClass for EqLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Hashable;
}

impl TypeClass for ClassicLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Classic;
}

impl TypeClass for AnyLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Simple;
}

#[derive(Clone, PartialEq, Debug, Eq)]
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
