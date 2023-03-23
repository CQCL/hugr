use crate::macros::impl_box_clone;
use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;
use std::any::Any;

/// An opaque type element. Contains an unique identifier and a reference to its definition.
///
/// TODO: We could replace the `Box` with an `Arc` to reduce memory usage,
/// but it adds atomic ops and a serialization-deserialization roundtrip
/// would still generate copies.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    /// Unique identifier of the opaque type.
    id: SmolStr,
    custom_type: Box<dyn CustomTypeTrait>,
}

impl PartialEq for CustomType {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CustomType {}

#[typetag::serde]
impl CustomTypeTrait for CustomType {
    fn is_linear(&self) -> bool {
        self.custom_type.is_linear()
    }
}

/// A custom defined type that can be downcasted by the extensions that know
/// about it.
///
/// Note that any implementation of this trait must include the
/// `#[typetag::serde]` attribute.
///
/// TODO: Is this trait necessary? Can't we just use a struct?
#[typetag::serde]
pub trait CustomTypeTrait:
    Send + Sync + std::fmt::Debug + Any + Downcast + CustomTypeBoxClone
{
    fn is_linear(&self) -> bool {
        false
    }
}

impl_downcast!(CustomTypeTrait);
impl_box_clone!(CustomTypeTrait, CustomTypeBoxClone);
