//! Representation of custom constant values.
//!
//! These can be used as [`Const`] operations in HUGRs.
//!
//! [`Const`]: crate::ops::Const

use std::any::Any;

use downcast_rs::{impl_downcast, Downcast};

use crate::extension::ExtensionSet;
use crate::macros::impl_box_clone;

use crate::types::{CustomCheckFailure, Type};

use super::ValueName;

/// Constant value for opaque [`CustomType`]s.
///
/// When implementing this trait, include the `#[typetag::serde]` attribute to
/// enable serialization.
///
/// [`CustomType`]: crate::types::CustomType
#[typetag::serde(tag = "c")]
pub trait CustomConst:
    Send + Sync + std::fmt::Debug + CustomConstBoxClone + Any + Downcast
{
    /// An identifier for the constant.
    fn name(&self) -> ValueName;

    /// The extension(s) defining the custom constant
    /// (a set to allow, say, a [List] of [USize])
    ///
    /// [List]: crate::std_extensions::collections::LIST_TYPENAME
    /// [USize]: crate::extension::prelude::USIZE_T
    fn extension_reqs(&self) -> ExtensionSet;

    /// Check the value is a valid instance of the provided type.
    fn validate(&self) -> Result<(), CustomCheckFailure> {
        Ok(())
    }

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    ///
    /// If the type implements `PartialEq`, use [`downcast_equal_consts`] to compare the values.
    fn equal_consts(&self, _other: &dyn CustomConst) -> bool {
        // false unless overloaded
        false
    }

    /// report the type
    fn get_type(&self) -> Type;
}

/// Const equality for types that have PartialEq
pub fn downcast_equal_consts<T: CustomConst + PartialEq>(
    constant: &T,
    other: &dyn CustomConst,
) -> bool {
    if let Some(other) = other.as_any().downcast_ref::<T>() {
        constant == other
    } else {
        false
    }
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A constant value stored as a serialized blob that can report its own type.
pub struct CustomSerialized {
    typ: Type,
    value: serde_yaml::Value,
    extensions: ExtensionSet,
}

impl CustomSerialized {
    /// Creates a new [`CustomSerialized`].
    pub fn new(
        typ: impl Into<Type>,
        value: serde_yaml::Value,
        exts: impl Into<ExtensionSet>,
    ) -> Self {
        Self {
            typ: typ.into(),
            value,
            extensions: exts.into(),
        }
    }

    /// Returns the inner value.
    pub fn value(&self) -> &serde_yaml::Value {
        &self.value
    }
}

#[typetag::serde]
impl CustomConst for CustomSerialized {
    fn name(&self) -> ValueName {
        format!("yaml:{:?}", self.value).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        Some(self) == other.downcast_ref()
    }

    fn extension_reqs(&self) -> ExtensionSet {
        self.extensions.clone()
    }
    fn get_type(&self) -> Type {
        self.typ.clone()
    }
}

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}
