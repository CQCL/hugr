//! Representation of values (shared between [Const] and in future [TypeArg])
//!
//! [Const]: crate::ops::Const
//! [TypeArg]: crate::types::type_param::TypeArg

use std::any::Any;

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

use crate::extension::ExtensionSet;
use crate::macros::impl_box_clone;

use crate::types::{CustomCheckFailure, Type};

/// Constant value for opaque [`CustomType`]s.
///
/// When implementing this trait, include the `#[typetag::serde]` attribute to
/// enable serialization.
///
/// [CustomType]: crate::types::CustomType
#[typetag::serde(tag = "c")]
pub trait CustomConst:
    Send + Sync + std::fmt::Debug + CustomConstBoxClone + Any + Downcast
{
    /// An identifier for the constant.
    fn name(&self) -> SmolStr;

    /// The extension(s) defining the custom value
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
    // Can't derive PartialEq for trait objects
    fn equal_consts(&self, _other: &dyn CustomConst) -> bool {
        // false unless overloaded
        false
    }

    /// report the type
    fn get_type(&self) -> Type;
}

/// Const equality for types that have PartialEq
pub fn downcast_equal_consts<T: CustomConst + PartialEq>(
    value: &T,
    other: &dyn CustomConst,
) -> bool {
    if let Some(other) = other.as_any().downcast_ref::<T>() {
        value == other
    } else {
        false
    }
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A value stored as a serialized blob that can report its own type.
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
    fn name(&self) -> SmolStr {
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

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::ops::Const;
    use crate::std_extensions::arithmetic::float_types::{self, FLOAT64_TYPE};
    use crate::types::CustomType;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]

    /// A custom constant value used in testing
    pub(crate) struct CustomTestValue(pub CustomType);
    #[typetag::serde]
    impl CustomConst for CustomTestValue {
        fn name(&self) -> SmolStr {
            format!("CustomTestValue({:?})", self.0).into()
        }

        fn extension_reqs(&self) -> ExtensionSet {
            ExtensionSet::singleton(self.0.extension())
        }

        fn get_type(&self) -> Type {
            self.0.clone().into()
        }
    }

    pub(crate) fn serialized_float(f: f64) -> Const {
        CustomSerialized {
            typ: FLOAT64_TYPE,
            value: serde_yaml::Value::Number(f.into()),
            extensions: float_types::EXTENSION_ID.into(),
        }
        .into()
    }
}
