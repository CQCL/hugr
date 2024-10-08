//! Basic floating-point types

use crate::ops::constant::{TryHash, ValueName};
use crate::types::TypeName;
use crate::{
    extension::{ExtensionId, ExtensionSet},
    ops::constant::CustomConst,
    types::{CustomType, Type, TypeBound},
    Extension,
};
use lazy_static::lazy_static;

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.float.types");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// Identifier for the 64-bit IEEE 754-2019 floating-point type.
const FLOAT_TYPE_ID: TypeName = TypeName::new_inline("float64");

/// 64-bit IEEE 754-2019 floating-point type (as [CustomType])
pub const FLOAT64_CUSTOM_TYPE: CustomType =
    CustomType::new_simple(FLOAT_TYPE_ID, EXTENSION_ID, TypeBound::Copyable);

/// 64-bit IEEE 754-2019 floating-point type (as [Type])
pub const FLOAT64_TYPE: Type = Type::new_extension(FLOAT64_CUSTOM_TYPE);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// A floating-point value.
pub struct ConstF64 {
    /// The value.
    value: f64,
}

impl std::ops::Deref for ConstF64 {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl ConstF64 {
    /// Create a new [`ConstF64`]
    pub fn new(value: f64) -> Self {
        // This function can't be `const` because `is_finite()` is not yet stable as a const function.
        if !value.is_finite() {
            panic!("ConstF64 must have a finite value.");
        }
        Self { value }
    }

    /// Returns the value of the constant
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl TryHash for ConstF64 {}

#[typetag::serde]
impl CustomConst for ConstF64 {
    fn name(&self) -> ValueName {
        format!("f64({})", self.value).into()
    }

    fn get_type(&self) -> Type {
        FLOAT64_TYPE
    }

    fn equal_consts(&self, _: &dyn CustomConst) -> bool {
        false
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&EXTENSION_ID)
    }
}

lazy_static! {
    /// Extension defining the float type.
    pub static ref EXTENSION: Extension = {
        let mut extension = Extension::new(EXTENSION_ID, VERSION);

        extension
            .add_type(
                FLOAT_TYPE_ID,
                vec![],
                "64-bit IEEE 754-2019 floating-point value".to_owned(),
                TypeBound::Copyable.into(),
            )
            .unwrap();

        extension
    };
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_float_types_extension() {
        let r = &EXTENSION;
        assert_eq!(r.name() as &str, "arithmetic.float.types");
        assert_eq!(r.types().count(), 1);
        assert_eq!(r.operations().count(), 0);
    }

    #[test]
    fn test_float_consts() {
        let const_f64_1 = ConstF64::new(1.0);
        let const_f64_2 = ConstF64::new(2.0);

        assert_eq!(const_f64_1.value(), 1.0);
        assert_eq!(*const_f64_2, 2.0);
        assert_eq!(const_f64_1.name(), "f64(1)");
        // ConstF64 does not support `equal_consts`
        assert!(!const_f64_1.equal_consts(&ConstF64::new(1.0)));
    }
}
