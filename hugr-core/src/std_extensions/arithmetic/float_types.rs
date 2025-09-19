//! Basic floating-point types

use std::sync::{Arc, LazyLock, Weak};

use crate::ops::constant::{TryHash, ValueName};
use crate::types::TypeName;
use crate::{
    Extension,
    extension::ExtensionId,
    ops::constant::CustomConst,
    types::{CustomType, Type, TypeBound},
};

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.float.types");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// Identifier for the 64-bit IEEE 754-2019 floating-point type.
pub const FLOAT_TYPE_ID: TypeName = TypeName::new_inline("float64");

/// 64-bit IEEE 754-2019 floating-point type (as [`CustomType`])
#[must_use]
pub fn float64_custom_type(extension_ref: &Weak<Extension>) -> CustomType {
    CustomType::new(
        FLOAT_TYPE_ID,
        vec![],
        EXTENSION_ID,
        TypeBound::Copyable,
        extension_ref,
    )
}

/// 64-bit IEEE 754-2019 floating-point type (as [Type])
#[must_use]
pub fn float64_type() -> Type {
    float64_custom_type(&Arc::downgrade(&EXTENSION)).into()
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// A floating-point value.
///
/// This constant type does **not** implement equality. Any two instances of
/// `ConstF64` are considered different.
//
// The main problem for equality checking is comparisons of serialized values
// with different precision in `CustomSerialized`.
// For example, `3.3508025818765467e243 /= 3.350802581876547e243` for serde,
// but they would be equal after loaded as a `ConstF64` type.
//
// `serde_json` provides some options to overcome this issue, but since
// custom values are encoded inside `serde_json::Value`s they are not directly
// reachable by these solutions.
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
    /// Name of the constructor for creating constant 64bit floats.
    pub(crate) const CTR_NAME: &'static str = "arithmetic.float.const_f64";

    /// Create a new [`ConstF64`]
    #[must_use]
    pub fn new(value: f64) -> Self {
        // This function can't be `const` because `is_finite()` is not yet stable as a const function.
        assert!(value.is_finite(), "ConstF64 must have a finite value.");
        Self { value }
    }

    /// Returns the value of the constant
    #[must_use]
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
        float64_type()
    }

    fn equal_consts(&self, _: &dyn CustomConst) -> bool {
        false
    }
}

/// Extension defining the float type.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        extension
            .add_type(
                FLOAT_TYPE_ID,
                vec![],
                "64-bit IEEE 754-2019 floating-point value".to_owned(),
                TypeBound::Copyable.into(),
                extension_ref,
            )
            .unwrap();
    })
});

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
