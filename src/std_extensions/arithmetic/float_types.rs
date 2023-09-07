//! Basic floating-point types

use smol_str::SmolStr;

use crate::{
    types::{CustomCheckFailure, CustomType, Type, TypeBound},
    values::{CustomConst, KnownTypeConst},
    Extension,
};

/// The extension identifier.
pub const EXTENSION_ID: SmolStr = SmolStr::new_inline("arithmetic.float.types");

/// Identfier for the 64-bit IEEE 754-2019 floating-point type.
const FLOAT_TYPE_ID: SmolStr = SmolStr::new_inline("float64");

/// 64-bit IEEE 754-2019 floating-point type (as [CustomType])
pub const FLOAT64_CUSTOM_TYPE: CustomType =
    CustomType::new_simple(FLOAT_TYPE_ID, EXTENSION_ID, TypeBound::Copyable);

/// 64-bit IEEE 754-2019 floating-point type (as [Type])
pub const FLOAT64_TYPE: Type = Type::new_extension(FLOAT64_CUSTOM_TYPE);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A floating-point value.
pub struct ConstF64 {
    #[allow(missing_docs)]
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
        Self { value }
    }
}

impl KnownTypeConst for ConstF64 {
    const TYPE: CustomType = FLOAT64_CUSTOM_TYPE;
}

#[typetag::serde]
impl CustomConst for ConstF64 {
    fn name(&self) -> SmolStr {
        format!("f64({})", self.value).into()
    }

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        self.check_known_type(typ)
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }
}

/// Extension for basic floating-point types.
pub fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_ID);

    extension
        .add_type(
            FLOAT_TYPE_ID,
            vec![],
            "64-bit IEEE 754-2019 floating-point value".to_owned(),
            TypeBound::Copyable.into(),
        )
        .unwrap();

    extension
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_float_types_extension() {
        let r = extension();
        assert_eq!(r.name(), "arithmetic.float.types");
        assert_eq!(r.types().count(), 1);
        assert_eq!(r.operations().count(), 0);
    }

    #[test]
    fn test_float_consts() {
        let const_f64_1 = ConstF64::new(1.0);
        let const_f64_2 = ConstF64::new(2.0);
        assert_ne!(const_f64_1, const_f64_2);
        assert_eq!(const_f64_1, ConstF64::new(1.0));
    }
}
