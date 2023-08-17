//! Basic floating-point types

use smol_str::SmolStr;

use crate::{
    types::{CustomCheckFailure, CustomType, Type, TypeBound},
    values::CustomConst,
    Resource,
};

/// The resource identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("arithmetic.float.types");

/// Identfier for the 64-bit IEEE 754-2019 floating-point type.
const FLOAT_TYPE_ID: SmolStr = SmolStr::new_inline("float64");

fn float64_custom_type() -> CustomType {
    CustomType::new(FLOAT_TYPE_ID, [], RESOURCE_ID, TypeBound::Copyable)
}

/// 64-bit IEEE 754-2019 floating-point type
pub fn float64_type() -> Type {
    Type::new_extension(float64_custom_type())
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A floating-point value.
pub struct ConstF64(f64);

impl ConstF64 {
    /// Create a new [`ConstF64`]
    pub fn new(value: f64) -> Self {
        Self(value)
    }
}

#[typetag::serde]
impl CustomConst for ConstF64 {
    fn name(&self) -> SmolStr {
        format!("f64({})", self.0).into()
    }

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if typ.clone() == float64_custom_type() {
            Ok(())
        } else {
            Err(CustomCheckFailure::Message(
                "Floating-point constant type mismatch.".into(),
            ))
        }
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }
}

/// Resource for basic floating-point types.
pub fn resource() -> Resource {
    let mut resource = Resource::new(RESOURCE_ID);

    resource
        .add_type(
            FLOAT_TYPE_ID,
            vec![],
            "64-bit IEEE 754-2019 floating-point value".to_owned(),
            TypeBound::Copyable.into(),
        )
        .unwrap();

    resource
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_float_types_resource() {
        let r = resource();
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
