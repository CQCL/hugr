//! Basic integer types

use smol_str::SmolStr;

use crate::{
    types::{
        type_param::{TypeArg, TypeParam},
        ConstTypeError, CustomCheckFailure, CustomType, Type, TypeBound,
    },
    values::CustomConst,
    Extension,
};

/// The extension identifier.
pub const EXTENSION_ID: SmolStr = SmolStr::new_inline("arithmetic.int.types");

/// Identfier for the integer type.
const INT_TYPE_ID: SmolStr = SmolStr::new_inline("int");

fn int_custom_type(n: u8) -> CustomType {
    CustomType::new(
        INT_TYPE_ID,
        [TypeArg::BoundedUSize(n as u64)],
        EXTENSION_ID,
        TypeBound::Copyable,
    )
}

/// Integer type of a given bit width.
/// Depending on the operation, the semantic interpretation may be unsigned integer, signed integer
/// or bit string.
pub fn int_type(width_power: u8) -> Type {
    Type::new_extension(int_custom_type(width_power))
}

const fn is_valid_log_width(n: u8) -> bool {
    n < POWERS_OF_TWO
}

const POWERS_OF_TWO: u8 = 8;
/// Type parameter for the log width of the integer.
pub const LOG_WIDTH_TYPE_PARAM: TypeParam = TypeParam::BoundedUSize(POWERS_OF_TWO as u64);

/// Get the bit width of the specified integer type, or error if the width is not supported.
pub fn get_width_power(arg: &TypeArg) -> u8 {
    match arg {
        TypeArg::BoundedUSize(n) if is_valid_log_width(*n as u8) => *n as u8,
        _ => panic!("type check should prevent this."),
    }
}

/// An unsigned integer
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstIntU {
    log_width: u8,
    value: u128,
}

/// A signed integer
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstIntS {
    log_width: u8,
    value: i128,
}

impl ConstIntU {
    /// Create a new [`ConstIntU`]
    pub fn new(log_width: u8, value: u128) -> Result<Self, ConstTypeError> {
        if !is_valid_log_width(log_width) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid integer width.".to_owned()),
            ));
        }
        if (log_width <= 6) && (value >= (1u128 << (1u8 << log_width))) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid unsigned integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { log_width, value })
    }
}

impl ConstIntS {
    /// Create a new [`ConstIntS`]
    pub fn new(log_width: u8, value: i128) -> Result<Self, ConstTypeError> {
        if !is_valid_log_width(log_width) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid integer width.".to_owned()),
            ));
        }
        let width = 1u8 << log_width;
        if (log_width <= 6) && (value >= (1i128 << (width - 1)) || value < -(1i128 << (width - 1)))
        {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid signed integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { log_width, value })
    }
}

#[typetag::serde]
impl CustomConst for ConstIntU {
    fn name(&self) -> SmolStr {
        format!("u{}({})", self.log_width, self.value).into()
    }
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if typ.clone() == int_custom_type(self.log_width) {
            Ok(())
        } else {
            Err(CustomCheckFailure::Message(
                "Unsigned integer constant type mismatch.".into(),
            ))
        }
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }
}

#[typetag::serde]
impl CustomConst for ConstIntS {
    fn name(&self) -> SmolStr {
        format!("i{}({})", self.log_width, self.value).into()
    }
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if typ.clone() == int_custom_type(self.log_width) {
            Ok(())
        } else {
            Err(CustomCheckFailure::Message(
                "Signed integer constant type mismatch.".into(),
            ))
        }
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }
}

/// Extension for basic integer types.
pub fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_ID);

    extension
        .add_type(
            INT_TYPE_ID,
            vec![LOG_WIDTH_TYPE_PARAM],
            "integral value of a given bit width".to_owned(),
            TypeBound::Copyable.into(),
        )
        .unwrap();

    extension
}

#[cfg(test)]
mod test {
    use cool_asserts::{assert_matches, assert_panics};

    use super::*;

    #[test]
    fn test_int_types_extension() {
        let r = extension();
        assert_eq!(r.name(), "arithmetic.int.types");
        assert_eq!(r.types().count(), 1);
        assert_eq!(r.operations().count(), 0);
    }

    #[test]
    fn test_int_widths() {
        let type_arg_32 = TypeArg::BoundedUSize(5);
        assert_matches!(get_width_power(&type_arg_32), 5);

        let type_arg_128 = TypeArg::BoundedUSize(7);
        assert_matches!(get_width_power(&type_arg_128), 7);

        assert_panics!(get_width_power(&TypeArg::BoundedUSize(8)));
    }

    #[test]
    fn test_int_consts() {
        let const_u32_7 = ConstIntU::new(5, 7);
        let const_u64_7 = ConstIntU::new(6, 7);
        let const_u32_8 = ConstIntU::new(5, 8);
        assert_ne!(const_u32_7, const_u64_7);
        assert_ne!(const_u32_7, const_u32_8);
        assert_eq!(const_u32_7, ConstIntU::new(5, 7));
        assert_matches!(
            ConstIntU::new(3, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert_matches!(
            ConstIntU::new(9, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert_matches!(
            ConstIntS::new(3, 128),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert_matches!(ConstIntS::new(3, -128), Ok(_));
    }
}
