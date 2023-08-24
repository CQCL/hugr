//! Basic integer types

use smol_str::SmolStr;

use crate::{
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        ConstTypeError, CustomCheckFailure, CustomType, Type, TypeBound,
    },
    values::CustomConst,
    Extension,
};
use lazy_static::lazy_static;
/// The extension identifier.
pub const EXTENSION_ID: SmolStr = SmolStr::new_inline("arithmetic.int.types");

/// Identifier for the integer type.
const INT_TYPE_ID: SmolStr = SmolStr::new_inline("int");

fn int_custom_type(width_arg: TypeArg) -> CustomType {
    CustomType::new(INT_TYPE_ID, [width_arg], EXTENSION_ID, TypeBound::Copyable)
}

/// Integer type of a given bit width (specified by the TypeArg).
/// Depending on the operation, the semantic interpretation may be unsigned integer, signed integer
/// or bit string.
pub(super) fn int_type(width_arg: TypeArg) -> Type {
    Type::new_extension(int_custom_type(width_arg))
}

lazy_static! {
    /// Array of valid integer types, indexed by log width of the integer.
    pub static ref INT_TYPES: [Type; (MAX_LOG_WIDTH + 1) as usize] = (0..MAX_LOG_WIDTH + 1)
        .map(|i| int_type(TypeArg::BoundedUSize(i as u64)))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
}

const fn is_valid_log_width(n: u8) -> bool {
    n <= MAX_LOG_WIDTH
}

/// The largest allowed log width.
pub const MAX_LOG_WIDTH: u8 = 7;
/// Type parameter for the log width of the integer.
pub const LOG_WIDTH_TYPE_PARAM: TypeParam = TypeParam::BoundedUSize(MAX_LOG_WIDTH as u64);

/// Get the log width  of the specified type argument or error if the argument
/// is invalid.
pub(super) fn get_log_width(arg: &TypeArg) -> Result<u8, TypeArgError> {
    match arg {
        TypeArg::BoundedUSize(n) if is_valid_log_width(*n as u8) => Ok(*n as u8),
        _ => Err(TypeArgError::TypeMismatch {
            arg: arg.clone(),
            param: LOG_WIDTH_TYPE_PARAM,
        }),
    }
}

pub(super) const fn type_arg(log_width: u8) -> TypeArg {
    TypeArg::BoundedUSize(log_width as u64)
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
        if typ.clone() == int_custom_type(type_arg(self.log_width)) {
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
        if typ.clone() == int_custom_type(type_arg(self.log_width)) {
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
    use cool_asserts::assert_matches;

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
        assert_matches!(get_log_width(&type_arg_32), Ok(5));

        let type_arg_128 = TypeArg::BoundedUSize(7);
        assert_matches!(get_log_width(&type_arg_128), Ok(7));
        let type_arg_256 = TypeArg::BoundedUSize(8);
        assert_matches!(
            get_log_width(&type_arg_256),
            Err(TypeArgError::TypeMismatch { .. })
        );
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
