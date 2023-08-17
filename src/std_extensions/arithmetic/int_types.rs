//! Basic integer types

use smol_str::SmolStr;

use crate::{
    extension::SignatureError,
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        ConstTypeError, CustomCheckFailure, CustomType, Type, TypeBound,
    },
    values::CustomConst,
    Extension,
};

/// The extension identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("arithmetic.int.types");

/// Identfier for the integer type.
const INT_TYPE_ID: SmolStr = SmolStr::new_inline("int");

fn int_custom_type(n: u8) -> CustomType {
    CustomType::new(
        INT_TYPE_ID,
        [TypeArg::USize(n as u64)],
        RESOURCE_ID,
        TypeBound::Copyable,
    )
}

/// Integer type of a given bit width.
/// Depending on the operation, the semantic interpretation may be unsigned integer, signed integer
/// or bit string.
pub fn int_type(n: u8) -> Type {
    Type::new_extension(int_custom_type(n))
}

fn is_valid_width(n: u8) -> bool {
    (n == 1)
        || (n == 2)
        || (n == 4)
        || (n == 8)
        || (n == 16)
        || (n == 32)
        || (n == 64)
        || (n == 128)
}

/// Get the bit width of the specified integer type, or error if the width is not supported.
pub fn get_width(arg: &TypeArg) -> Result<u8, SignatureError> {
    let n: u8 = match arg {
        TypeArg::USize(n) => *n as u8,
        _ => {
            return Err(TypeArgError::TypeMismatch {
                arg: arg.clone(),
                param: TypeParam::USize,
            }
            .into());
        }
    };
    if !is_valid_width(n) {
        return Err(TypeArgError::InvalidValue(arg.clone()).into());
    }
    Ok(n)
}

/// An unsigned integer
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstIntU {
    width: u8,
    value: u128,
}

/// A signed integer
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstIntS {
    width: u8,
    value: i128,
}

impl ConstIntU {
    /// Create a new [`ConstIntU`]
    pub fn new(width: u8, value: u128) -> Result<Self, ConstTypeError> {
        if !is_valid_width(width) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid integer width.".to_owned()),
            ));
        }
        if (width <= 64) && (value >= (1u128 << width)) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid unsigned integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { width, value })
    }
}

impl ConstIntS {
    /// Create a new [`ConstIntS`]
    pub fn new(width: u8, value: i128) -> Result<Self, ConstTypeError> {
        if !is_valid_width(width) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid integer width.".to_owned()),
            ));
        }
        if (width <= 64) && (value >= (1i128 << (width - 1)) || value < -(1i128 << (width - 1))) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid signed integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { width, value })
    }
}

#[typetag::serde]
impl CustomConst for ConstIntU {
    fn name(&self) -> SmolStr {
        format!("u{}({})", self.width, self.value).into()
    }
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if typ.clone() == int_custom_type(self.width) {
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
        format!("i{}({})", self.width, self.value).into()
    }
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if typ.clone() == int_custom_type(self.width) {
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
    let mut extension = Extension::new(RESOURCE_ID);

    extension
        .add_type(
            INT_TYPE_ID,
            vec![TypeParam::USize],
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
        let type_arg_32 = TypeArg::USize(32);
        assert_matches!(get_width(&type_arg_32), Ok(32));

        let type_arg_33 = TypeArg::USize(33);
        assert_matches!(
            get_width(&type_arg_33),
            Err(SignatureError::TypeArgMismatch(_))
        );

        let type_arg_128 = TypeArg::USize(128);
        assert_matches!(get_width(&type_arg_128), Ok(128));

        let type_arg_256 = TypeArg::USize(256);
        assert_matches!(
            get_width(&type_arg_256),
            Err(SignatureError::TypeArgMismatch(_))
        );
    }

    #[test]
    fn test_int_consts() {
        let const_u32_7 = ConstIntU::new(32, 7);
        let const_u64_7 = ConstIntU::new(64, 7);
        let const_u32_8 = ConstIntU::new(32, 8);
        assert_ne!(const_u32_7, const_u64_7);
        assert_ne!(const_u32_7, const_u32_8);
        assert_eq!(const_u32_7, ConstIntU::new(32, 7));
        assert_matches!(
            ConstIntU::new(8, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert_matches!(
            ConstIntU::new(9, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert_matches!(
            ConstIntS::new(8, 128),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert_matches!(ConstIntS::new(8, -128), Ok(_));
    }
}
