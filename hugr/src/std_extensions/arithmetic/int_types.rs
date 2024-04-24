//! Basic integer types

use std::num::NonZeroU64;

use smol_str::SmolStr;

use crate::{
    extension::{ExtensionId, ExtensionSet},
    ops::constant::CustomConst,
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        ConstTypeError, CustomType, Type, TypeBound,
    },
    Extension,
};
use lazy_static::lazy_static;
/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.int.types");

/// Identifier for the integer type.
pub const INT_TYPE_ID: SmolStr = SmolStr::new_inline("int");

fn int_custom_type(width_arg: TypeArg) -> CustomType {
    CustomType::new(INT_TYPE_ID, [width_arg], EXTENSION_ID, TypeBound::Eq)
}

/// Integer type of a given bit width (specified by the TypeArg).
/// Depending on the operation, the semantic interpretation may be unsigned integer, signed integer
/// or bit string.
pub(super) fn int_type(width_arg: TypeArg) -> Type {
    Type::new_extension(int_custom_type(width_arg))
}

lazy_static! {
    /// Array of valid integer types, indexed by log width of the integer.
    pub static ref INT_TYPES: [Type; LOG_WIDTH_BOUND as usize] = (0..LOG_WIDTH_BOUND)
        .map(|i| int_type(TypeArg::BoundedNat { n: i as u64 }))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
}

const fn is_valid_log_width(n: u8) -> bool {
    n < LOG_WIDTH_BOUND
}

/// The maximum allowed log width.
pub const LOG_WIDTH_MAX: u8 = 6;

/// The smallest forbidden log width.
pub const LOG_WIDTH_BOUND: u8 = LOG_WIDTH_MAX + 1;

/// Type parameter for the log width of the integer.
#[allow(clippy::assertions_on_constants)]
pub const LOG_WIDTH_TYPE_PARAM: TypeParam = TypeParam::bounded_nat({
    assert!(LOG_WIDTH_BOUND > 0);
    NonZeroU64::MIN.saturating_add(LOG_WIDTH_BOUND as u64 - 1)
});

/// Get the log width  of the specified type argument or error if the argument
/// is invalid.
pub(super) fn get_log_width(arg: &TypeArg) -> Result<u8, TypeArgError> {
    match arg {
        TypeArg::BoundedNat { n } if is_valid_log_width(*n as u8) => Ok(*n as u8),
        _ => Err(TypeArgError::TypeMismatch {
            arg: arg.clone(),
            param: LOG_WIDTH_TYPE_PARAM,
        }),
    }
}

const fn type_arg(log_width: u8) -> TypeArg {
    TypeArg::BoundedNat {
        n: log_width as u64,
    }
}

/// An integer (either signed or unsigned)
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstInt {
    log_width: u8,
    // We always use a u64 for the value. The interpretation is:
    // - as an unsigned integer, (value mod 2^N);
    // - as a signed integer, (value mod 2^(N-1) - 2^(N-1)*a)
    // where N = 2^log_width and a is the (N-1)th bit of x (counting from
    // 0 = least significant bit).
    value: u64,
}

/// An unsigned integer
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstIntU {
    log_width: u8,
    value: u64,
}

/// A signed integer
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstIntS {
    log_width: u8,
    value: i64,
}

impl ConstInt {
    /// Create a new [`ConstInt`] with a given width and unsigned value
    pub fn new_u(log_width: u8, value: u64) -> Result<Self, ConstTypeError> {
        if !is_valid_log_width(log_width) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid integer width.".to_owned()),
            ));
        }
        if (log_width <= 5) && (value >= (1u64 << (1u8 << log_width))) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid unsigned integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { log_width, value })
    }

    /// Create a new [`ConstInt`] with a given width and signed value
    pub fn new_s(log_width: u8, value: i64) -> Result<Self, ConstTypeError> {
        if !is_valid_log_width(log_width) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid integer width.".to_owned()),
            ));
        }
        let width = 1u8 << log_width;
        if (log_width <= 5) && (value >= (1i64 << (width - 1)) || value < -(1i64 << (width - 1))) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid signed integer value.".to_owned(),
                ),
            ));
        }

        Ok(Self {
            log_width,
            value: (if value >= 0 || log_width == LOG_WIDTH_MAX {
                value
            } else {
                value + (1i64 << width)
            }) as u64,
        })
    }

    /// Returns the number of bits of the constant
    pub fn log_width(&self) -> u8 {
        self.log_width
    }

    /// Returns the value of the constant as an unsigned integer
    pub fn value_u(&self) -> u64 {
        self.value
    }

    /// Returns the value of the constant as a signed integer
    pub fn value_s(&self) -> i64 {
        if self.log_width == LOG_WIDTH_MAX {
            self.value as i64
        } else {
            let width = 1u8 << self.log_width;
            if (self.value << 1 >> width) == 0 {
                self.value as i64
            } else {
                self.value as i64 - (1i64 << width)
            }
        }
    }
}

impl ConstIntU {
    /// Create a new [`ConstIntU`]
    pub fn new(log_width: u8, value: u64) -> Result<Self, ConstTypeError> {
        if !is_valid_log_width(log_width) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid integer width.".to_owned()),
            ));
        }
        if (log_width <= 5) && (value >= (1u64 << (1u8 << log_width))) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid unsigned integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { log_width, value })
    }

    /// Returns the value of the constant
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Returns the number of bits of the constant
    pub fn log_width(&self) -> u8 {
        self.log_width
    }
}

impl ConstIntS {
    /// Create a new [`ConstIntS`]
    pub fn new(log_width: u8, value: i64) -> Result<Self, ConstTypeError> {
        if !is_valid_log_width(log_width) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid integer width.".to_owned()),
            ));
        }
        let width = 1u8 << log_width;
        if (log_width <= 5) && (value >= (1i64 << (width - 1)) || value < -(1i64 << (width - 1))) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid signed integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { log_width, value })
    }

    /// Returns the value of the constant
    pub fn value(&self) -> i64 {
        self.value
    }

    /// Returns the number of bits of the constant
    pub fn log_width(&self) -> u8 {
        self.log_width
    }
}

#[typetag::serde]
impl CustomConst for ConstInt {
    fn name(&self) -> SmolStr {
        format!("u{}({})", self.log_width, self.value).into()
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&EXTENSION_ID)
    }

    fn get_type(&self) -> Type {
        int_type(type_arg(self.log_width))
    }
}

#[typetag::serde]
impl CustomConst for ConstIntU {
    fn name(&self) -> SmolStr {
        format!("u{}({})", self.log_width, self.value).into()
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&EXTENSION_ID)
    }

    fn get_type(&self) -> Type {
        int_type(type_arg(self.log_width))
    }
}

#[typetag::serde]
impl CustomConst for ConstIntS {
    fn name(&self) -> SmolStr {
        format!("i{}({})", self.log_width, self.value).into()
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&EXTENSION_ID)
    }

    fn get_type(&self) -> Type {
        int_type(type_arg(self.log_width))
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
            TypeBound::Eq.into(),
        )
        .unwrap();

    extension
}

lazy_static! {
    /// Lazy reference to int types extension.
    pub static ref EXTENSION: Extension = extension();
}

/// get an integer type with width corresponding to a type variable with id `var_id`
pub(super) fn int_tv(var_id: usize) -> Type {
    Type::new_extension(
        EXTENSION
            .get_type(&INT_TYPE_ID)
            .unwrap()
            .instantiate(vec![TypeArg::new_var_use(var_id, LOG_WIDTH_TYPE_PARAM)])
            .unwrap(),
    )
}
#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use super::*;

    #[test]
    fn test_int_types_extension() {
        let r = extension();
        assert_eq!(r.name() as &str, "arithmetic.int.types");
        assert_eq!(r.types().count(), 1);
        assert_eq!(r.operations().count(), 0);
    }

    #[test]
    fn test_int_widths() {
        let type_arg_32 = TypeArg::BoundedNat { n: 5 };
        assert_matches!(get_log_width(&type_arg_32), Ok(5));

        let type_arg_128 = TypeArg::BoundedNat { n: 7 };
        assert_matches!(
            get_log_width(&type_arg_128),
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
        assert!(ConstIntS::new(3, -128).is_ok());

        let const_u32_7 = const_u32_7.unwrap();
        assert!(const_u32_7.equal_consts(&ConstIntU::new(5, 7).unwrap()));
        assert_eq!(const_u32_7.log_width(), 5);
        assert_eq!(const_u32_7.value(), 7);
        assert!(const_u32_7.validate().is_ok());

        assert_eq!(const_u32_7.name(), "u5(7)");

        let const_i32_2 = ConstIntS::new(5, -2).unwrap();
        assert!(const_i32_2.equal_consts(&ConstIntS::new(5, -2).unwrap()));
        assert_eq!(const_i32_2.log_width(), 5);
        assert_eq!(const_i32_2.value(), -2);
        assert!(const_i32_2.validate().is_ok());
        assert_eq!(const_i32_2.name(), "i5(-2)");

        ConstIntS::new(50, -2).unwrap_err();
        ConstIntU::new(50, 2).unwrap_err();
    }
}
