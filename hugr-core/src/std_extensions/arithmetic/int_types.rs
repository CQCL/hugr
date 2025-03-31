//! Basic integer types

use std::num::NonZeroU64;
use std::sync::{Arc, Weak};

use crate::ops::constant::ValueName;
use crate::types::TypeName;
use crate::{
    extension::ExtensionId,
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
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// Identifier for the integer type.
pub const INT_TYPE_ID: TypeName = TypeName::new_inline("int");

/// Integer type of a given bit width (specified by the TypeArg).  Depending on
/// the operation, the semantic interpretation may be unsigned integer, signed
/// integer or bit string.
pub fn int_custom_type(
    width_arg: impl Into<TypeArg>,
    extension_ref: &Weak<Extension>,
) -> CustomType {
    CustomType::new(
        INT_TYPE_ID,
        [width_arg.into()],
        EXTENSION_ID,
        TypeBound::Copyable,
        extension_ref,
    )
}

/// Integer type of a given bit width (specified by the TypeArg).
///
/// Constructed from [int_custom_type].
pub fn int_type(width_arg: impl Into<TypeArg>) -> Type {
    int_custom_type(width_arg.into(), &Arc::<Extension>::downgrade(&EXTENSION)).into()
}

lazy_static! {
    /// Array of valid integer types, indexed by log width of the integer.
    pub static ref INT_TYPES: [Type; LOG_WIDTH_BOUND as usize] = (0..LOG_WIDTH_BOUND)
        .map(|i| int_type(TypeArg::BoundedNat { n: i as u64 }))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
}

/// Returns whether `n` is a valid `log_width` for an [int_type].
pub const fn is_valid_log_width(n: u8) -> bool {
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
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ConstInt {
    log_width: u8,
    // We always use a u64 for the value. The interpretation is:
    // - as an unsigned integer, (value mod 2^N);
    // - as a signed integer, (value mod 2^(N-1) - 2^(N-1)*a)
    // where N = 2^log_width and a is the (N-1)th bit of x (counting from
    // 0 = least significant bit).
    value: u64,
}

impl ConstInt {
    /// Name of the constructor for creating constant integers.
    #[cfg_attr(not(feature = "model_unstable"), allow(dead_code))]
    pub(crate) const CTR_NAME: &'static str = "arithmetic.int.const";

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
            if ((self.value << 1) >> width) == 0 {
                self.value as i64
            } else {
                self.value as i64 - (1i64 << width)
            }
        }
    }
}

#[typetag::serde]
impl CustomConst for ConstInt {
    fn name(&self) -> ValueName {
        format!("u{}({})", 1u8 << self.log_width, self.value).into()
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn get_type(&self) -> Type {
        int_type(type_arg(self.log_width))
    }
}

/// Extension for basic integer types.
fn extension() -> Arc<Extension> {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        extension
            .add_type(
                INT_TYPE_ID,
                vec![LOG_WIDTH_TYPE_PARAM],
                "integral value of a given bit width".to_owned(),
                TypeBound::Copyable.into(),
                extension_ref,
            )
            .unwrap();
    })
}

lazy_static! {
    /// Lazy reference to int types extension.
    pub static ref EXTENSION: Arc<Extension> = extension();
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
        let const_u32_7 = ConstInt::new_u(5, 7);
        let const_u64_7 = ConstInt::new_u(6, 7);
        let const_u32_8 = ConstInt::new_u(5, 8);
        assert_ne!(const_u32_7, const_u64_7);
        assert_ne!(const_u32_7, const_u32_8);
        assert_eq!(const_u32_7, ConstInt::new_u(5, 7));

        assert_matches!(
            ConstInt::new_u(3, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert_matches!(
            ConstInt::new_u(9, 256),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert_matches!(
            ConstInt::new_s(3, 128),
            Err(ConstTypeError::CustomCheckFail(_))
        );
        assert!(ConstInt::new_s(3, -128).is_ok());

        let const_u32_7 = const_u32_7.unwrap();
        assert!(const_u32_7.equal_consts(&ConstInt::new_u(5, 7).unwrap()));
        assert_eq!(const_u32_7.log_width(), 5);
        assert_eq!(const_u32_7.value_u(), 7);
        assert!(const_u32_7.validate().is_ok());

        assert_eq!(const_u32_7.name(), "u32(7)");

        let const_i32_2 = ConstInt::new_s(5, -2).unwrap();
        assert!(const_i32_2.equal_consts(&ConstInt::new_s(5, -2).unwrap()));
        assert_eq!(const_i32_2.log_width(), 5);
        assert_eq!(const_i32_2.value_s(), -2);
        assert!(const_i32_2.validate().is_ok());
        assert_eq!(const_i32_2.name(), "u32(4294967294)");

        ConstInt::new_s(50, -2).unwrap_err();
        ConstInt::new_u(50, 2).unwrap_err();
    }

    mod proptest {
        use super::{ConstInt, LOG_WIDTH_MAX};
        use ::proptest::prelude::*;
        use i64;
        impl Arbitrary for ConstInt {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                let signed_strat = any_signed_int_with_log_width().prop_map(|(log_width, v)| {
                    ConstInt::new_s(log_width, v).expect("guaranteed to be in bounds")
                });
                let unsigned_strat = (..=LOG_WIDTH_MAX).prop_flat_map(|log_width| {
                    (0..2u64.pow(log_width as u32)).prop_map(move |v| {
                        ConstInt::new_u(log_width, v).expect("guaranteed to be in bounds")
                    })
                });

                prop_oneof![unsigned_strat, signed_strat].boxed()
            }
        }

        fn any_signed_int_with_log_width() -> impl Strategy<Value = (u8, i64)> {
            (..=LOG_WIDTH_MAX).prop_flat_map(|log_width| {
                let width = 2u64.pow(log_width as u32);
                let max_val = ((1u64 << (width - 1)) - 1u64) as i64;
                let min_val = -max_val - 1;
                prop_oneof![(min_val..=max_val), Just(min_val), Just(max_val)]
                    .prop_map(move |x| (log_width, x))
            })
        }

        proptest! {
            #[test]
            fn valid_signed_int((log_width, x) in any_signed_int_with_log_width()) {
                let (min,max) = match log_width {
                    0 => (-1, 0),
                    1 => (-2, 1),
                    2 => (-8, 7),
                    3 => (i8::MIN as i64, i8::MAX as i64),
                    4 => (i16::MIN as i64, i16::MAX as i64),
                    5 => (i32::MIN as i64, i32::MAX as i64),
                    6 => (i64::MIN, i64::MAX),
                    _ => unreachable!(),
                };
                let width = 2i64.pow(log_width as u32);
                // the left hand side counts the number of valid values as follows:
                //  - use i128 to be able to hold the number of valid i64s
                //  - there are exactly `max` valid positive values;
                //  - there are exactly `-min` valid negative values;
                //  - there are exactly 1 zero values.
                prop_assert_eq!((max as i128) - (min as i128) + 1, 1 << width);
                prop_assert!(x >= min);
                prop_assert!(x <= max);
                prop_assert!(ConstInt::new_s(log_width, x).is_ok())
            }
        }
    }
}
