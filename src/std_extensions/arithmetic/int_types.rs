//! Basic integer types

use smol_str::SmolStr;

use crate::{
    extension::SignatureError,
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        CustomType, Type, TypeBound,
    },
    Extension,
};

/// The extension identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("arithmetic.int.types");

/// Identfier for the integer type.
const INT_TYPE_ID: SmolStr = SmolStr::new_inline("int");

/// Integer type of a given bit width.
/// Depending on the operation, the semantic interpretation may be unsigned integer, signed integer
/// or bit string.
pub fn int_type(n: u8) -> Type {
    Type::new_extension(CustomType::new(
        INT_TYPE_ID,
        [TypeArg::USize(n as u64)],
        RESOURCE_ID,
        TypeBound::Copyable,
    ))
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
    if (n != 1)
        && (n != 2)
        && (n != 4)
        && (n != 8)
        && (n != 16)
        && (n != 32)
        && (n != 64)
        && (n != 128)
    {
        return Err(TypeArgError::InvalidValue(arg.clone()).into());
    }
    Ok(n)
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
}
