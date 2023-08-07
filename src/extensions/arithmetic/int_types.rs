//! Basic integer types

use smol_str::SmolStr;

use crate::{
    resource::SignatureError,
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        CustomType, HashableType, SimpleType, TypeTag,
    },
    Resource,
};

/// The resource identifier.
pub const fn resource_id() -> SmolStr {
    SmolStr::new_inline("arithmetic.int.types")
}

/// Parameter for an integer type or operation representing the bit width.
/// Allowed values are: 1, 2, 4, 8, 16, 32, 64, 128.
pub const INT_PARAM: TypeParam = TypeParam::Value(HashableType::Int(8));

/// Identfier for the integer type.
const INT_TYPE_ID: SmolStr = SmolStr::new_inline("int");

/// Integer type of a given bit width.
/// Depending on the operation, the semantic interpretation may be unsigned integer, signed integer
/// or bit string.
pub fn int_type(n: u8) -> SimpleType {
    CustomType::new(
        INT_TYPE_ID,
        [TypeArg::Int(n as u128)],
        resource_id(),
        TypeTag::Classic,
    )
    .into()
}

/// Get the bit width of the specified integer type, or error if the width is not supported.
pub fn get_width(arg: &TypeArg) -> Result<u8, SignatureError> {
    let n: u8 = match arg {
        TypeArg::Int(n) => *n as u8,
        _ => {
            return Err(TypeArgError::TypeMismatch(arg.clone(), INT_PARAM).into());
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

/// Resource for basic integer types.
pub fn resource() -> Resource {
    let mut resource = Resource::new(resource_id());

    resource
        .add_type(
            INT_TYPE_ID,
            vec![INT_PARAM],
            "integral value of a given bit width".to_owned(),
            TypeTag::Classic.into(),
        )
        .unwrap();

    resource
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_int_types_resource() {
        let r = resource();
        assert_eq!(r.name(), "arithmetic.int.types");
        assert_eq!(r.types().count(), 1);
        assert_eq!(r.operations().count(), 0);
    }
}
