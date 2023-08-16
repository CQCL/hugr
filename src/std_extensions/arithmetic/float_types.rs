//! Basic floating-point types

use smol_str::SmolStr;

use crate::{
    types::{CustomType, Type, TypeBound},
    Extension,
};

/// The extension identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("arithmetic.float.types");

/// Identfier for the 64-bit IEEE 754-2019 floating-point type.
const FLOAT_TYPE_ID: SmolStr = SmolStr::new_inline("float64");

/// 64-bit IEEE 754-2019 floating-point type
pub fn float64_type() -> Type {
    Type::new_extension(CustomType::new(
        FLOAT_TYPE_ID,
        [],
        RESOURCE_ID,
        TypeBound::Copyable,
    ))
}

/// Extension for basic floating-point types.
pub fn extension() -> Extension {
    let mut extension = Extension::new(RESOURCE_ID);

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
}
