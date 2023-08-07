//! Basic floating-point types

use smol_str::SmolStr;

use crate::{
    types::{CustomType, SimpleType, TypeTag},
    Resource,
};

/// The resource identifier.
pub const fn resource_id() -> SmolStr {
    SmolStr::new_inline("arithmetic.float.types")
}

/// Identfier for the 64-bit IEEE 754-2019 floating-point type.
const FLOAT_TYPE_ID: SmolStr = SmolStr::new_inline("float64");

/// 64-bit IEEE 754-2019 floating-point type
pub fn float64_type() -> SimpleType {
    CustomType::new(FLOAT_TYPE_ID, [], resource_id(), TypeTag::Classic).into()
}

/// Resource for basic floating-point types.
pub fn resource() -> Resource {
    let mut resource = Resource::new(resource_id());

    resource
        .add_type(
            FLOAT_TYPE_ID,
            vec![],
            "64-bit IEEE 754-2019 floating-point value".to_owned(),
            TypeTag::Classic.into(),
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
}
