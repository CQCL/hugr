//! Conversions between integer and floating-point values.

use crate::{
    extension::{prelude::sum_with_error, ExtensionId, ExtensionSet},
    type_row,
    types::{FunctionType, PolyFuncType},
    Extension,
};

use super::int_types::int_type_var;
use super::{float_types::FLOAT64_TYPE, int_types::LOG_WIDTH_TYPE_PARAM};

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.conversions");

/// Extension for basic arithmetic operations.
pub fn extension() -> Extension {
    let ftoi_sig = PolyFuncType::new(
        vec![LOG_WIDTH_TYPE_PARAM],
        FunctionType::new(
            type_row![FLOAT64_TYPE],
            vec![sum_with_error(int_type_var(0))],
        ),
    );

    let itof_sig = PolyFuncType::new(
        vec![LOG_WIDTH_TYPE_PARAM],
        FunctionType::new(vec![int_type_var(0)], type_row![FLOAT64_TYPE]),
    );

    let mut extension = Extension::new_with_reqs(
        EXTENSION_ID,
        ExtensionSet::from_iter(vec![
            super::int_types::EXTENSION_ID,
            super::float_types::EXTENSION_ID,
        ]),
    );
    extension
        .add_op_simple(
            "trunc_u".into(),
            "float to unsigned int".to_owned(),
            ftoi_sig.clone(),
        )
        .unwrap();
    extension
        .add_op_simple("trunc_s".into(), "float to signed int".to_owned(), ftoi_sig)
        .unwrap();
    extension
        .add_op_simple(
            "convert_u".into(),
            "unsigned int to float".to_owned(),
            itof_sig.clone(),
        )
        .unwrap();
    extension
        .add_op_simple(
            "convert_s".into(),
            "signed int to float".to_owned(),
            itof_sig,
        )
        .unwrap();

    extension
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_conversions_extension() {
        let r = extension();
        assert_eq!(r.name() as &str, "arithmetic.conversions");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with("convert") || name.starts_with("trunc"));
        }
    }
}
