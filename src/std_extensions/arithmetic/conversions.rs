//! Conversions between integer and floating-point values.

use crate::{
    extension::{
        prelude::sum_with_error, ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError,
        PRELUDE,
    },
    type_row,
    types::{FunctionType, PolyFuncType, Type},
    Extension,
};

use super::int_types::{int_type_var, INT_TYPE_ID};
use super::{float_types::FLOAT64_TYPE, int_types::LOG_WIDTH_TYPE_PARAM};

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.conversions");

fn ftoi_sig(
    int_type_var: Type,
    temp_reg: &ExtensionRegistry,
) -> Result<PolyFuncType, SignatureError> {
    let body = FunctionType::new(type_row![FLOAT64_TYPE], vec![sum_with_error(int_type_var)]);

    PolyFuncType::new_validated(vec![LOG_WIDTH_TYPE_PARAM], body, temp_reg)
}

fn itof_sig(
    int_type_var: Type,
    temp_reg: &ExtensionRegistry,
) -> Result<PolyFuncType, SignatureError> {
    let body = FunctionType::new(vec![int_type_var], type_row![FLOAT64_TYPE]);

    PolyFuncType::new_validated(vec![LOG_WIDTH_TYPE_PARAM], body, temp_reg)
}

/// Extension for basic arithmetic operations.
pub fn extension() -> Extension {
    let mut extension = Extension::new_with_reqs(
        EXTENSION_ID,
        ExtensionSet::from_iter(vec![
            super::int_types::EXTENSION_ID,
            super::float_types::EXTENSION_ID,
        ]),
    );
    let int_types_extension = super::int_types::extension();
    let int_type_def = int_types_extension.get_type(&INT_TYPE_ID).unwrap();
    let int_type_var = int_type_var(0, int_type_def).unwrap();
    let temp_reg: ExtensionRegistry = [
        extension.clone(),
        int_types_extension,
        super::float_types::extension(),
        PRELUDE.to_owned(),
    ]
    .into();
    extension
        .add_op_type_scheme_simple(
            "trunc_u".into(),
            "float to unsigned int".to_owned(),
            ftoi_sig(int_type_var.clone(), &temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "trunc_s".into(),
            "float to signed int".to_owned(),
            ftoi_sig(int_type_var.clone(), &temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "convert_u".into(),
            "unsigned int to float".to_owned(),
            itof_sig(int_type_var.clone(), &temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "convert_s".into(),
            "signed int to float".to_owned(),
            itof_sig(int_type_var, &temp_reg).unwrap(),
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
