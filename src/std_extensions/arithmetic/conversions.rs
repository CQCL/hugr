//! Conversions between integer and floating-point values.

use std::collections::HashSet;

use smol_str::SmolStr;

use crate::{
    extension::{ExtensionSet, SignatureError},
    type_row,
    types::{
        type_param::{TypeArg, TypeParam},
        Type, TypeRow,
    },
    utils::collect_array,
    Extension,
};

use super::float_types::FLOAT64_TYPE;
use super::int_types::int_type;

/// The extension identifier.
pub const EXTENSION_ID: SmolStr = SmolStr::new_inline("arithmetic.conversions");

fn ftoi_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ExtensionSet), SignatureError> {
    let [arg] = collect_array(arg_values);
    Ok((
        type_row![FLOAT64_TYPE],
        vec![Type::new_sum(vec![
            int_type(arg.clone()),
            crate::extension::prelude::ERROR_TYPE,
        ])]
        .into(),
        ExtensionSet::default(),
    ))
}

fn itof_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ExtensionSet), SignatureError> {
    let [arg] = collect_array(arg_values);
    Ok((
        vec![int_type(arg.clone())].into(),
        type_row![FLOAT64_TYPE],
        ExtensionSet::default(),
    ))
}

/// Extension for basic arithmetic operations.
pub fn extension() -> Extension {
    let mut extension = Extension::new_with_reqs(
        EXTENSION_ID,
        ExtensionSet::new_from_extensions(HashSet::from_iter(vec![
            super::int_types::EXTENSION_ID,
            super::float_types::EXTENSION_ID,
        ])),
    );

    extension
        .add_op_custom_sig_simple(
            "trunc_u".into(),
            "float to unsigned int".to_owned(),
            vec![TypeParam::max_usize()],
            ftoi_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "trunc_s".into(),
            "float to signed int".to_owned(),
            vec![TypeParam::max_usize()],
            ftoi_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "convert_u".into(),
            "unsigned int to float".to_owned(),
            vec![TypeParam::max_usize()],
            itof_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "convert_s".into(),
            "signed int to float".to_owned(),
            vec![TypeParam::max_usize()],
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
        assert_eq!(r.name(), "arithmetic.conversions");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with("convert") || name.starts_with("trunc"));
        }
    }
}
