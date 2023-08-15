//! Conversions between integer and floating-point values.

use std::collections::HashSet;

use smol_str::SmolStr;

use crate::{
    resource::{ResourceSet, SignatureError},
    types::{
        type_param::{TypeArg, TypeParam},
        Type, TypeRow,
    },
    utils::collect_array,
    Resource,
};

use super::float_types::float64_type;
use super::int_types::{get_width, int_type};

/// The resource identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("arithmetic.conversions");

fn ftoi_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg] = collect_array(arg_values);
    let n: u8 = get_width(arg)?;
    Ok((
        vec![float64_type()].into(),
        vec![Type::new_sum(vec![
            int_type(n),
            crate::resource::prelude::ERROR_TYPE,
        ])]
        .into(),
        ResourceSet::default(),
    ))
}

fn itof_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg] = collect_array(arg_values);
    let n: u8 = get_width(arg)?;
    Ok((
        vec![int_type(n)].into(),
        vec![float64_type()].into(),
        ResourceSet::default(),
    ))
}

/// Resource for basic arithmetic operations.
pub fn resource() -> Resource {
    let mut resource = Resource::new_with_reqs(
        RESOURCE_ID,
        ResourceSet::new_from_resources(HashSet::from_iter(vec![
            super::int_types::RESOURCE_ID,
            super::float_types::RESOURCE_ID,
        ])),
    );

    resource
        .add_op_custom_sig_simple(
            "trunc_u".into(),
            "float to unsigned int".to_owned(),
            vec![TypeParam::USize],
            ftoi_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "trunc_s".into(),
            "float to signed int".to_owned(),
            vec![TypeParam::USize],
            ftoi_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "convert_u".into(),
            "unsigned int to float".to_owned(),
            vec![TypeParam::USize],
            itof_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "convert_s".into(),
            "signed int to float".to_owned(),
            vec![TypeParam::USize],
            itof_sig,
        )
        .unwrap();

    resource
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_conversions_resource() {
        let r = resource();
        assert_eq!(r.name(), "arithmetic.conversions");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with("convert") || name.starts_with("trunc"));
        }
    }
}
