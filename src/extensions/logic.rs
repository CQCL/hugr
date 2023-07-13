//! Basic logical operations.

use std::collections::HashMap;

use itertools::Itertools;
use smol_str::SmolStr;

use crate::{
    resource::{OpDef, ResourceSet, SignatureError},
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        SimpleType,
    },
    Resource,
};

/// The resource identifier.
pub const fn resource_id() -> SmolStr {
    SmolStr::new_inline("Logic")
}

/// Construct a boolean type.
pub fn bool_type() -> SimpleType {
    SimpleType::new_simple_predicate(2)
}

/// Resource for basic logical operations.
pub fn resource() -> Resource {
    let mut resource = Resource::new(resource_id());

    let not_op = OpDef::new_with_custom_sig(
        "Not".into(),
        "logical 'not'".into(),
        vec![],
        HashMap::default(),
        |_arg_values: &[TypeArg]| {
            Ok((
                vec![bool_type()].into(),
                vec![bool_type()].into(),
                ResourceSet::default(),
            ))
        },
    );

    let and_op = OpDef::new_with_custom_sig(
        "And".into(),
        "logical 'and'".into(),
        vec![TypeParam::Int],
        HashMap::default(),
        |arg_values: &[TypeArg]| {
            let a = arg_values.iter().exactly_one().unwrap();
            let n: Result<u128, SignatureError> = match a {
                TypeArg::Int(n) => Ok(*n),
                _ => {
                    return Err(TypeArgError::TypeMismatch(a.clone(), TypeParam::Int).into());
                }
            };
            Ok((
                vec![bool_type(); n.unwrap() as usize].into(),
                vec![bool_type()].into(),
                ResourceSet::default(),
            ))
        },
    );

    let or_op = OpDef::new_with_custom_sig(
        "Or".into(),
        "logical 'or'".into(),
        vec![TypeParam::Int],
        HashMap::default(),
        |arg_values: &[TypeArg]| {
            let a = arg_values.iter().exactly_one().unwrap();
            let n: Result<u128, SignatureError> = match a {
                TypeArg::Int(n) => Ok(*n),
                _ => {
                    return Err(TypeArgError::TypeMismatch(a.clone(), TypeParam::Int).into());
                }
            };
            Ok((
                vec![bool_type(); n.unwrap() as usize].into(),
                vec![bool_type()].into(),
                ResourceSet::default(),
            ))
        },
    );

    resource.add_op(not_op).unwrap();
    resource.add_op(and_op).unwrap();
    resource.add_op(or_op).unwrap();

    resource
}

#[cfg(test)]
mod test {
    use crate::Resource;

    use super::resource;

    #[test]
    fn test_logic_resource() {
        let r: Resource = resource();
        assert_eq!(r.name(), "Logic");
        let ops = r.operations();
        assert_eq!(ops.len(), 3);
    }
}
