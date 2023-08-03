//! Basic logical operations.

use std::collections::HashMap;

use itertools::Itertools;
use smol_str::SmolStr;

use crate::{
    ops::constant::HugrIntValueStore,
    resource::ResourceSet,
    types::{
        type_param::{ArgValue, TypeArg, TypeParam},
        HashableType, SimpleType,
    },
    values::HashableValue,
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
    const H_INT: TypeParam = TypeParam::Value(HashableType::Int(8));
    let mut resource = Resource::new(resource_id());

    resource
        .add_op_custom_sig(
            "Not".into(),
            "logical 'not'".into(),
            vec![],
            HashMap::default(),
            Vec::new(),
            |_arg_values: &[TypeArg]| {
                Ok((
                    vec![bool_type()].into(),
                    vec![bool_type()].into(),
                    ResourceSet::default(),
                ))
            },
        )
        .unwrap();

    resource
        .add_op_custom_sig(
            "And".into(),
            "logical 'and'".into(),
            vec![H_INT],
            HashMap::default(),
            Vec::new(),
            |arg_values: &[TypeArg]| {
                let a = arg_values.iter().exactly_one().unwrap();
                let n = get_n(a);
                Ok((
                    vec![bool_type(); n as usize].into(),
                    vec![bool_type()].into(),
                    ResourceSet::default(),
                ))
            },
        )
        .unwrap();

    resource
        .add_op_custom_sig(
            "Or".into(),
            "logical 'or'".into(),
            vec![H_INT],
            HashMap::default(),
            Vec::new(),
            |arg_values: &[TypeArg]| {
                let a = arg_values.iter().exactly_one().unwrap();
                let n = get_n(a);
                Ok((
                    vec![bool_type(); n as usize].into(),
                    vec![bool_type()].into(),
                    ResourceSet::default(),
                ))
            },
        )
        .unwrap();

    resource
}

fn get_n(a: &TypeArg) -> HugrIntValueStore {
    match a {
        TypeArg::Value(ArgValue::Hashable(HashableValue::Int(n))) => *n,
        _ => panic!("Type arg should be checked before this.,"),
    }
}

#[cfg(test)]
mod test {
    use crate::Resource;

    use super::resource;

    #[test]
    fn test_logic_resource() {
        let r: Resource = resource();
        assert_eq!(r.name(), "Logic");
        assert_eq!(r.operations().count(), 3);
    }
}
