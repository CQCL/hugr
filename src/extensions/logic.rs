//! Basic logical operations.

use itertools::Itertools;
use smol_str::SmolStr;

use crate::{
    resource::ResourceSet,
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        Type,
    },
    Resource,
};

/// The resource identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("logic");

/// Construct a boolean type.
pub fn bool_type() -> Type {
    Type::new_simple_predicate(2)
}

/// Resource for basic logical operations.
pub fn resource() -> Resource {
    const H_INT: TypeParam = TypeParam::USize;
    let mut resource = Resource::new(RESOURCE_ID);

    resource
        .add_op_custom_sig_simple(
            "Not".into(),
            "logical 'not'".into(),
            vec![],
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
        .add_op_custom_sig_simple(
            "And".into(),
            "logical 'and'".into(),
            vec![H_INT],
            |arg_values: &[TypeArg]| {
                let a = arg_values.iter().exactly_one().unwrap();
                let n: u64 = match a {
                    TypeArg::USize(n) => *n,
                    _ => {
                        return Err(TypeArgError::TypeMismatch {
                            arg: a.clone(),
                            param: H_INT,
                        }
                        .into());
                    }
                };
                Ok((
                    vec![bool_type(); n as usize].into(),
                    vec![bool_type()].into(),
                    ResourceSet::default(),
                ))
            },
        )
        .unwrap();

    resource
        .add_op_custom_sig_simple(
            "Or".into(),
            "logical 'or'".into(),
            vec![H_INT],
            |arg_values: &[TypeArg]| {
                let a = arg_values.iter().exactly_one().unwrap();
                let n: u64 = match a {
                    TypeArg::USize(n) => *n,
                    _ => {
                        return Err(TypeArgError::TypeMismatch {
                            arg: a.clone(),
                            param: H_INT,
                        }
                        .into());
                    }
                };
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

#[cfg(test)]
mod test {
    use crate::Resource;

    use super::resource;

    #[test]
    fn test_logic_resource() {
        let r: Resource = resource();
        assert_eq!(r.name(), "logic");
        assert_eq!(r.operations().count(), 3);
    }
}
